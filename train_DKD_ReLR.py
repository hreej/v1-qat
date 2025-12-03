"""
train_distillation.py
知识蒸馏训练脚本: 使用 DKD (Decoupled Knowledge Distillation) 方法
教师: ResNet18 | 学生: LiteNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from tqdm import tqdm
import numpy as np

from net_model import Litenet
from net_model_pretrained import LitenetResNet
from visualize import visualize_all, plot_training_curves, plot_confusion_matrix, plot_per_class_accuracy

# ==================== 配置 ====================
class DistillConfig:
    """蒸馏训练配置"""
    
    # 数据路径
    DATASET_PATH = r"D:\study\CNN_demo\Litenet\dataset_v5"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    
    # 模型保存路径 (根据参数动态命名)
    CHECKPOINT_DIR = "distill_checkpoints/DKD_T3_A8_B2_ReLR"
    INDICATOR_DIR = "distill_indicator/DKD_T3_A8_B2_ReLR"
    
    # 教师模型路径
    TEACHER_CKPT = r"pre_checkpoints/resnet18_best_model.pth"
    
    # 训练超参数
    NUM_CLASSES = 12
    IMG_SIZE = 128
    BATCH_SIZE = 128
    EPOCHS = 180
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MIN_LR = 1e-5
    PATIENCE = 20  # 早停耐心值
    
    # 🔥 DKD 蒸馏特定参数
    # 参考文献推荐: T=4.0, alpha=1.0, beta=2.0 (针对 ResNet 架构)
    TEMPERATURE = 3.0  # 蒸馏温度
    DKD_ALPHA = 8.0    # TCKD 权重 (目标类知识)
    DKD_BETA = 2.0     # NCKD 权重 (非目标类知识)
    
    # 设备设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载设置
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # 学习率调度器
    # LR_SCHEDULER = "CosineAnnealingLR"
    # LR_SCHEDULER = "MultiStepLR"
    LR_SCHEDULER = "ReduceLROnPlateau"

# ==================== 早停机制 ====================
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# ==================== DKD 损失函数 ====================
def _get_gt_mask(logits, target):
    """辅助函数: 生成目标类别的掩码"""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    """辅助函数: 生成非目标类别的掩码"""
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    Decoupled Knowledge Distillation Loss (CVPR 2022)
    Loss = alpha * TCKD + beta * NCKD
    """
    # 获取掩码
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # 计算带温度的 Softmax
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    # ----------- TCKD (Target Class Knowledge Distillation) -----------
    # 构造二分类概率分布: [目标类概率, 非目标类概率之和]
    pred_student_tckd = torch.cat([
        pred_student.gather(1, target.unsqueeze(1)),
        (pred_student * other_mask).sum(dim=1, keepdim=True)
    ], dim=1)
    
    pred_teacher_tckd = torch.cat([
        pred_teacher.gather(1, target.unsqueeze(1)),
        (pred_teacher * other_mask).sum(dim=1, keepdim=True)
    ], dim=1)
    
    # 计算 TCKD 的 KL 散度
    log_pred_student_tckd = torch.log(pred_student_tckd + 1e-8)
    tckd_loss = F.kl_div(log_pred_student_tckd, pred_teacher_tckd, reduction='batchmean') * (temperature**2)
    
    # ----------- NCKD (Non-Target Class Knowledge Distillation) -----------
    # 构造非目标类的概率分布 (排除目标类后重新归一化)
    # 技巧: 减去大数屏蔽目标类，然后做 Softmax
    pred_student_nckd = F.softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)
    pred_teacher_nckd = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    
    # 计算 NCKD 的 KL 散度
    log_pred_student_nckd = torch.log(pred_student_nckd + 1e-8)
    nckd_loss = F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction='batchmean') * (temperature**2)
    
    return alpha * tckd_loss + beta * nckd_loss

# ==================== 数据加载 ====================
def get_data_loaders(config):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=transform)
    valid_dataset = datasets.ImageFolder(root=config.VALID_DIR, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, valid_loader, train_dataset.classes

# ==================== 训练单个Epoch ====================
def train_one_epoch_distill(student, teacher, train_loader, optimizer, device, epoch, config):
    """训练一个epoch (使用 DKD)"""
    student.train()
    teacher.eval() # 教师模型始终为评估模式
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]", leave=False, ncols=100)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 教师模型前向传播 (不计算梯度)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
            
        # 学生模型前向传播
        student_logits = student(inputs)
        
        # 1. 计算基础 CrossEntropy 损失 (Hard Label)
        loss_ce = F.cross_entropy(student_logits, labels)
        
        # 2. 计算 DKD 损失 (Soft Label)
        loss_dkd = dkd_loss(
            student_logits, 
            teacher_logits, 
            labels, 
            alpha=config.DKD_ALPHA, 
            beta=config.DKD_BETA, 
            temperature=config.TEMPERATURE
        )
        
        # 总损失
        loss = loss_ce + loss_dkd
        
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(student_logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# ==================== 验证 ====================
def validate(model, valid_loader, device, epoch, config):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(valid_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Valid]", leave=False, ncols=100)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
            
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ==================== 主训练流程 ====================
def train_distill(config):
    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.INDICATOR_DIR, exist_ok=True)
    
    # 加载数据
    train_loader, valid_loader, class_names = get_data_loaders(config)

    # 加载教师模型
    print(f"\n正在加载教师模型: {config.TEACHER_CKPT}")
    teacher = LitenetResNet(num_classes=config.NUM_CLASSES, freeze_backbone=False).to(config.DEVICE)
    
    if os.path.exists(config.TEACHER_CKPT):
        teacher_ckpt = torch.load(config.TEACHER_CKPT, map_location=config.DEVICE)
        # 处理可能存在的 'model_state_dict' 键
        if 'model_state_dict' in teacher_ckpt:
            teacher.load_state_dict(teacher_ckpt['model_state_dict'])
        else:
            teacher.load_state_dict(teacher_ckpt)
        print("✓ 教师模型加载成功")
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {config.TEACHER_CKPT}")
        
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 创建学生模型
    print("正在创建学生模型 (Litenet)...")
    student = Litenet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # 定义优化器
    optimizer = optim.Adam(student.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 定义学习率调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
    # )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10)

    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True)

    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lr_history = []
    
    best_acc = 0.0
    best_epoch = 0
    best_val_preds = []
    best_val_labels = []
    
    print("=" * 80)
    print("开始 DKD 蒸馏训练...")
    print(f"教师: ResNet18 | 学生: Litenet")
    print(f"参数: Temp={config.TEMPERATURE} | Alpha(TCKD)={config.DKD_ALPHA} | Beta(NCKD)={config.DKD_BETA}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch_distill(
            student, teacher, train_loader, optimizer, config.DEVICE, epoch, config
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            student, valid_loader, config.DEVICE, epoch, config
        )
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step(val_acc) # 传入验证准确率
        
        # 计算耗时
        epoch_time = time.time() - epoch_start_time
        
        # 打印结果
        print(f"\nEpoch [{epoch}/{config.EPOCHS}] - 用时: {epoch_time:.2f}s")
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("\n" + "=" * 30)
            print(f"早停触发! 验证集准确率在 {config.PATIENCE} 个Epoch内未提升")
            print("=" * 30 + "\n")
            break
        
        # 每5个epoch实时更新可视化图表
        if epoch % 5 == 0:
            print(f"\n📊 正在更新可视化图表 (Epoch {epoch})...")
            
            # 更新训练曲线
            plot_training_curves(
                train_losses, train_accs, val_losses, val_accs, 
                save_path=config.INDICATOR_DIR
            )
            
            # 更新混淆矩阵
            plot_confusion_matrix(
                val_labels, val_preds, class_names, 
                save_path=config.INDICATOR_DIR
            )
            
            # 更新各类识别准确率
            plot_per_class_accuracy(
                val_labels, val_preds, class_names,
                save_path=config.INDICATOR_DIR
            )
            print(f"✓ 可视化图表已更新\n")
        
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_val_preds = val_preds
            best_val_labels = val_labels
            
            torch.save({
                'epoch': epoch, 
                'model_state_dict': student.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_names': class_names
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model_distill.pth'))
            
            print(f"✓ 新的最优模型已保存! 验证准确率: {best_acc:.2f}%")
        
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("蒸馏训练完成！")
    print("=" * 80)
    print(f"总用时: {total_time / 60:.2f} 分钟")
    print(f"最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 80 + "\n")
    
    # 生成配置字典用于日志
    config_dict = {
        'MODEL': "Litenet (DKD Distilled from ResNet18)",
        'TEACHER': "ResNet18",
        'TEMPERATURE': config.TEMPERATURE,
        'DKD_ALPHA': config.DKD_ALPHA,
        'DKD_BETA': config.DKD_BETA,
        'EPOCHS': config.EPOCHS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'OPTIMIZER': 'Adam',
        'LR_SCHEDULER': config.LR_SCHEDULER,
        'IMG_SIZE': config.IMG_SIZE,
        'NUM_CLASSES': config.NUM_CLASSES,
        'DEVICE': str(config.DEVICE),
        'TOTAL_TRAINING_TIME': f'{total_time / 60:.2f} min',
        'TRAIN_SAMPLES': len(train_loader.dataset),
        'VALID_SAMPLES': len(valid_loader.dataset),
    }
    
    # 生成最终的所有可视化结果
    visualize_all(
        y_true=best_val_labels,
        y_pred=best_val_preds,
        class_names=class_names,
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        config=config_dict,
        best_acc=best_acc,
        best_epoch=best_epoch,
        lr_history=lr_history,
        save_path=config.INDICATOR_DIR
    )

if __name__ == "__main__":
    config = DistillConfig()
    train_distill(config)