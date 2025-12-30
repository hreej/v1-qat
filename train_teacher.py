"""
教师网络训练脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from tqdm import tqdm
import numpy as np

# 导入网络模型和可视化工具
from net_model import Litenet
from net_model_pretrained import (LitenetResNet, LitenetEfficientNet, 
                                   LitenetDenseNet, LitenetGoogleNet,
                                   LitenetMobileNet, LitenetSqueezeNet,
                                   LitenetAlexNet)  # 预训练模型
from visualize import visualize_all


# ==================== 超参数配置 ====================
class Config:
    """统一的超参数配置类，便于修改"""
    
    # 数据路径
    DATASET_PATH = r"D:\study\CNN_demo\Litenet\dataset_v5"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VALID_DIR = os.path.join(DATASET_PATH, "valid")
    
    # 模型保存路径
    CHECKPOINT_DIR = "pre_checkpoints/squeezenet"
    INDICATOR_DIR = "pre_indicator/squeezenet"
    
    # 🔥 模型选择配置
    USE_PRETRAINED = True  # True: 使用预训练模型, False: 使用原始 Litenet
    PRETRAINED_MODEL = "squeezenet"  # 可选: "resnet18", "efficientnet_b0", "densenet121", "googlenet", "mobilenet_v2", "squeezenet", "alexnet"
    FREEZE_BACKBONE = True  # 是否冻结预训练层 (True: 只训练分类层, False: 微调整个网络)
    
    # 训练超参数
    EPOCHS = 80
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # L2正则化
    
    # 优化器参数（Adam）
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    
    # 学习率调度器参数
    LR_SCHEDULER = "CosineAnnealingLR"  # 可选: "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"
    LR_PATIENCE = 5  # ReduceLROnPlateau的耐心值，10个epoch不提升则降低学习率
    LR_FACTOR = 0.5   # 学习率衰减因子
    LR_STEP_SIZE = 30  # StepLR的步长
    
    # 数据增强参数（测试时不使用）
    IMG_SIZE = 128
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # 训练设置
    NUM_WORKERS = 8  # 数据加载线程数
    PIN_MEMORY = True  # 是否将数据固定到GPU内存
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    
    # 设备设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    NUM_CLASSES = 12  # 根据数据集类别数修改

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (list/tensor): 类别权重 (Class Weights)，用于解决样本数量不平衡。
                                 例如: [1, 2, 2, 1, ...]，少样本的类别权重给大一点。
                                 如果不设置，则传 None。
            gamma (float): 聚焦参数。Gamma 越大，模型越关注难分样本。通常取 2.0。
            reduction (str): 'mean' (默认) | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) 模型输出的 logits
        # targets: (N) 真实标签
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # p_t 是模型预测正确的概率
        
        # Focal Loss 公式: FL = - alpha * (1-pt)^gamma * log(pt)
        # 这里的 ce_loss 已经包含了 -log(pt) 和 alpha (如果设置了 weight)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def init_weights(m):
    """
    使用 Kaiming (He) 初始化权重
    适用于 ReLU 激活函数的网络
    """
    if isinstance(m, nn.Conv2d):
        # 卷积层使用 Kaiming 正态分布初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BN层的权重初始化为1，偏置初始化为0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # 全连接层使用 Kaiming 正态分布初始化
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_data_loaders(config):
    """
    创建训练集和验证集的数据加载器
    注意：数据集已经进行了离线增强，所以这里只做基本的预处理
    """
    
    # 训练集和验证集使用相同的预处理（因为已经做过离线增强）
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=transform)
    valid_dataset = datasets.ImageFolder(root=config.VALID_DIR, transform=transform)
    
    # 创建数据加载器
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
    
    # 获取类别名称
    class_names = train_dataset.classes
    
    print(f"\n数据集信息:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(valid_dataset)}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}\n")
    
    return train_loader, valid_loader, class_names


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [训练]", 
                leave=False, ncols=100)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device, epoch, config):
    """验证模型"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    # 使用tqdm显示进度条
    pbar = tqdm(valid_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [验证]", 
                leave=False, ncols=100)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 保存预测结果用于混淆矩阵
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train(config):
    """完整的训练流程"""
    
    # 创建必要的目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.INDICATOR_DIR, exist_ok=True)
    
    # 获取数据加载器
    train_loader, valid_loader, class_names = get_data_loaders(config)
    
    # 🔥 根据配置创建模型
    if config.USE_PRETRAINED:
        print(f"\n{'='*80}")
        print(f"使用预训练模型: {config.PRETRAINED_MODEL.upper()}")
        print(f"{'='*80}\n")
        
        if config.PRETRAINED_MODEL == "resnet18":
            model = LitenetResNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "efficientnet_b0":
            model = LitenetEfficientNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "densenet121":
            model = LitenetDenseNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "googlenet":
            model = LitenetGoogleNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "mobilenet_v2":
            model = LitenetMobileNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "squeezenet":
            model = LitenetSqueezeNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        elif config.PRETRAINED_MODEL == "alexnet":
            model = LitenetAlexNet(
                num_classes=config.NUM_CLASSES,
                freeze_backbone=config.FREEZE_BACKBONE
            ).to(config.DEVICE)
        else:
            raise ValueError(f"不支持的预训练模型: {config.PRETRAINED_MODEL}")
        
        # ✨ 只对新添加的分类层应用 Kaiming 初始化
        def init_new_layers(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 根据不同模型初始化分类层
        if config.PRETRAINED_MODEL == "resnet18":
            model.backbone.fc.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "efficientnet_b0":
            model.backbone.classifier.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "densenet121":
            model.backbone.classifier.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "googlenet":
            model.backbone.fc.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "mobilenet_v2":
            model.backbone.classifier.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "squeezenet":
            model.backbone.classifier.apply(init_new_layers)
        elif config.PRETRAINED_MODEL == "alexnet":
            model.backbone.classifier.apply(init_new_layers)
        
        print("✓ 已应用 Kaiming 初始化到新添加的分类层")
        
    else:
        print(f"\n{'='*80}")
        print("使用原始 Litenet 网络")
        print(f"{'='*80}\n")
        
        model = Litenet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        
        # ✨ 应用 Kaiming 初始化
        model.apply(init_weights)
        print("✓ 已应用 Kaiming (He) 权重初始化")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型信息:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"设备: {config.DEVICE}\n")

    # class_weights = torch.tensor([0.8910, 0.9653, 0.9653, 1.4515, 0.9800, 0.9142, 1.0055, 0.9049, 1.1074]).to(config.DEVICE)
    
    # 定义损失函数
    # 🔥 使用预训练模型时，推荐用普通 CE Loss
    if config.USE_PRETRAINED:
        criterion = nn.CrossEntropyLoss().to(config.DEVICE)
        print("✓ 使用 Cross Entropy Loss")
    else:
        # 原始 Litenet 使用 Focal Loss
        criterion = FocalLoss(gamma=2.0).to(config.DEVICE)
        print("✓ 使用 Focal Loss (gamma=2.0)")
    
    # 定义优化器 (Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 定义学习率调度器
    if config.LR_SCHEDULER == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config.LR_FACTOR, 
            patience=config.LR_PATIENCE
        )
    elif config.LR_SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_FACTOR
        )
    elif config.LR_SCHEDULER == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=1e-5
        )
    else:
        scheduler = None
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lr_history = []
    
    # 最佳模型记录
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 开始训练
    print("=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch, config
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model, valid_loader, criterion, config.DEVICE, epoch, config
        )
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        if scheduler is not None:
            if config.LR_SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # 计算epoch用时
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f"\nEpoch [{epoch}/{config.EPOCHS}] - 用时: {epoch_time:.2f}s")
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 每5个epoch实时更新可视化图表
        if epoch % 5 == 0:
            print(f"\n📊 正在更新可视化图表 (Epoch {epoch})...")
            from visualize import plot_training_curves, plot_confusion_matrix, plot_per_class_accuracy
            
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
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_names': class_names
            }, best_model_path)
            
            print(f"✓ 新的最佳模型已保存！验证准确率: {best_acc:.2f}%")
            
            # 保存最佳模型的预测结果(用于后续可视化)
            best_val_preds = val_preds
            best_val_labels = val_labels
        else:
            patience_counter += 1
            print(f"未改善计数: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        print("-" * 80)
        
        # 早停检查
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发！验证准确率连续{config.EARLY_STOPPING_PATIENCE}个epoch未改善。")
            break
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"总用时: {total_time / 60:.2f} 分钟")
    print(f"最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    print("=" * 80 + "\n")
    
    # 生成配置字典用于日志
    config_dict = {
        'MODEL': f"{config.PRETRAINED_MODEL.upper()} (预训练)" if config.USE_PRETRAINED else "Litenet (从头训练)",
        'FREEZE_BACKBONE': config.FREEZE_BACKBONE if config.USE_PRETRAINED else 'N/A',
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
    
    # 生成所有可视化结果
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
    
    return model, best_acc


def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 打印配置信息
    print("\n" + "=" * 80)
    print("训练配置")
    print("=" * 80)
    print(f"模型类型: {config.PRETRAINED_MODEL.upper() if config.USE_PRETRAINED else 'Litenet'}")
    if config.USE_PRETRAINED:
        print(f"预训练权重: ImageNet")
        print(f"冻结骨干网络: {'是' if config.FREEZE_BACKBONE else '否（微调整个网络）'}")
    print(f"数据集路径: {config.DATASET_PATH}")
    print(f"训练轮数: {config.EPOCHS}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"优化器: Adam")
    print(f"学习率调度器: {config.LR_SCHEDULER}")
    print(f"设备: {config.DEVICE}")
    print(f"图像尺寸: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print("=" * 80 + "\n")
    
    # 开始训练
    model, best_acc = train(config)
    
    print(f"\n✓ 训练任务完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"✓ 模型已保存至: {config.CHECKPOINT_DIR}/best_model.pth")
    print(f"✓ 可视化结果已保存至: {config.INDICATOR_DIR}/")


if __name__ == "__main__":
    main()
