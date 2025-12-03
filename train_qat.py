from typing import Any


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import copy

# 导入上面定义的 QAT 模型
from net_model_qat import Litenet_QAT

# ================= 配置 =================
FLOAT_MODEL_PATH = "pth/best_model_distill.pth"
QAT_CHECKPOINT_DIR = "qat_checkpoints"
DATA_DIR = r"D:\study\CNN_demo\Litenet\dataset_v5\train" # 用训练集微调
VALID_DIR = r"D:\study\CNN_demo\Litenet\dataset_v5\valid"
NUM_CLASSES = 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ========================================

def main_qat():
    os.makedirs(QAT_CHECKPOINT_DIR, exist_ok=True)

    # 1. 数据准备
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(datasets.ImageFolder(DATA_DIR, transform=transform), 
                              batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader[Any](datasets.ImageFolder(VALID_DIR, transform=transform), 
                              batch_size=32, shuffle=False)

    # 2. 初始化 QAT 模型并加载浮点权重
    print("[-] 构建 QAT 模型...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 加载你之前训练好的 best_model_distill.pth
    # 注意：由于 Litenet_QAT 结构为了量化微调了 (如 nn.ReLU), load_state_dict 可能需要 strict=False
    ckpt = torch.load(FLOAT_MODEL_PATH, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # 过滤掉一些不匹配的键（例如原来的 F.relu 没有参数，新的有）
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    # 3. 融合模块 (Fusion)
    model.eval()
    model.fuse_model()
    print("[-] 模块融合完成 (Conv+BN+ReLU)")

    # 4. 配置 QAT
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    print("[-] QAT 准备就绪")

    # 5. 微调训练 (Fine-tuning)
    # QAT 不需要训练很久，通常几个 Epoch 就够了，学习率要低
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("[-] 开始微调训练...")
    model.train() # 必须切换到 train 模式，观察者才能收集数据分布
    
    for epoch in range(1, 4): # 训练 3 个 Epoch 即可
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if i % 10 == 0:
                print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

        # 在每个 Epoch 结束后，冻结 BN 统计数据再训练一小会儿通常更好，这里简化处理
        # 验证一下当前的量化精度模拟
        model.eval()
        # 注意：这里验证的是“模拟量化”的精度，还不是真正的 INT8 推理
        acc = validate(model, valid_loader, DEVICE)
        print(f"Epoch {epoch} QAT 模拟准确率: {acc:.2f}%")
        model.train()

    # 6. 转换为真正的 INT8 模型 (CPU Only)
    # PyTorch 目前只能在 CPU 上运行转换后的 INT8 模型
    print("[-] 正在转换为 INT8 模型...")
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)

    # 7. 保存
    save_path = os.path.join(QAT_CHECKPOINT_DIR, "litenet_int8_qat.pth")
    torch.save(quantized_model.state_dict(), save_path)
    print(f"[-] 量化模型已保存: {save_path}")

def validate(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    main_qat()