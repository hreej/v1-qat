import torch
import torch.nn as nn
import time
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 导入你的 QAT 模型定义
from net_model_qat import Litenet_QAT

# ================= 配置区域 =================
# 1. 量化模型路径
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"

# 2. 原始浮点模型路径 (用于对比文件大小)
FLOAT_MODEL_PATH = "pth/best_model_distill.pth"

# 3. 验证集路径
VALID_DIR = r"D:\study\CNN_demo\Litenet\dataset_v5\valid"

# 4. 设置
NUM_CLASSES = 12
IMG_SIZE = 128
# PyTorch 的 INT8 推理目前只能在 CPU 上运行
DEVICE = torch.device("cpu") 
# ===========================================

def load_quantized_model():
    """
    加载 INT8 模型的关键步骤：
    1. 创建原始 QAT 模型
    2. 执行完全相同的融合 (Fuse)
    3. 执行 prepare_qat 配置量化参数
    4. 执行 convert 转换为量化结构 (这一步会把结构变成 QuantizedConv2d 等)
    5. 最后加载 state_dict
    """
    print(f"[-] 正在重构量化模型结构...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 步骤 1: 融合 (必须与训练时完全一致)
    model.eval()
    model.fuse_model()
    
    # 步骤 2: 配置 (必须与训练时一致，通常是 fbgemm)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 步骤 3: 转换结构 (Float -> INT8 Structure)
    model.cpu()
    # 这一步之后，模型里的 Conv2d 会变成 QuantizedConv2d
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # 步骤 4: 加载权重
    print(f"[-] 正在加载 INT8 权重: {QAT_MODEL_PATH}")
    state_dict = torch.load(QAT_MODEL_PATH, map_location="cpu")
    quantized_model.load_state_dict(state_dict)
    
    return quantized_model

def get_file_size(filepath):
    """获取文件大小 (MB)"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return 0

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    print("[-] 开始验证精度...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return 100 * correct / total

def benchmark_speed(model, iterations=500):
    """测试 CPU 单张图片推理速度"""
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # 预热
    for _ in range(20):
        _ = model(dummy_input)
        
    times = []
    print(f"[-] 开始测速 ({iterations} 次循环)...")
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            times.append((time.time() - start) * 1000)
            
    return np.mean(times)

def main():
    print("="*60)
    print("INT8 量化模型评估")
    print("="*60)
    
    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=VALID_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. 加载模型
    try:
        int8_model = load_quantized_model()
    except Exception as e:
        print(f"加载失败: {e}")
        print("提示: 请确保 net_model_qat.py 中的 fuse_model 逻辑与 train_qat.py 中完全一致。")
        return

    # 3. 对比文件大小
    float_size = get_file_size(FLOAT_MODEL_PATH)
    int8_size = get_file_size(QAT_MODEL_PATH)
    print(f"\n[1] 模型体积对比:")
    print(f"    浮点模型: {float_size:.2f} MB")
    print(f"    量化模型: {int8_size:.2f} MB")
    print(f"    压缩比率: {float_size / int8_size:.1f}x (通常应接近 4x)")
    
    # 4. 测试精度
    acc = evaluate_accuracy(int8_model, loader)
    print(f"\n[2] INT8 验证集精度: {acc:.2f}%")
    
    # 5. 测试速度 (CPU)
    print(f"\n[3] CPU 推理速度 (Batch=1):")
    avg_time = benchmark_speed(int8_model)
    print(f"    平均耗时: {avg_time:.4f} ms")
    print(f"    FPS:      {1000/avg_time:.1f}")

    # 6. 检查结构 (验证是否真的量化了)
    print(f"\n[4] 结构抽查 (Block1 Pointwise):")
    # 检查第一层的 pointwise 卷积是否变成了 QuantizedConvReLU2d
    print(f"    类型: {type(int8_model.block1_conv.pointwise)}")
    print(f"    (应包含 'Quantized' 字样)")

    # 保存结果到文件
    with open("eval_results.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("INT8 量化模型评估结果\n")
        f.write("="*60 + "\n")
        f.write(f"\n[1] 模型体积对比:\n")
        f.write(f"    浮点模型: {float_size:.2f} MB\n")
        f.write(f"    量化模型: {int8_size:.2f} MB\n")
        f.write(f"    压缩比率: {float_size / int8_size:.1f}x\n")
        f.write(f"\n[2] INT8 验证集精度: {acc:.2f}%\n")
        f.write(f"\n[3] CPU 推理速度 (Batch=1):\n")
        f.write(f"    平均耗时: {avg_time:.4f} ms\n")
        f.write(f"    FPS:      {1000/avg_time:.1f}\n")
        f.write(f"\n[4] 结构抽查 (Block1 Pointwise):\n")
        f.write(f"    类型: {type(int8_model.block1_conv.pointwise)}\n")
        f.write("="*60 + "\n")
    
    print(f"\n[-] 评估结果已保存至 eval_results.txt")

    print("="*60)

if __name__ == "__main__":
    main()