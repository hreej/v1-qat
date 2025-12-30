"""
float2half.py
将训练好的 FP32 模型转换为 FP16 (Half Precision) 模型。
这可以减少模型大小（约减半）并加速支持 FP16 的硬件（如 Tensor Cores）上的推理。
"""

import torch
import torch.nn as nn
import os
import argparse
import sys
from net_model import Litenet

# ==================== 配置 ====================
class Config:
    # 输入 FP32 模型路径
    FLOAT_MODEL_PATH = "quant_results/float2half/float_pth/best_model_distill.pth"
    # 输出 FP16 模型路径
    HALF_MODEL_PATH = "quant_results/float2half/half_pth/litenet_fp16.pth"

def convert_to_fp16(checkpoint_path, output_path):
    print("=" * 60)
    print("FP32 -> FP16 模型转换工具")
    print("=" * 60)
    
    # 1. 检查输入文件
    if not os.path.exists(checkpoint_path):
        print(f"[Error] 输入模型文件不存在: {checkpoint_path}")
        return

    print(f"[-] 正在加载模型: {checkpoint_path}")
    
    # 2. 实例化模型结构
    # 注意：这里假设 num_classes=12，如果不同需要修改
    try:
        model = Litenet(num_classes=12)
    except Exception as e:
        print(f"[Error] 模型实例化失败: {e}")
        return
    
    # 3. 加载权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理 checkpoint 字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("    检测到 checkpoint 包含元数据，正在提取 model_state_dict...")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("[+] 模型权重加载成功 (FP32)")
    except Exception as e:
        print(f"[Error] 权重加载失败: {e}")
        return

    # 4. 转换为 FP16
    print("[-] 正在转换为 FP16 (Half Precision)...")
    model.half()
    
    # 5. 保存 FP16 模型
    # 注意：保存的是 state_dict，其中的 tensor 已经是 half 类型
    print(f"[-] 正在保存 FP16 模型到: {output_path}")
    try:
        # 确保输出目录存在
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        torch.save(model.state_dict(), output_path)
        print("[+] 保存成功")
    except Exception as e:
        print(f"[Error] 保存失败: {e}")
        return
    
    # 6. 统计信息
    fp32_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "-" * 40)
    print(f"转换统计:")
    print(f"原始模型 (FP32): {fp32_size:.2f} MB")
    print(f"转换模型 (FP16): {fp16_size:.2f} MB")
    print(f"压缩率: {fp16_size/fp32_size*100:.2f}% (预期约 50%)")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 PyTorch 模型转换为 FP16")
    parser.add_argument('--input', type=str, default=Config.FLOAT_MODEL_PATH, help="输入 FP32 模型路径")
    parser.add_argument('--output', type=str, default=Config.HALF_MODEL_PATH, help="输出 FP16 模型路径")
    
    args = parser.parse_args()
    
    convert_to_fp16(args.input, args.output)
