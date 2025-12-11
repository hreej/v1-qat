import torch
import os
import numpy as np
import torch.nn as nn
from net_model_qat import Litenet_QAT

# ================= 配置 =================
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
EXPORT_DIR = "fpga_params"
NUM_CLASSES = 12
# ========================================

def load_quantized_model():
    """
    加载量化模型的标准流程（与 eval 脚本一致）
    """
    print(f"[-] 正在加载量化模型架构...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 1. 融合 & 准备
    model.eval()
    model.fuse_model()
    model.train() # 必须切回 train 才能 prepare
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 2. 转换 & 加载权重
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    print(f"[-] 加载权重: {QAT_MODEL_PATH}")
    state_dict = torch.load(QAT_MODEL_PATH, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    return quantized_model

def save_layer_params(layer_name, module):
    """
    提取并保存单层的参数
    """
    if hasattr(module, 'weight'):
        target_dir = os.path.join(EXPORT_DIR, layer_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"    正在导出: {layer_name}")
        
        # 1. 导出权重 (INT8)
        w_int8 = module.weight().int_repr().numpy().astype(np.int8)
        w_int8.tofile(os.path.join(target_dir, "weights.bin"))
        
        # 1.1 导出权重为 .h 头文件
        h_path = os.path.join(target_dir, "weights.h")
        with open(h_path, "w") as f:
            f.write(f"// {layer_name} weights (int8)\n")
            f.write(f"#ifndef {layer_name.upper()}_WEIGHTS_H\n")
            f.write(f"#define {layer_name.upper()}_WEIGHTS_H\n\n")
            f.write("#include <stdint.h>\n\n")
            
            flat_w = w_int8.flatten()
            f.write(f"static const int8_t {layer_name}_weights[{len(flat_w)}] = {{\n")
            
            for i, val in enumerate(flat_w):
                f.write(f"{val}, ")
                if (i + 1) % 16 == 0:
                    f.write("\n")
            
            f.write("\n};\n")
            f.write(f"#endif\n")
        print(f"      ✓ 导出 weights.h")
        
        # 2. 导出 Bias (只有存在时才导出)
        has_bias = module.bias() is not None
        
        if has_bias:
            bias = module.bias().detach().numpy().astype(np.float32)
            bias.tofile(os.path.join(target_dir, "bias.bin"))
            print(f"      ✓ 导出 bias: shape={bias.shape}")
            
            # 2.1 导出 Bias 为 .h 头文件
            h_path = os.path.join(target_dir, "bias.h")
            with open(h_path, "w") as f:
                f.write(f"// {layer_name} bias (float32)\n")
                f.write(f"#ifndef {layer_name.upper()}_BIAS_H\n")
                f.write(f"#define {layer_name.upper()}_BIAS_H\n\n")
                
                f.write(f"static const float {layer_name}_bias[{len(bias)}] = {{\n")
                for i, val in enumerate(bias):
                    f.write(f"{val:.8f}f, ")
                    if (i + 1) % 8 == 0:
                        f.write("\n")
                f.write("\n};\n")
                f.write(f"#endif\n")
            print(f"      ✓ 导出 bias.h")
        else:
            print(f"      ✗ 无 bias (跳过)")
        
        # 3. 导出量化参数
        scale = float(module.scale)
        zero_point = int(module.zero_point)
        
        with open(os.path.join(target_dir, "quant_info.txt"), "w") as f:
            f.write(f"scale: {scale}\n")
            f.write(f"zero_point: {zero_point}\n")
            f.write(f"weight_shape: {w_int8.shape}\n")
            f.write(f"has_bias: {has_bias}\n")  # ← 添加标记

def main():
    if not os.path.exists(QAT_MODEL_PATH):
        print("错误：找不到模型文件")
        return

    model = load_quantized_model()
    
    print(f"\n[-] 开始导出参数至目录: {EXPORT_DIR}/")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # === 逐层导出 ===
    # Litenet 结构: SeparableConv (Depthwise + Pointwise)
    # 注意：model.block1_conv 现在已经是 Quantized 模块了
    
    # Block 1
    save_layer_params("block1_depthwise", model.block1_conv.depthwise)
    save_layer_params("block1_pointwise", model.block1_conv.pointwise) # 这里的 pointwise 已包含 fuse 进来的 BN+ReLU
    
    # Block 2
    save_layer_params("block2_depthwise", model.block2_conv.depthwise)
    save_layer_params("block2_pointwise", model.block2_conv.pointwise)
    
    # Block 3
    save_layer_params("block3_depthwise", model.block3_conv.depthwise)
    save_layer_params("block3_pointwise", model.block3_conv.pointwise)
    
    # Classifier (Linear 层也量化了)
    # Linear 层在 convert 后通常变成 QuantizedLinear
    save_layer_params("classifier", model.classifier)

    print("\n[-] 导出完成！")
    print("    请检查 fpga_params 文件夹，每个层都有 weights.bin 和 bias.bin")

if __name__ == "__main__":
    main()