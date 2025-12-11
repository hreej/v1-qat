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

def calculate_and_save_bias_int32(layer_name, module, input_scale):
    """
    计算并保存 INT32 格式的 Bias
    公式: bias_int32 = round( bias_float / (input_scale * weight_scale) )
    """
    if module.bias() is None:
        return

    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. 获取 Float Bias
    bias_float = module.bias().detach()

    # 2. 获取 Weight Scale
    q_weight = module.weight()
    try:
        # 尝试获取 Per-Channel Scales
        s_weight = q_weight.q_per_channel_scales()
    except RuntimeError:
        # 降级为 Per-Tensor Scale
        s_weight = q_weight.q_scale()

    # 3. 计算 Bias Int32
    scale_factor = input_scale * s_weight
    bias_int32_tensor = torch.round(bias_float / scale_factor).to(torch.int32)
    bias_int32_np = bias_int32_tensor.numpy()

    # 4. 保存为 .bin
    bin_path = os.path.join(target_dir, "bias_int32.bin")
    bias_int32_np.tofile(bin_path)

    # 5. 保存为 .h 头文件
    h_path = os.path.join(target_dir, "bias_int32.h")
    with open(h_path, "w") as f:
        f.write(f"// Layer: {layer_name} (INT32 Bias)\n")
        f.write(f"// Input Scale: {input_scale:.8f}\n")
        f.write(f"#ifndef {layer_name.upper()}_BIAS_INT32_H\n")
        f.write(f"#define {layer_name.upper()}_BIAS_INT32_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"static const int32_t {layer_name}_bias_int32[{len(bias_int32_np)}] = {{\n    ")
        for i, val in enumerate(bias_int32_np):
            f.write(f"{val}, ")
            if (i + 1) % 8 == 0:
                f.write("\n    ")
        f.write("\n};\n")
        f.write(f"#endif\n")
    
    print(f"      ✓ 导出 bias_int32.h")

def save_layer_params(layer_name, module, input_scale):
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
            # 2.1 导出 Bias 为 .h 头文件 (Int32 - 用于 FPGA 计算)
            calculate_and_save_bias_int32(layer_name, module, input_scale)
            
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
    
    # 获取各层的 Input Scale (用于计算 Bias Int32)
    # Input Scale = 上一层的 Output Scale
    input_scale_global = float(model.quant.scale)
    
    # === 逐层导出 ===
    
    # Block 1
    # Depthwise 输入来自 Global Input
    save_layer_params("block1_depthwise", model.block1_conv.depthwise, input_scale_global)
    
    # Pointwise 输入来自 Depthwise 输出
    scale_b1_dw = float(model.block1_conv.depthwise.scale)
    save_layer_params("block1_pointwise", model.block1_conv.pointwise, scale_b1_dw)
    
    # Block 2
    # 输入来自 Block1 Pointwise (经过 Pool 层，Scale 不变)
    scale_b1_pw = float(model.block1_conv.pointwise.scale)
    save_layer_params("block2_depthwise", model.block2_conv.depthwise, scale_b1_pw)
    
    scale_b2_dw = float(model.block2_conv.depthwise.scale)
    save_layer_params("block2_pointwise", model.block2_conv.pointwise, scale_b2_dw)
    
    # Block 3
    scale_b2_pw = float(model.block2_conv.pointwise.scale)
    save_layer_params("block3_depthwise", model.block3_conv.depthwise, scale_b2_pw)
    
    scale_b3_dw = float(model.block3_conv.depthwise.scale)
    save_layer_params("block3_pointwise", model.block3_conv.pointwise, scale_b3_dw)
    
    # Classifier
    scale_b3_pw = float(model.block3_conv.pointwise.scale)
    save_layer_params("classifier", model.classifier, scale_b3_pw)

    print("\n[-] 导出完成！")
    print("    请检查 fpga_params 文件夹，每个层都有 weights.h, bias.h 和 bias_int32.h")

if __name__ == "__main__":
    main()