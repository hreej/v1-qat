import torch
import os
import numpy as np
import math  # 新增: 用于数学计算
import torch.nn as nn
from net_model_qat import Litenet_QAT

# ================= 配置 =================
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
EXPORT_DIR = "fpga_params"
NUM_CLASSES = 12
# ========================================

def load_quantized_model():
    """
    加载量化模型的标准流程
    """
    print(f"[-] 正在加载量化模型架构...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 1. 融合 & 准备
    model.eval()
    model.fuse_model()
    model.train() 
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

def get_quantized_scale(scale_val):
    """
    [新增] 将浮点 scale 转换为 (multiplier, shift) 的形式
    原理: scale ~= multiplier * 2^(-shift)
    其中 multiplier 是 32位整数
    """
    if scale_val == 0:
        return 0, 0
        
    # 使用 frexp 分解: scale = significand * 2^exponent
    # significand 范围在 [0.5, 1)
    significand, exponent = math.frexp(scale_val)
    
    # 将小数部分映射到 int32 的高位 (乘以 2^31) 以最大化精度
    significand_q = int(round(significand * (1 << 31)))
    
    # 调整 shift
    # 我们希望: significand_q * 2^(-31) * 2^exponent = scale
    # 所以: scale = significand_q * 2^(exponent - 31)
    # 定义 shift = 31 - exponent (因为通常是右移)
    shift = 31 - exponent
    
    # 限制 shift 范围 (防止溢出，虽然一般不会)
    if shift < 0:
        shift = 0
        # 如果 shift < 0，说明 scale 非常大，可能需要特殊处理，但量化网络中罕见
        
    return significand_q, shift

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
        s_weight = q_weight.q_per_channel_scales()
    except RuntimeError:
        s_weight = q_weight.q_scale()

    # 3. 计算 Bias Int32
    # 注意广播机制: input_scale(标量) * s_weight(向量)
    scale_factor = input_scale * s_weight
    
    # 添加 epsilon 防止除零，虽然 scale 不应为 0
    bias_int32_tensor = torch.round(bias_float / (scale_factor + 1e-9)).to(torch.int32)
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

def calculate_and_save_quant_params(layer_name, module, input_scale):
    """
    [新增] 计算并保存定点乘数 (Multiplier) 和移位 (Shift)
    公式: M = (S_in * S_w) / S_out
    """
    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. 获取各种 Scale
    s_out = float(module.scale)
    zp_out = int(module.zero_point)
    
    q_weight = module.weight()
    try:
        s_weight = q_weight.q_per_channel_scales().numpy()
    except RuntimeError:
        s_weight = np.array([q_weight.q_scale()])

    # 确保 s_weight 是 numpy 数组
    if not isinstance(s_weight, np.ndarray):
        s_weight = np.array([s_weight])

    # 2. 计算 Effective Scale (M)
    # M = (Input Scale * Weight Scale) / Output Scale
    effective_scales = (input_scale * s_weight) / s_out

    # 3. 转换为 Multiplier 和 Shift
    multipliers = []
    shifts = []
    
    for s in effective_scales:
        m, sh = get_quantized_scale(s)
        multipliers.append(m)
        shifts.append(sh)
    
    multipliers = np.array(multipliers, dtype=np.int32)
    shifts = np.array(shifts, dtype=np.int32)

    # 4. 导出 mult.h (定点乘数)
    with open(os.path.join(target_dir, "mult.h"), "w") as f:
        f.write(f"// Layer: {layer_name} (Quantized Multiplier)\n")
        f.write(f"#ifndef {layer_name.upper()}_MULT_H\n")
        f.write(f"#define {layer_name.upper()}_MULT_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"static const int32_t {layer_name}_mult[{len(multipliers)}] = {{\n    ")
        for i, val in enumerate(multipliers):
            f.write(f"{val}, ")
            if (i + 1) % 8 == 0: f.write("\n    ")
        f.write("\n};\n#endif\n")

    # 5. 导出 shift.h (右移位数)
    with open(os.path.join(target_dir, "shift.h"), "w") as f:
        f.write(f"// Layer: {layer_name} (Quantized Shift)\n")
        f.write(f"#ifndef {layer_name.upper()}_SHIFT_H\n")
        f.write(f"#define {layer_name.upper()}_SHIFT_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"static const int32_t {layer_name}_shift[{len(shifts)}] = {{\n    ")
        for i, val in enumerate(shifts):
            f.write(f"{val}, ")
            if (i + 1) % 16 == 0: f.write("\n    ")
        f.write("\n};\n#endif\n")

    # 6. 导出 output zero point (zp_out)
    # 这是一个标量，但也保存为 .h 方便读取
    with open(os.path.join(target_dir, "zp.h"), "w") as f:
        f.write(f"// Layer: {layer_name} (Zero Points)\n")
        f.write(f"#ifndef {layer_name.upper()}_ZP_H\n")
        f.write(f"#define {layer_name.upper()}_ZP_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"static const int8_t {layer_name}_zp_out = {zp_out};\n")
        f.write(f"#endif\n")

    print(f"      ✓ 导出 mult.h, shift.h, zp.h")


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
        
        # 1.1 导出权重为 .h
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
            f.write("\n};\n#endif\n")
        print(f"      ✓ 导出 weights.h")
        
        # 2. 导出 Bias (Int32)
        if module.bias() is not None:
            calculate_and_save_bias_int32(layer_name, module, input_scale)
        else:
            print(f"      ✗ 无 bias (跳过)")
        
        # 3. 导出量化参数 (Multiplier, Shift, ZP)
        calculate_and_save_quant_params(layer_name, module, input_scale)
        
        # 4. 导出原始文本信息 (备用)
        scale = float(module.scale)
        zero_point = int(module.zero_point)
        with open(os.path.join(target_dir, "quant_info.txt"), "w") as f:
            f.write(f"scale: {scale}\n")
            f.write(f"zero_point: {zero_point}\n")
            f.write(f"weight_shape: {w_int8.shape}\n")

def main():
    if not os.path.exists(QAT_MODEL_PATH):
        print("错误：找不到模型文件")
        return

    model = load_quantized_model()
    
    print(f"\n[-] 开始导出参数至目录: {EXPORT_DIR}/")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 获取 Global Input Scale
    input_scale_global = float(model.quant.scale)
    
    # === 逐层导出 ===
    # 传递规则: 下一层的 input_scale 等于 上一层的 output_scale
    
    # Block 1
    save_layer_params("block1_depthwise", model.block1_conv.depthwise, input_scale_global)
    
    scale_b1_dw = float(model.block1_conv.depthwise.scale)
    save_layer_params("block1_pointwise", model.block1_conv.pointwise, scale_b1_dw)
    
    # Block 2
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
    print("    请检查 fpga_params 文件夹，包含 weights.h, bias_int32.h, mult.h, shift.h, zp.h")

if __name__ == "__main__":
    main()