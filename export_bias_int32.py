import torch
import os
import numpy as np
from net_model_qat import Litenet_QAT

# ================= 配置 =================
# 指向你的量化模型检查点
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
# 输出目录 (会生成在 fpga_params 的各个子文件夹中)
EXPORT_DIR = "fpga_params"
NUM_CLASSES = 12
# ========================================

def load_quantized_model():
    """
    加载量化模型并恢复结构，与 export_for_fpga.py 逻辑一致
    """
    print(f"[-] 正在加载量化模型...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 1. 融合与配置
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 2. 转换为量化结构
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # 3. 加载权重
    state_dict = torch.load(QAT_MODEL_PATH, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    return quantized_model

def get_layer_scales(model):
    """
    定义数据流图，准确获取每一层的 Input Scale。
    因为 Input Scale = 上一层的 Output Scale。
    """
    layers = {}
    
    # 1. 获取输入端的 Scale (由 QuantStub 决定)
    input_scale_global = float(model.quant.scale)
    print(f"[-] Global Input Scale: {input_scale_global}")

    # === Block 1 ===
    # Depthwise 输入来自 Global Input
    layers['block1_depthwise'] = {
        'module': model.block1_conv.depthwise,
        'input_scale': input_scale_global
    }
    # Pointwise 输入来自 Depthwise 输出
    scale_b1_dw = float(model.block1_conv.depthwise.scale)
    layers['block1_pointwise'] = {
        'module': model.block1_conv.pointwise,
        'input_scale': scale_b1_dw
    }

    # === Block 2 ===
    # 输入来自 Block1 Pointwise (经过 Pool 层，Scale 不变)
    scale_b1_pw = float(model.block1_conv.pointwise.scale)
    layers['block2_depthwise'] = {
        'module': model.block2_conv.depthwise,
        'input_scale': scale_b1_pw
    }
    scale_b2_dw = float(model.block2_conv.depthwise.scale)
    layers['block2_pointwise'] = {
        'module': model.block2_conv.pointwise,
        'input_scale': scale_b2_dw
    }

    # === Block 3 ===
    scale_b2_pw = float(model.block2_conv.pointwise.scale)
    layers['block3_depthwise'] = {
        'module': model.block3_conv.depthwise,
        'input_scale': scale_b2_pw
    }
    scale_b3_dw = float(model.block3_conv.depthwise.scale)
    layers['block3_pointwise'] = {
        'module': model.block3_conv.pointwise,
        'input_scale': scale_b3_dw
    }

    # === Classifier ===
    scale_b3_pw = float(model.block3_conv.pointwise.scale)
    layers['classifier'] = {
        'module': model.classifier,
        'input_scale': scale_b3_pw
    }

    return layers

def calculate_and_save_bias(layer_name, info):
    module = info['module']
    s_in = info['input_scale']
    
    # 检查是否存在 Bias
    if module.bias() is None:
        print(f"[{layer_name}] 跳过 (无 Bias)")
        return

    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. 获取 Float Bias
    bias_float = module.bias().detach()

    # 2. 获取 Weight Scale (关键步骤)
    # PyTorch 量化模型将 Scale 存储在 packed_params 中，可以通过 module.weight() 访问
    q_weight = module.weight()
    
    try:
        # 尝试获取 Per-Channel Scales (通常 Conv2d/Linear 默认是这个)
        s_weight = q_weight.q_per_channel_scales()
        print(f"[{layer_name}] 模式: Per-Channel | Bias Shape: {bias_float.shape}")
    except RuntimeError:
        # 降级为 Per-Tensor Scale
        s_weight = q_weight.q_scale()
        print(f"[{layer_name}] 模式: Per-Tensor")

    # 3. 计算 Bias Int32
    # 公式: bias_int32 = round( bias_float / (s_in * s_weight) )
    # 注意: s_in 是标量, s_weight 是向量, bias_float 是向量 -> 广播机制自动处理
    scale_factor = s_in * s_weight
    
    # 加上 1e-6 防止除以0（虽然 Scale 不应为0）
    bias_int32_tensor = torch.round(bias_float / scale_factor).to(torch.int32)
    bias_int32_np = bias_int32_tensor.numpy()

    # 4. 保存为 .bin (二进制)
    bin_path = os.path.join(target_dir, "bias_int32.bin")
    bias_int32_np.tofile(bin_path)

    # 5. 保存为 .h 头文件格式 (方便 C/FPGA 复制)
    h_path = os.path.join(target_dir, "bias_int32.h")
    with open(h_path, "w") as f:
        f.write(f"// Layer: {layer_name}\n")
        f.write(f"// Input Scale: {s_in:.8f}\n")
        f.write(f"// Count: {len(bias_int32_np)}\n")
        f.write(f"const int32_t {layer_name}_bias[] = {{\n    ")
        for i, val in enumerate(bias_int32_np):
            f.write(f"{val}, ")
            if (i + 1) % 8 == 0:
                f.write("\n    ")
        f.write("\n};\n")

    print(f"    -> 已生成: {bin_path}")
    print(f"    -> 已生成: {h_path}")

def main():
    if not os.path.exists(QAT_MODEL_PATH):
        print(f"错误: 找不到模型文件 {QAT_MODEL_PATH}")
        return

    # 加载模型
    model = load_quantized_model()
    
    # 获取每一层的 Input Scale
    layers_map = get_layer_scales(model)
    
    print("\n" + "="*50)
    print("开始计算 Bias Int32")
    print("="*50)
    
    # 逐层处理
    for name, info in layers_map.items():
        calculate_and_save_bias(name, info)

    print("\n完成！请检查 fpga_params 文件夹。")

if __name__ == "__main__":
    main()