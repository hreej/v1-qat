import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from net_model_qat import Litenet_QAT

# ================= 配置 =================
# 指向你的量化模型权重
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
# 指向一张用于测试的图片 (可以是训练集中的任意一张)
TEST_IMAGE_PATH = r"D:\study\CNN_demo\Litenet\dataset_v5\valid\cotton_target_spot\crop_20.jpg" 
# 如果没有图片，脚本会生成随机噪声进行测试
USE_DUMMY_IMAGE = False 

EXPORT_DIR = "debug_outputs"
NUM_CLASSES = 12
# ========================================

# 用于存储每一层的输出
activation_cache = {}

def get_activation(name):
    """
    Hook 函数：在 forward 过程中截获输出
    """
    def hook(model, input, output):
        # 如果输出是量化张量，提取 int_repr (整数表现形式)
        if output.is_quantized:
            # int_repr() 返回底层的 int8/uint8 数值，不包含 scale/zp
            activation_cache[name] = {
                "data": output.int_repr().numpy(),
                "scale": output.q_scale(),
                "zp": output.q_zero_point()
            }
        else:
            # 如果是浮点（例如最后一层 dequant 之后），直接保存
            activation_cache[name] = {
                "data": output.detach().numpy(),
                "scale": 1.0,
                "zp": 0
            }
        
        # 保存输入 Scale 用于计算 Bias 量化值
        if len(input) > 0 and hasattr(input[0], 'is_quantized') and input[0].is_quantized:
            activation_cache[name]["input_scale"] = input[0].q_scale()
            # 保存输入数据和 ZP 用于手动计算卷积结果
            activation_cache[name]["input_data"] = input[0].int_repr().numpy()
            activation_cache[name]["input_zp"] = input[0].q_zero_point()
            
    return hook

def load_quantized_model():
    print(f"[-] 加载模型架构...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 必须执行标准的量化转换流程
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    print(f"[-] 加载权重: {QAT_MODEL_PATH}")
    state_dict = torch.load(QAT_MODEL_PATH, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    return quantized_model

def save_activations_to_file(layer_name, data_dict):
    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)
    
    data = data_dict["data"]
    scale = data_dict["scale"]
    zp = data_dict["zp"]
    
    # 1. 保存基本信息
    with open(os.path.join(target_dir, "info.txt"), "w") as f:
        f.write(f"Shape: {data.shape}\n")
        f.write(f"Scale: {scale}\n")
        f.write(f"Zero Point: {zp}\n")
        f.write(f"Min Val: {np.min(data)}\n")
        f.write(f"Max Val: {np.max(data)}\n")

    # 2. 保存纯文本数据 (HLS 友好格式)
    # 将数据展平，每行一个整数
    flat_data = data.flatten()
    np.savetxt(os.path.join(target_dir, "output_int8.txt"), flat_data, fmt='%d')
    
    # 3. 打印预览
    print(f"Layer: {layer_name}")
    print(f"  Shape: {data.shape}")
    print(f"  Scale: {scale:.6f}, ZP: {zp}")
    print(f"  前20个数值 (Int8): {flat_data[:20]}")
    print("-" * 40)

def save_bias(layer_name, layer_obj, input_scale):
    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)
    
    if hasattr(layer_obj, 'bias') and layer_obj.bias() is not None:
        # 1. 获取浮点 Bias
        bias_float = layer_obj.bias().detach().numpy()
        np.savetxt(os.path.join(target_dir, "bias_float.txt"), bias_float, fmt='%.8f')
        
        # 2. 计算 Int32 Bias (如果可能)
        # Bias_int32 = Bias_float / (Input_Scale * Weight_Scale)
        if input_scale is not None:
            try:
                w = layer_obj.weight()
                if w.qscheme() in [torch.per_channel_symmetric, torch.per_channel_affine]:
                    w_scale = w.q_per_channel_scales().numpy()
                else:
                    w_scale = w.q_scale()
                
                # 广播计算
                scale_product = input_scale * w_scale
                bias_int32 = np.round(bias_float / scale_product).astype(np.int32)
                
                np.savetxt(os.path.join(target_dir, "bias_int32.txt"), bias_int32, fmt='%d')
                print(f"Layer: {layer_name} Bias saved.")
                print(f"  Bias Int32 Head: {bias_int32[:5]}")
            except Exception as e:
                print(f"  Could not calculate int32 bias for {layer_name}: {e}")
        else:
             print(f"  No input scale found for {layer_name}, skipping int32 bias.")

def calculate_conv2d_int32_output(layer_name, layer_obj, input_data, input_zp):
    """
    手动计算卷积层在加 Bias 之前的 Int32 累加值 (MAC结果)
    仅支持 1x1 Pointwise 卷积 (因为使用了矩阵乘法简化)
    """
    print(f"[-] Calculating Int32 output (before bias) for {layer_name}...")
    target_dir = os.path.join(EXPORT_DIR, layer_name)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 1. 获取权重 Int8
        # layer_obj.weight() 返回 QuantizedTensor
        w_quant = layer_obj.weight()
        w_int8 = w_quant.int_repr().numpy() # Shape: (Cout, Cin, kH, kW)
        
        # === 新增：保存权重 ===
        weight_save_path = os.path.join(target_dir, "weights_int8.txt")
        np.savetxt(weight_save_path, w_int8.flatten(), fmt='%d')
        print(f"  Saved Int8 weights to {weight_save_path}")
        print(f"  Weights Shape: {w_int8.shape}")
        
        # 检查是否是 1x1 卷积
        if w_int8.shape[2] != 1 or w_int8.shape[3] != 1:
            print(f"  Skipping {layer_name}: Only 1x1 convolution is supported for this debug function.")
            return

        # 2. 准备数据
        # Input: (N, Cin, H, W)
        # Weight: (Cout, Cin, 1, 1) -> (Cout, Cin)
        
        N, Cin, H, W = input_data.shape
        Cout, _, _, _ = w_int8.shape
        
        # 3. 处理 Input Zero Point
        # 公式: Sum( (Input - Input_ZP) * Weight )
        # 注意：这里假设 Weight 是对称量化 (ZP=0)，这是 FBGEMM 的默认行为
        # 如果 Weight 也有 ZP，需要额外减去。但通常 Weight ZP 为 0。
        
        # 将 Input 转换为 Int32 并减去 ZP
        input_shifted = input_data.astype(np.int32) - input_zp
        
        # 变形为 (N*H*W, Cin) 以便进行矩阵乘法
        # permute (N, C, H, W) -> (N, H, W, C) -> flatten
        input_reshaped = input_shifted.transpose(0, 2, 3, 1).reshape(-1, Cin)
        
        # 变形权重 (Cout, Cin) -> 转置为 (Cin, Cout)
        w_reshaped = w_int8.reshape(Cout, Cin).transpose(1, 0).astype(np.int32)
        
        # 4. 矩阵乘法
        # (Pixels, Cin) @ (Cin, Cout) -> (Pixels, Cout)
        output_flat = np.dot(input_reshaped, w_reshaped)
        
        # 5. 恢复形状 (N, H, W, Cout) -> (N, Cout, H, W)
        output_int32 = output_flat.reshape(N, H, W, Cout).transpose(0, 3, 1, 2)
        
        # 6. 保存
        save_path = os.path.join(target_dir, "output_int32_before_requant.txt")
        np.savetxt(save_path, output_int32.flatten(), fmt='%d')
        
        print(f"  Saved Int32 output (before bias) to {save_path}")
        print(f"  Shape: {output_int32.shape}")
        print(f"  First 10 values: {output_int32.flatten()[:10]}")
        
    except Exception as e:
        print(f"  Error calculating int32 output: {e}")

def main():
    model = load_quantized_model()
    
    # === 注册 Hooks ===
    # 我们需要监控每一层卷积后的输出
    # 注意：在量化模型中，Conv+BN+ReLU 已经被融合为一个模块
    
    # 1. 输入量化层 (QuantStub) - 这是进入第一个卷积前的 int8 数据
    model.quant.register_forward_hook(get_activation('00_input_quantized'))
    
    # 2. Block 1
    model.block1_conv.depthwise.register_forward_hook(get_activation('01_block1_dw'))
    model.block1_conv.pointwise.register_forward_hook(get_activation('02_block1_pw'))
    
    # 3. Block 2
    model.block2_conv.depthwise.register_forward_hook(get_activation('03_block2_dw'))
    model.block2_conv.pointwise.register_forward_hook(get_activation('04_block2_pw'))
    
    # 4. Block 3
    model.block3_conv.depthwise.register_forward_hook(get_activation('05_block3_dw'))
    model.block3_conv.pointwise.register_forward_hook(get_activation('06_block3_pw'))
    
    # 5. Classifier (注意：通常分类器输出后会 Dequantize，我们需要看 Dequant 之前的)
    # 如果 model.classifier 是 QuantizedLinear，它输出的是量化值
    model.classifier.register_forward_hook(get_activation('07_classifier'))

    # === 准备输入数据 ===
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if USE_DUMMY_IMAGE or not os.path.exists(TEST_IMAGE_PATH):
        print("[-] 使用随机生成的数据进行测试...")
        # 模拟一张随机图片
        img_tensor = torch.randn(1, 3, 128, 128)
    else:
        print(f"[-] 加载测试图片: {TEST_IMAGE_PATH}")
        img = Image.open(TEST_IMAGE_PATH).convert('RGB')
        img_tensor = transform(img).unsqueeze(0) # Add batch dim

    # === 运行推理 ===
    print("[-] 运行推理...")
    with torch.no_grad():
        model(img_tensor)

    # === 保存结果 ===
    print(f"\n[-] 正在保存调试数据至 {EXPORT_DIR}/ ...\n")
    
    # 按名称排序保存
    for name in sorted(activation_cache.keys()):
        save_activations_to_file(name, activation_cache[name])

    # === 新增：保存 Pointwise Bias ===
    print("\n[-] 正在保存 Bias 数据...")
    layers_with_bias = {
        '02_block1_pw': model.block1_conv.pointwise,
        '04_block2_pw': model.block2_conv.pointwise,
        '06_block3_pw': model.block3_conv.pointwise,
        '07_classifier': model.classifier
    }
    
    for name, layer in layers_with_bias.items():
        if name in activation_cache:
            input_scale = activation_cache[name].get("input_scale")
            save_bias(name, layer, input_scale)

    # === 新增：计算 Block1 Pointwise 卷积后、加偏置前的值 ===
    if '02_block1_pw' in activation_cache:
        pw_cache = activation_cache['02_block1_pw']
        if "input_data" in pw_cache:
            calculate_conv2d_int32_output(
                '02_block1_pw', 
                model.block1_conv.pointwise, 
                pw_cache["input_data"], 
                pw_cache["input_zp"]
            )

    print("\n完成！你可以在 HLS 中读取 '00_input_quantized/output_int8.txt' 作为输入，")
    print("然后对比每一层的输出结果。")

if __name__ == "__main__":
    main()