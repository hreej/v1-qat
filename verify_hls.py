import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from net_model_qat import Litenet_QAT

# ================= 配置 =================
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
IMAGE_PATH = "TUNGRO1_067.jpg"
NUM_CLASSES = 12
# ========================================

def print_layer_quantization_params(model):
    """打印所有层的量化参数"""
    print("\n" + "="*70)
    print("量化参数详情")
    print("="*70)
    
    # 修改：访问 SeparableConv2d_QAT 的子层
    layers = [
        ('quant (输入量化)', model.quant),
        ('block1_conv.depthwise (Depthwise)', model.block1_conv.depthwise),
        ('block1_conv.pointwise (Pointwise)', model.block1_conv.pointwise),
        ('block2_conv.depthwise (Depthwise)', model.block2_conv.depthwise),
        ('block2_conv.pointwise (Pointwise)', model.block2_conv.pointwise),
        ('block3_conv.depthwise (Depthwise)', model.block3_conv.depthwise),
        ('block3_conv.pointwise (Pointwise)', model.block3_conv.pointwise),
        ('classifier (FC)', model.classifier),
        ('dequant (输出反量化)', model.dequant),
    ]
    
    for name, layer in layers:
        print(f"\n{name}:")
        
        # 输入/输出的量化参数
        if hasattr(layer, 'scale'):
            try:
                if callable(layer.scale):
                    scale = layer.scale()
                else:
                    scale = layer.scale
                print(f"  Activation Scale: {scale}")
            except:
                pass
        
        if hasattr(layer, 'zero_point'):
            try:
                if callable(layer.zero_point):
                    zp = layer.zero_point()
                else:
                    zp = layer.zero_point
                print(f"  Activation Zero Point: {zp}")
            except:
                pass
        
        # 权重的量化参数
        if hasattr(layer, 'weight'):
            w = layer.weight()
            if hasattr(w, 'q_per_channel_zero_points'):
                w_zp = w.q_per_channel_zero_points()
                w_scale = w.q_per_channel_scales()
                
                print(f"  Weight Shape: {w.shape}")
                print(f"  Weight dtype: {w.dtype}")
                print(f"  Weight zero_points (first 5): {w_zp[:5].tolist()}")
                print(f"  Weight scales (first 5): {w_scale[:5].tolist()}")
                print(f"  All weight zp == 0? {torch.all(w_zp == 0).item()}")
                
                if not torch.all(w_zp == 0):
                    print(f"  ⚠️  Non-zero zp stats:")
                    print(f"     Min: {w_zp.min().item()}, Max: {w_zp.max().item()}")
                    print(f"     Mean: {w_zp.float().mean().item():.2f}")
                    print(f"     Non-zero count: {(w_zp != 0).sum().item()}/{len(w_zp)}")

def analyze_depthwise_computation(model):
    """分析 Depthwise 卷积的计算需求"""
    print("\n" + "="*70)
    print("Depthwise 卷积计算分析")
    print("="*70)
    
    # 修改：访问子层
    separable_blocks = [
        ('block1_conv', model.block1_conv),
        ('block2_conv', model.block2_conv),
        ('block3_conv', model.block3_conv),
    ]
    
    for block_name, sep_conv in separable_blocks:
        print(f"\n{block_name}:")
        
        # Depthwise 层
        dw_conv = sep_conv.depthwise
        if hasattr(dw_conv, 'weight'):
            w = dw_conv.weight()
            w_int8 = w.int_repr()
            w_zp = w.q_per_channel_zero_points()
            w_scale = w.q_per_channel_scales()
            
            print(f"  [Depthwise]")
            print(f"    输入/输出通道数: {w.shape[0]}")
            print(f"    卷积核大小: {w.shape[2]}x{w.shape[3]}")
            print(f"    每个通道的操作数: {w.shape[2] * w.shape[3]}")
            
            print(f"    权重 int8 值范围 (前3通道):")
            for ch in range(min(3, w.shape[0])):
                ch_w = w_int8[ch, 0, :, :].flatten()
                print(f"      Channel {ch}: [{ch_w.min().item()}, {ch_w.max().item()}], zp={w_zp[ch].item()}")
            
            if torch.all(w_zp == 0):
                print(f"    ✓ 所有 weight zp = 0")
            else:
                print(f"    ⚠️  Weight zp ≠ 0")
                print(f"       Min zp: {w_zp.min().item()}, Max zp: {w_zp.max().item()}")
        
        # Pointwise 层
        pw_conv = sep_conv.pointwise
        if hasattr(pw_conv, 'weight'):
            w = pw_conv.weight()
            w_zp = w.q_per_channel_zero_points()
            
            print(f"  [Pointwise]")
            print(f"    输入通道: {w.shape[1]}, 输出通道: {w.shape[0]}")
            print(f"    All weight zp == 0? {torch.all(w_zp == 0).item()}")

def export_layer_params_for_hls(model, output_dir="hls_params"):
    """导出每层的量化参数供 HLS 使用"""
    os.makedirs(output_dir, exist_ok=True)
    
    layers_info = {}
    
    # Quant layer
    try:
        quant_scale = model.quant.scale() if callable(model.quant.scale) else model.quant.scale
        quant_zp = model.quant.zero_point() if callable(model.quant.zero_point) else model.quant.zero_point
        layers_info['quant'] = {
            'scale': quant_scale,
            'zero_point': quant_zp,
        }
    except Exception as e:
        print(f"Warning: Could not extract quant params: {e}")
    
    # 分别导出 depthwise 和 pointwise 层
    separable_blocks = [
        ('block1_conv', model.block1_conv),
        ('block2_conv', model.block2_conv),
        ('block3_conv', model.block3_conv),
    ]
    
    for block_name, sep_conv in separable_blocks:
        # Depthwise
        dw_conv = sep_conv.depthwise
        if hasattr(dw_conv, 'weight'):
            layers_info[f'{block_name}_depthwise'] = {
                'weight_scales': dw_conv.weight().q_per_channel_scales().numpy(),
                'weight_zero_points': dw_conv.weight().q_per_channel_zero_points().numpy(),
                'weight_int8': dw_conv.weight().int_repr().numpy(),
            }
        
        # Pointwise
        pw_conv = sep_conv.pointwise
        if hasattr(pw_conv, 'weight'):
            layers_info[f'{block_name}_pointwise'] = {
                'weight_scales': pw_conv.weight().q_per_channel_scales().numpy(),
                'weight_zero_points': pw_conv.weight().q_per_channel_zero_points().numpy(),
                'weight_int8': pw_conv.weight().int_repr().numpy(),
            }
    
    # Classifier
    if hasattr(model.classifier, 'weight'):
        layers_info['classifier'] = {
            'weight_scales': model.classifier.weight().q_per_channel_scales().numpy(),
            'weight_zero_points': model.classifier.weight().q_per_channel_zero_points().numpy(),
            'weight_int8': model.classifier.weight().int_repr().numpy(),
        }
    
    # 保存为 .npy 文件
    for layer_name, params in layers_info.items():
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                param_value = np.array([param_value])
            
            filename = f"{output_dir}/{layer_name}_{param_name}.npy"
            np.save(filename, param_value)
            print(f"[+] Saved: {filename}")
    
    print(f"\n[+] 总共导出 {sum(len(p) for p in layers_info.values())} 个参数文件")

def main():
    if not os.path.exists(QAT_MODEL_PATH):
        print(f"错误: 找不到模型 {QAT_MODEL_PATH}")
        return

    # 1. 加载量化模型
    print("[-] 正在加载量化模型...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    state_dict = torch.load(QAT_MODEL_PATH, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    
    # 打印量化参数
    print_layer_quantization_params(quantized_model)
    
    # 分析 Depthwise 计算
    analyze_depthwise_computation(quantized_model)
    
    # 导出参数
    print("\n" + "="*70)
    print("导出 HLS 参数")
    print("="*70)
    export_layer_params_for_hls(quantized_model)
    
    # 2. 准备输入数据
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(IMAGE_PATH):
        print(f"警告: 找不到图片 {IMAGE_PATH}")
        return

    img_pil = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = preprocess(img_pil).unsqueeze(0)
    
    # 3. 运行推理
    with torch.no_grad():
        x = quantized_model.quant(input_tensor)
        x = quantized_model.block1_conv(x)
        x = quantized_model.block1_bn(x)
        x = quantized_model.block1_relu(x)
        x = quantized_model.block1_pool(x)
        
        x = quantized_model.block2_conv(x)
        x = quantized_model.block2_bn(x)
        x = quantized_model.block2_relu(x)
        x = quantized_model.block2_pool(x)
        
        x = quantized_model.block3_conv(x)
        x = quantized_model.block3_bn(x)
        x = quantized_model.block3_relu(x)
        
        x = quantized_model.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = quantized_model.dropout(x)
        
        int8_output = quantized_model.classifier(x)
        int8_values = int8_output.int_repr().numpy().flatten()
        
        float_output = quantized_model.dequant(int8_output)
        probs = torch.nn.functional.softmax(float_output, dim=1).numpy().flatten()

    # 4. 打印结果
    print("\n" + "="*70)
    print("PyTorch 标准输出 (Golden Reference)")
    print("="*70)
    print(f"{'Class':<6} | {'Int8 Value':<12} | {'Prob':<8}")
    print("-" * 40)
    
    max_idx = np.argmax(int8_values)
    
    for i, val in enumerate(int8_values):
        mark = " ★" if i == max_idx else ""
        print(f"{i:<6} | {val:<12} | {probs[i]:.4f}{mark}")
    
    print("-" * 40)
    print(f"预测类别: {max_idx}")
    print("="*70)
    
    print("\n" + "="*70)
    print("HLS 实现关键点")
    print("="*70)
    print("1. Depthwise 卷积:")
    print("   - 检查 Weight zp 是否为 0")
    print("   - 如果 zp ≠ 0: acc += (x_int8 - zp_in) * (w_int8 - zp_w[ch])")
    print("   - 如果 zp = 0: acc += (x_int8 - zp_in) * w_int8")
    print("   - 每个通道有独立的 zp_w[ch] 和 scale_w[ch]")
    print("")
    print("2. Pointwise 卷积:")
    print("   - 通常 Weight zp = 0，可以简化")
    print("   - 公式: acc += (x_int8 - zp_in) * w_int8")
    print("")
    print("3. 重量化:")
    print("   - M[ch] = (scale_in * scale_w[ch]) / scale_out")
    print("   - output = clamp(round(acc * M[ch]) + zp_out, -128, 127)")
    print("")
    print("4. 已导出参数文件到 hls_params/ 目录")
    print("   - 注意：DeQuantize 层的参数需要从 classifier 的输出量化参数推断")

if __name__ == "__main__":
    main()