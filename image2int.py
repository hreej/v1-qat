import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import struct
from net_model_qat import Litenet_QAT

# ================= 配置 =================
# 1. 模型路径 (必须加载才能获取 Input Scale/ZP)
QAT_MODEL_PATH = "qat_checkpoints/litenet_int8_qat.pth"
# 2. 你的测试图片路径
IMAGE_PATH = r"D:\study\CNN_demo\Litenet\dataset_v5\valid\cotton_target_spot\crop_20.jpg"  # <--- 修改为你的图片文件名
# 3. 输出头文件名称
OUTPUT_HEADER = "image/input_image.h"
OUTPUT_PACKED_HEADER = "image/input_image_packed.h"
OUTPUT_BIN = "image/image.bin"
# 4. 类别数
NUM_CLASSES = 12
# ========================================

def load_quant_params():
    """从保存的模型中提取全局输入的 Scale 和 Zero Point"""
    print("[-] 加载模型以获取输入量化参数...")
    model = Litenet_QAT(num_classes=NUM_CLASSES)
    
    # 必须执行与导出时相同的步骤
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # 加载权重
    state_dict = torch.load(QAT_MODEL_PATH, map_location='cpu')
    quantized_model.load_state_dict(state_dict)
    
    # 获取输入端的 QuantStub 参数
    input_scale = float(quantized_model.quant.scale)
    input_zp = int(quantized_model.quant.zero_point)
    
    print(f"    Input Scale: {input_scale}")
    print(f"    Input Zero Point: {input_zp}")
    
    return input_scale, input_zp

def process_image(image_path, input_scale, input_zp):
    """
    读取图片 -> Resize -> Normalize -> Quantize -> Int8 Array
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片 {image_path}")
        return

    # 1. 定义预处理 (必须与 train_DKD_ReLR.py 中的 get_data_loaders 保持完全一致)
    # 引用自 train_DKD_ReLR.py
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # 归一化到 [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ])

    # 2. 加载图片
    img_pil = Image.open(image_path).convert('RGB')
    
    # 3. 执行预处理 (得到 Float Tensor)
    img_tensor = preprocess(img_pil) # Shape: [3, 128, 128]
    
    # 4. 执行量化
    # 公式: q = round(x / scale) + zp
    img_quantized = torch.round(img_tensor / input_scale) + input_zp
    
    # 5. 截断到 int8 范围 [-128, 127]
    # 注意: 虽然 QuantStub 可能输出 uint8 (0-255)，但在 HLS 里的 ap_int<8> 是带符号的。
    # 为了安全，我们通常将其 Clamp 到带符号范围。
    # 如果你的 HLS 输入定义为 ap_uint<8>，则 clamp 到 0-255。
    # 这里假设通用 int8:
    img_quantized = torch.clamp(img_quantized, -128, 127)
    
    # 转为 numpy int8
    img_np = img_quantized.to(torch.int8).numpy()
    
    return img_np

def save_packed_to_header(img_np, filename):
    """将 int8 数据打包为 int32 并保存为 C++ 头文件"""
    # 展平数组
    flat_data = img_np.flatten()
    
    # 补齐到 4 的倍数
    if len(flat_data) % 4 != 0:
        pad_len = 4 - (len(flat_data) % 4)
        flat_data = np.pad(flat_data, (0, pad_len), 'constant')

    packed_data = []
    for i in range(0, len(flat_data), 4):
        # Little-endian packing
        b0 = int(flat_data[i]) & 0xFF
        b1 = int(flat_data[i+1]) & 0xFF
        b2 = int(flat_data[i+2]) & 0xFF
        b3 = int(flat_data[i+3]) & 0xFF
        
        packed_val = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
        packed_data.append(packed_val)

    print(f"[-] 生成打包头文件: {filename}")
    with open(filename, "w") as f:
        f.write("// Auto-generated packed data header from image2int8.py\n")
        f.write(f"// Original Image: {IMAGE_PATH}\n")
        f.write("#ifndef INPUT_IMAGE_PACKED_H\n")
        f.write("#define INPUT_IMAGE_PACKED_H\n\n")
        f.write("#include <ap_int.h>\n\n")
        f.write(f"// Original elements: {len(flat_data)}\n")
        f.write(f"// Packed elements: {len(packed_data)}\n")
        f.write(f"static ap_int<32> input_packed_arr[{len(packed_data)}] = {{\n")
        
        for i, val in enumerate(packed_data):
            f.write(f"    0x{val:08X}")
            if i < len(packed_data) - 1:
                f.write(",")
            if (i + 1) % 8 == 0:
                f.write("\n")
            else:
                f.write(" ")
                
        f.write("\n};\n\n")
        f.write("#endif\n")
    print("    打包完成！")

def save_to_header(img_np, filename):
    """保存为 C++ 头文件"""
    # 展平数组: [3, 128, 128] -> [49152]
    flat_data = img_np.flatten()
    
    print(f"[-] 生成头文件: {filename}")
    with open(filename, "w") as f:
        f.write(f"// Image: {IMAGE_PATH}\n")
        f.write(f"// Shape: 128x128x3 (NCHW format in PyTorch, verify HLS requirement)\n")
        f.write(f"// Total elements: {len(flat_data)}\n")
        f.write(f"#ifndef INPUT_IMAGE_H\n")
        f.write(f"#define INPUT_IMAGE_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write(f"static const int8_t input_data[{len(flat_data)}] = {{\n    ")
        
        for i, val in enumerate(flat_data):
            f.write(f"{val}, ")
            if (i + 1) % 16 == 0:
                f.write("\n    ")
        
        f.write("\n};\n")
        f.write(f"#endif\n")
    print("    完成！")

def save_packed_to_bin(img_np, filename):
    """将 int8 数据打包为 int32 并保存为二进制文件"""
    # 展平数组
    flat_data = img_np.flatten()
    
    # 补齐到 4 的倍数
    if len(flat_data) % 4 != 0:
        pad_len = 4 - (len(flat_data) % 4)
        flat_data = np.pad(flat_data, (0, pad_len), 'constant')

    packed_data = []
    for i in range(0, len(flat_data), 4):
        # Little-endian packing
        b0 = int(flat_data[i]) & 0xFF
        b1 = int(flat_data[i+1]) & 0xFF
        b2 = int(flat_data[i+2]) & 0xFF
        b3 = int(flat_data[i+3]) & 0xFF
        
        packed_val = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0
        packed_data.append(packed_val)

    print(f"[-] 生成二进制文件: {filename}")
    with open(filename, 'wb') as f_out:
        for val in packed_data:
            f_out.write(struct.pack('<I', val))
    print(f"    二进制文件生成完成! 大小: {len(packed_data) * 4} 字节")

def main():
    # 1. 获取量化参数
    s_in, zp_in = load_quant_params()
    
    # 2. 处理图片
    # 创建一个随机噪声图片用于演示，实际使用请确保 IMAGE_PATH 指向真实文件
    if not os.path.exists(IMAGE_PATH):
        print(f"提示: 未找到 {IMAGE_PATH}，生成一张随机噪声图片用于测试...")
        Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)).save(IMAGE_PATH)
        
    img_int8 = process_image(IMAGE_PATH, s_in, zp_in)
    
    # 3. 导出
    if img_int8 is not None:
        save_to_header(img_int8, OUTPUT_HEADER)
        save_packed_to_header(img_int8, OUTPUT_PACKED_HEADER)
        save_packed_to_bin(img_int8, OUTPUT_BIN)
        
        # 打印部分数据预览
        print("\n数据预览 (前20个数值):")
        print(img_int8.flatten()[:20])

if __name__ == "__main__":
    main()