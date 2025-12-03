import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """
    PyTorch 实现的深度可分离卷积
    包含: Depthwise Conv -> Pointwise Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise: groups = in_channels, 每个通道独立卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=bias)
        # Pointwise: 1x1 卷积，负责混合通道信息
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Litenet(nn.Module):
    def __init__(self, num_classes=9):
        super(Litenet, self).__init__()
        
        # ===== Block 1 =====
        # 输入: (Batch, 3, 128, 128)
        # SeparableConv2d(3->16, 9x9, padding=0)
        # 输出: (Batch, 16, 120, 120) | 尺寸计算: (128-9)/1+1 = 120
        self.block1_conv = SeparableConv2d(3, 16, kernel_size=9)
        self.block1_bn = nn.BatchNorm2d(16)
        # MaxPool2d(4x4, stride=4)
        # 输出: (Batch, 16, 30, 30) | 尺寸计算: (120-4)/4+1 = 30
        self.block1_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # ===== Block 2 =====
        # 输入: (Batch, 16, 30, 30)
        # SeparableConv2d(16->32, 5x5, padding=0)
        # 输出: (Batch, 32, 26, 26) | 尺寸计算: (30-5)/1+1 = 26
        self.block2_conv = SeparableConv2d(16, 32, kernel_size=5)
        self.block2_bn = nn.BatchNorm2d(32)
        # MaxPool2d(3x3, stride=3)
        # 输出: (Batch, 32, 8, 8) | 尺寸计算: (26-3)/3+1 = 8.67 ≈ 8
        self.block2_pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # ===== Block 3 =====
        # 输入: (Batch, 32, 8, 8)
        # SeparableConv2d(32->128, 6x6, padding=0)
        # 输出: (Batch, 128, 3, 3) | 尺寸计算: (8-6)/1+1 = 3
        self.block3_conv = SeparableConv2d(32, 128, kernel_size=6)
        self.block3_bn = nn.BatchNorm2d(128)
        
        # ===== 尾部结构 =====
        # AdaptiveAvgPool2d(1x1)
        # 输入: (Batch, 128, 3, 3)
        # 输出: (Batch, 128, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout(p=0.2)
        # 输入: (Batch, 128, 1, 1)
        # 输出: (Batch, 128, 1, 1)
        self.dropout = nn.Dropout(p=0.2)
        
        # Linear(128 -> num_classes=9)
        # 输入: (Batch, 128)
        # 输出: (Batch, 9)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入: (Batch, 3, 128, 128)
        
        # --- Block 1 ---
        x = self.block1_conv(x)
        # SeparableConv2d(3->16, 9x9, padding=0)
        # 输出: (Batch, 16, 120, 120) | 计算: (128-9+0)/1+1 = 120
        
        x = self.block1_bn(x)
        # BatchNorm2d
        # 输出: (Batch, 16, 120, 120)
        
        x = F.relu(x)
        # ReLU 激活
        # 输出: (Batch, 16, 120, 120)
        
        x = self.block1_pool(x)
        # MaxPool2d(4x4, stride=4)
        # 输出: (Batch, 16, 30, 30) | 计算: (120-4)/4+1 = 30
        
        # --- Block 2 ---
        x = self.block2_conv(x)
        # SeparableConv2d(16->32, 5x5, padding=0)
        # 输出: (Batch, 32, 26, 26) | 计算: (30-5+0)/1+1 = 26
        
        x = self.block2_bn(x)
        # BatchNorm2d
        # 输出: (Batch, 32, 26, 26)
        
        x = F.relu(x)
        # ReLU 激活
        # 输出: (Batch, 32, 26, 26)
        
        x = self.block2_pool(x)
        # MaxPool2d(3x3, stride=3)
        # 输出: (Batch, 32, 8, 8) | 计算: (26-3)/3+1 = 8.67 ≈ 8
        
        # --- Block 3 ---
        x = self.block3_conv(x)
        # SeparableConv2d(32->128, 6x6, padding=0)
        # 输出: (Batch, 128, 3, 3) | 计算: (8-6+0)/1+1 = 3
        
        x = self.block3_bn(x)
        # BatchNorm2d
        # 输出: (Batch, 128, 3, 3)
        
        x = F.relu(x)
        # ReLU 激活
        # 输出: (Batch, 128, 3, 3)
        
        # --- Global Average Pooling ---
        x = self.global_avg_pool(x)
        # AdaptiveAvgPool2d(1x1)
        # 输出: (Batch, 128, 1, 1)
        
        # --- Flatten ---
        x = torch.flatten(x, 1)
        # 展平操作
        # 输出: (Batch, 128)
        
        # --- Dropout ---
        x = self.dropout(x)
        # Dropout(p=0.2)
        # 输出: (Batch, 128)
        
        # --- Classification ---
        x = self.classifier(x)
        # Linear(128 -> num_classes=9)
        # 输出: (Batch, 9)
        
        return x

# --- 实例化与测试代码 ---
if __name__ == "__main__":
    model = Litenet(num_classes=9)
    
    # 创建一个模拟输入张量 (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 128, 128)
    
    print("=" * 80)
    print("Litenet 网络结构分析")
    print("=" * 80)
    print(f"\n输入尺寸: {dummy_input.shape}")
    print(f"  - Batch Size: 1")
    print(f"  - Channels: 3 (RGB)")
    print(f"  - Height × Width: 128 × 128\n")
    
    print("=" * 80)
    print("各层输出尺寸")
    print("=" * 80)
    
    # 逐层跟踪
    x = dummy_input
    print(f"\n① 输入层:")
    print(f"   Shape: {x.shape}")
    
    x = model.block1_conv(x)
    print(f"\n② Block1 - SeparableConv2d (3→16, 9×9):")
    print(f"   Shape: {x.shape}")
    
    x = model.block1_bn(x)
    print(f"   After BatchNorm: {x.shape}")
    
    x = F.relu(x)
    print(f"   After ReLU: {x.shape}")
    
    x = model.block1_pool(x)
    print(f"   After MaxPool (4×4): {x.shape}")
    
    x = model.block2_conv(x)
    print(f"\n③ Block2 - SeparableConv2d (16→32, 5×5):")
    print(f"   Shape: {x.shape}")
    
    x = model.block2_bn(x)
    print(f"   After BatchNorm: {x.shape}")
    
    x = F.relu(x)
    print(f"   After ReLU: {x.shape}")
    
    x = model.block2_pool(x)
    print(f"   After MaxPool (3×3): {x.shape}")
    
    x = model.block3_conv(x)
    print(f"\n④ Block3 - SeparableConv2d (32→128, 6×6):")
    print(f"   Shape: {x.shape}")
    
    x = model.block3_bn(x)
    print(f"   After BatchNorm: {x.shape}")
    
    x = F.relu(x)
    print(f"   After ReLU: {x.shape}")
    
    x = model.global_avg_pool(x)
    print(f"\n⑤ Global Average Pooling (1×1):")
    print(f"   Shape: {x.shape}")
    
    x = torch.flatten(x, 1)
    print(f"\n⑥ Flatten:")
    print(f"   Shape: {x.shape}")
    
    x = model.dropout(x)
    print(f"\n⑦ Dropout (p=0.2):")
    print(f"   Shape: {x.shape}")
    
    x = model.classifier(x)
    print(f"\n⑧ 输出层 - Linear (128→9):")
    print(f"   Shape: {x.shape}")
    
    print("\n" + "=" * 80)
    
    # 模型统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    print("\n网络结构信息:")
    print(f"  - 输入尺寸: 128×128×3")
    print(f"  - 输出类别数: 9")
    print(f"  - 模型类型: 轻量级 CNN (Litenet)")
    print(f"  - 使用模块: 深度可分离卷积 (Depthwise Separable Conv)")
    
    print("\n" + "=" * 80)