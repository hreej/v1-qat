"""
使用预训练模型的网络架构
适用于小样本数据集，利用 ImageNet 预训练权重
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LitenetResNet(nn.Module):
    """
    使用预训练 ResNet18 作为骨干网络
    
    架构说明：
    - 骨干网络：ResNet18 (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~11M (预训练) + 少量分类层参数
    
    适用场景：
    - 原始数据量较少（<500张/类）
    - 需要快速达到较高准确率
    - 特征提取能力要求高
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
                - True: 只训练分类层，速度快但效果可能受限
                - False: 微调整个网络，效果更好但需要更多时间
        """
        super(LitenetResNet, self).__init__()
        
        # 加载预训练的 ResNet18
        print("正在加载预训练的 ResNet18...")
        self.backbone = models.resnet18(pretrained=True)
        print("✓ ResNet18 预训练权重加载完成")
        
        # 可选：冻结预训练层（前期快速收敛）
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换最后的分类层
        # ResNet18 的 fc.in_features = 512
        in_features = self.backbone.fc.in_features
        
        # 自定义分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状:
            x: (B, 3, 128, 128)
        
        输出形状:
            output: (B, num_classes)
        
        网络流程:
            输入 (B, 3, 128, 128)
              ↓ ResNet18 Conv1+BN+ReLU+MaxPool
            (B, 64, 32, 32)
              ↓ ResNet18 Layer1
            (B, 64, 32, 32)
              ↓ ResNet18 Layer2
            (B, 128, 16, 16)
              ↓ ResNet18 Layer3
            (B, 256, 8, 8)
              ↓ ResNet18 Layer4
            (B, 512, 4, 4)
              ↓ AvgPool
            (B, 512)
              ↓ 自定义分类层
            (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class LitenetEfficientNet(nn.Module):
    """
    使用预训练 EfficientNet-B0（更轻量）
    
    架构说明：
    - 骨干网络：EfficientNet-B0 (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~5M (比 ResNet18 更少)
    
    优势：
    - 参数量更少，推理更快
    - 准确率通常与 ResNet18 相当或更好
    - 更适合移动端/边缘设备部署
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetEfficientNet, self).__init__()
        
        # 加载预训练的 EfficientNet-B0
        print("正在加载预训练的 EfficientNet-B0...")
        self.backbone = models.efficientnet_b0(pretrained=True)
        print("✓ EfficientNet-B0 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类器
        # EfficientNet-B0 的 classifier[1].in_features = 1280
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class LitenetDenseNet(nn.Module):
    """
    使用预训练 DenseNet121
    
    架构说明：
    - 骨干网络：DenseNet121 (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~8M
    
    优势：
    - 密集连接，特征复用效率高
    - 梯度流动更顺畅
    - 参数效率高
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetDenseNet, self).__init__()
        
        # 加载预训练的 DenseNet121
        print("正在加载预训练的 DenseNet121...")
        self.backbone = models.densenet121(pretrained=True)
        print("✓ DenseNet121 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类器
        # DenseNet121 的 classifier.in_features = 1024
        in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class LitenetGoogleNet(nn.Module):
    """
    使用预训练 GoogLeNet (Inception v1)
    
    架构说明：
    - 骨干网络：GoogLeNet (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~6M
    
    优势：
    - Inception 模块，多尺度特征提取
    - 计算效率高
    - 经典网络架构
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetGoogleNet, self).__init__()
        
        # 加载预训练的 GoogLeNet
        print("正在加载预训练的 GoogLeNet...")
        self.backbone = models.googlenet(pretrained=True)
        print("✓ GoogLeNet 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类器
        # GoogLeNet 的 fc.in_features = 1024
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class LitenetMobileNet(nn.Module):
    """
    使用预训练 MobileNet v2
    
    架构说明：
    - 骨干网络：MobileNet v2 (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~3.5M
    
    优势：
    - 极致轻量化，专为移动端设计
    - 推理速度快
    - 适合边缘设备部署
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetMobileNet, self).__init__()
        
        # 加载预训练的 MobileNet v2
        print("正在加载预训练的 MobileNet v2...")
        self.backbone = models.mobilenet_v2(pretrained=True)
        print("✓ MobileNet v2 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换分类器
        # MobileNet v2 的 classifier[1].in_features = 1280
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.parameters():
            param.requires_grad = True


class LitenetSqueezeNet(nn.Module):
    """
    使用预训练 SqueezeNet 1.1
    
    架构说明：
    - 骨干网络：SqueezeNet 1.1 (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~1.2M (非常轻量)
    
    优势：
    - Fire Module 设计，参数极少
    - 模型体积小，适合嵌入式
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetSqueezeNet, self).__init__()
        
        # 加载预训练的 SqueezeNet 1.1
        print("正在加载预训练的 SqueezeNet 1.1...")
        self.backbone = models.squeezenet1_1(pretrained=True)
        print("✓ SqueezeNet 1.1 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # SqueezeNet 的分类器是 Conv2d
        # classifier[1] 是 Conv2d(512, 1000, kernel_size=(1,1))
        self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        self.backbone.num_classes = num_classes
        
        print(f"✓ 分类层已替换: Conv2d(512, {num_classes}, 1x1)")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.features.parameters():
            param.requires_grad = True


class LitenetAlexNet(nn.Module):
    """
    使用预训练 AlexNet
    
    架构说明：
    - 骨干网络：AlexNet (在 ImageNet 上预训练)
    - 输入尺寸：(B, 3, 128, 128)
    - 输出尺寸：(B, num_classes)
    - 参数量：~60M (原始) -> 大幅减少 (替换分类器后)
    
    优势：
    - 结构简单，经典的 CNN 架构
    - 卷积核较大，感受野大
    """
    
    def __init__(self, num_classes=12, freeze_backbone=False):
        """
        Args:
            num_classes (int): 分类类别数
            freeze_backbone (bool): 是否冻结预训练层
        """
        super(LitenetAlexNet, self).__init__()
        
        # 加载预训练的 AlexNet
        print("正在加载预训练的 AlexNet...")
        self.backbone = models.alexnet(pretrained=True)
        print("✓ AlexNet 预训练权重加载完成")
        
        # 可选：冻结预训练层
        if freeze_backbone:
            print("冻结预训练层，只训练分类器...")
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # 替换分类器
        # AlexNet 的 classifier 输入是 256 * 6 * 6 = 9216
        # 原始结构: 9216 -> 4096 -> 4096 -> num_classes
        # 我们替换为更轻量的: 9216 -> 256 -> num_classes
        in_features = 9216
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ 分类层已替换: {in_features} → 256 → {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        输入形状: (B, 3, 128, 128)
        输出形状: (B, num_classes)
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """解冻预训练层，用于第二阶段微调"""
        print("解冻预训练层，开始微调...")
        for param in self.backbone.features.parameters():
            param.requires_grad = True



# --- 测试代码 ---
if __name__ == "__main__":
    import torch
    
    # 尝试导入 thop 库计算 FLOPs
    try:
        from thop import profile, clever_format
        has_thop = True
    except ImportError:
        has_thop = False
        print("⚠️  未安装 thop 库，无法计算 FLOPs")
        print("   安装命令: pip install thop")
    
    print("=" * 80)
    print("测试预训练模型")
    print("=" * 80 + "\n")
    
    # 测试 ResNet18
    print("【1】测试 ResNet18")
    print("-" * 80)
    model_resnet = LitenetResNet(num_classes=12, freeze_backbone=False)
    
    # 创建模拟输入
    dummy_input = torch.randn(4, 3, 128, 128)
    print(f"输入形状: {dummy_input.shape}")
    
    # 前向传播
    output = model_resnet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model_resnet.parameters())
    trainable_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_resnet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 计算 FLOPs
    if has_thop:
        model_resnet_cpu = LitenetResNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_resnet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")
    
    print("\n" + "=" * 80)
    print("【2】测试 EfficientNet-B0")
    print("-" * 80)
    model_efficient = LitenetEfficientNet(num_classes=12, freeze_backbone=False)
    
    output = model_efficient(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_efficient.parameters())
    trainable_params = sum(p.numel() for p in model_efficient.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_efficient.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 计算 FLOPs
    if has_thop:
        model_efficient_cpu = LitenetEfficientNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_efficient_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")
    
    print("\n" + "=" * 80)
    print("【3】测试 DenseNet121")
    print("-" * 80)
    model_densenet = LitenetDenseNet(num_classes=12, freeze_backbone=False)
    
    output = model_densenet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_densenet.parameters())
    trainable_params = sum(p.numel() for p in model_densenet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_densenet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if has_thop:
        model_densenet_cpu = LitenetDenseNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_densenet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")
    
    print("\n" + "=" * 80)
    print("【4】测试 GoogLeNet")
    print("-" * 80)
    model_googlenet = LitenetGoogleNet(num_classes=12, freeze_backbone=False)
    
    output = model_googlenet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_googlenet.parameters())
    trainable_params = sum(p.numel() for p in model_googlenet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_googlenet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if has_thop:
        model_googlenet_cpu = LitenetGoogleNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_googlenet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")
    
    print("\n" + "=" * 80)
    print("【5】测试 MobileNet v2")
    print("-" * 80)
    model_mobilenet = LitenetMobileNet(num_classes=12, freeze_backbone=False)
    
    output = model_mobilenet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_mobilenet.parameters())
    trainable_params = sum(p.numel() for p in model_mobilenet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_mobilenet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if has_thop:
        model_mobilenet_cpu = LitenetMobileNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_mobilenet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")
    
    print("\n" + "=" * 80)
    print("【6】测试 SqueezeNet")
    print("-" * 80)
    model_squeezenet = LitenetSqueezeNet(num_classes=12, freeze_backbone=False)
    
    output = model_squeezenet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_squeezenet.parameters())
    trainable_params = sum(p.numel() for p in model_squeezenet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_squeezenet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if has_thop:
        model_squeezenet_cpu = LitenetSqueezeNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_squeezenet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")

    print("\n" + "=" * 80)
    print("【7】测试 AlexNet")
    print("-" * 80)
    model_alexnet = LitenetAlexNet(num_classes=12, freeze_backbone=False)
    
    output = model_alexnet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_alexnet.parameters())
    trainable_params = sum(p.numel() for p in model_alexnet.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model_alexnet.buffers())
    total_size = total_params + total_buffers
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"Buffer 数量: {total_buffers:,}")
    print(f"总大小 (参数+Buffer): {total_size:,}")
    print(f"模型大小: {total_size * 4 / 1024 / 1024:.2f} MB (float32)")
    
    if has_thop:
        model_alexnet_cpu = LitenetAlexNet(num_classes=12, freeze_backbone=False)
        input_single = torch.randn(1, 3, 128, 128)
        flops, params = profile(model_alexnet_cpu, inputs=(input_single,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量 (Params): {params}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
