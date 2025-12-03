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
    
    def __init__(self, num_classes=9, freeze_backbone=False):
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
    
    def __init__(self, num_classes=9, freeze_backbone=False):
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


# --- 测试代码 ---
if __name__ == "__main__":
    import torch
    
    print("=" * 80)
    print("测试预训练模型")
    print("=" * 80 + "\n")
    
    # 测试 ResNet18
    print("【1】测试 ResNet18")
    print("-" * 80)
    model_resnet = LitenetResNet(num_classes=9, freeze_backbone=False)
    
    # 创建模拟输入
    dummy_input = torch.randn(4, 3, 128, 128)
    print(f"输入形状: {dummy_input.shape}")
    
    # 前向传播
    output = model_resnet(dummy_input)
    print(f"输出形状: {output.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model_resnet.parameters())
    trainable_params = sum(p.numel() for p in model_resnet.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n" + "=" * 80)
    print("【2】测试 EfficientNet-B0")
    print("-" * 80)
    model_efficient = LitenetEfficientNet(num_classes=9, freeze_backbone=False)
    
    output = model_efficient(dummy_input)
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model_efficient.parameters())
    trainable_params = sum(p.numel() for p in model_efficient.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
