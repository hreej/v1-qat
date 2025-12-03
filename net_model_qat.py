import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

# 我们需要重写 SeparableConv2d 以便能够访问内部的层进行融合
class SeparableConv2d_QAT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SeparableConv2d_QAT, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Litenet_QAT(nn.Module):
    def __init__(self, num_classes=12):
        super(Litenet_QAT, self).__init__()
        
        # 量化桩：用于转换浮点输入到量化空间
        self.quant = QuantStub()

        # Block 1
        self.block1_conv = SeparableConv2d_QAT(3, 16, kernel_size=9)
        self.block1_bn = nn.BatchNorm2d(16)
        self.block1_relu = nn.ReLU(inplace=True) # 必须用 nn.Module
        self.block1_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        # Block 2
        self.block2_conv = SeparableConv2d_QAT(16, 32, kernel_size=5)
        self.block2_bn = nn.BatchNorm2d(32)
        self.block2_relu = nn.ReLU(inplace=True)
        self.block2_pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # Block 3
        self.block3_conv = SeparableConv2d_QAT(32, 128, kernel_size=6)
        self.block3_bn = nn.BatchNorm2d(128)
        self.block3_relu = nn.ReLU(inplace=True)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(128, num_classes)
        
        # 反量化桩：用于输出浮点结果（Softmax前）
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 1. 浮点 -> INT8
        x = self.quant(x)
        
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_relu(x)
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_relu(x)
        x = self.block2_pool(x)
        
        # Block 3
        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.block3_relu(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        # 2. INT8 -> 浮点
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        QAT 关键步骤：定义哪些层由于物理连接紧密，可以被融合计算。
        结构: Separable(DW->PW) -> BN -> ReLU
        融合链: PW卷积 + BN + ReLU
        """
        import torch.quantization
        
        # 融合 Block 1: pointwise + bn + relu
        torch.quantization.fuse_modules(self, 
            ['block1_conv.pointwise', 'block1_bn', 'block1_relu'], 
            inplace=True)
            
        # 融合 Block 2
        torch.quantization.fuse_modules(self, 
            ['block2_conv.pointwise', 'block2_bn', 'block2_relu'], 
            inplace=True)
            
        # 融合 Block 3
        torch.quantization.fuse_modules(self, 
            ['block3_conv.pointwise', 'block3_bn', 'block3_relu'], 
            inplace=True)