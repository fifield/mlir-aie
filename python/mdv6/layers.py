"""Layer implementations for MDV6-mit-yolov9-c based on Microsoft CameraTraps."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any


def auto_pad(kernel_size, dilation=1):
    """Auto padding for convolution blocks."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def round_up(x, div=1):
    """Rounds up x to the nearest multiple of div."""
    return x + (-x % div)


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, 
                 dilation=1, groups=1, activation="SiLU"):
        super().__init__()
        if padding is None:
            padding = auto_pad(kernel_size, dilation)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation == "SiLU" else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AConv(nn.Module):
    """Average pooling downsampling with convolution."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.conv = Conv(in_channels, out_channels, 3, stride=2)
    
    def forward(self, x):
        x = self.avg_pool(x)
        return self.conv(x)


class RepConv(nn.Module):
    """Reparameterizable convolution combining 3x3 and 1x1 convs."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, groups=1, activation="SiLU"):
        super().__init__()
        self.act = nn.SiLU(inplace=True) if activation == "SiLU" else nn.Identity()
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, activation=None)
        self.conv2 = Conv(in_channels, out_channels, 1, stride, 
                         padding=0, groups=groups, activation=None)
    
    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    """Bottleneck block with optional residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), 
                 residual=True, expand=1.0):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0])
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1])
        self.residual = residual and (in_channels == out_channels)
    
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with RepConv bottlenecks."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, 
                 csp_expand=0.5, repeat_num=1, neck_args=None):
        super().__init__()
        neck_args = neck_args or {}
        neck_channels = int(out_channels * csp_expand)
        
        self.conv1 = Conv(in_channels, neck_channels, kernel_size)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size)
        
        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)]
        )
    
    def forward(self, x):
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class ELAN(nn.Module):
    """ELAN-1 structure."""
    
    def __init__(self, in_channels, out_channels, part_channels, process_channels=None):
        super().__init__()
        if process_channels is None:
            process_channels = part_channels // 2
        
        self.conv1 = Conv(in_channels, part_channels, 1)
        self.conv2 = Conv(part_channels // 2, process_channels, 3, padding=1)
        self.conv3 = Conv(process_channels, process_channels, 3, padding=1)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1)
    
    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP with ELAN structure."""
    
    def __init__(self, in_channels, out_channels, part_channels, 
                 process_channels=None, csp_args=None, csp_neck_args=None):
        super().__init__()
        csp_args = csp_args or {}
        csp_neck_args = csp_neck_args or {}
        
        if process_channels is None:
            process_channels = part_channels // 2
        
        self.conv1 = Conv(in_channels, part_channels, 1)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1)
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1)
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1)
    
    def forward(self, x):
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class SPPELAN(nn.Module):
    """SPPELAN module with multiple pooling and convolution layers."""
    
    def __init__(self, in_channels, out_channels, neck_channels=None):
        super().__init__()
        neck_channels = neck_channels or out_channels // 2
        
        self.conv1 = Conv(in_channels, neck_channels, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(3)
        ])
        self.conv5 = Conv(4 * neck_channels, out_channels, 1)
    
    def forward(self, x):
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class Anchor2Vec(nn.Module):
    """Anchor to vector conversion for DFL."""
    
    def __init__(self, reg_max=16):
        super().__init__()
        reverse_reg = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1, 1)
        self.anc2vec = nn.Conv3d(in_channels=reg_max, out_channels=1, kernel_size=1, bias=False)
        self.anc2vec.weight = nn.Parameter(reverse_reg, requires_grad=False)
        self.reg_max = reg_max
    
    def forward(self, anchor_x):
        """
        Args:
            anchor_x: (B, 4*reg_max, H, W)
        Returns:
            anchor_x: (B, reg_max, 4, H, W) - distribution
            vector_x: (B, 4, H, W) - expected values
        """
        B, PR, h, w = anchor_x.shape
        P = 4
        R = PR // P
        
        # Reshape to (B, P, R, H, W) then permute to (B, R, P, H, W)
        anchor_x = anchor_x.reshape(B, P, R, h, w).permute(0, 2, 1, 3, 4)
        vector_x = anchor_x.softmax(dim=1)
        vector_x = self.anc2vec(vector_x)[:, 0]
        
        return anchor_x, vector_x


class Detection(nn.Module):
    """Single YOLO detection head."""
    
    def __init__(self, in_channels, num_classes, reg_max=16, use_group=True):
        super().__init__()
        
        first_neck = in_channels[0] if isinstance(in_channels, (list, tuple)) else in_channels
        in_ch = in_channels[-1] if isinstance(in_channels, (list, tuple)) else in_channels
        
        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max
        
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        class_neck = max(first_neck, min(num_classes * 2, 128))
        
        self.anchor_conv = nn.Sequential(
            Conv(in_ch, anchor_neck, 3),
            Conv(anchor_neck, anchor_neck, 3, groups=groups),
            nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups)
        )
        
        self.class_conv = nn.Sequential(
            Conv(in_ch, class_neck, 3),
            Conv(class_neck, class_neck, 3),
            nn.Conv2d(class_neck, num_classes, 1)
        )
        
        self.anc2vec = Anchor2Vec(reg_max=reg_max)
        
        # Initialize biases
        self.anchor_conv[-1].bias.data.fill_(1.0)
        self.class_conv[-1].bias.data.fill_(-10)
    
    def forward(self, x):
        """
        Returns:
            class_x: (B, num_classes, H, W)
            anchor_x: (B, reg_max, 4, H, W)
            vector_x: (B, 4, H, W)
        """
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)
        return class_x, anchor_x, vector_x


class MultiheadDetection(nn.Module):
    """Multi-scale detection head."""
    
    def __init__(self, in_channels, num_classes, reg_max=16, use_group=True):
        super().__init__()
        self.heads = nn.ModuleList([
            Detection((in_channels[0], in_ch), num_classes, reg_max, use_group)
            for in_ch in in_channels
        ])
    
    def forward(self, x_list):
        """
        Args:
            x_list: List of feature maps [P3, P4, P5]
        Returns:
            List of tuples [(class_x, anchor_x, vector_x), ...]
        """
        return [head(x) for x, head in zip(x_list, self.heads)]
