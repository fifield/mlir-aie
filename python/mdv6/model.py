"""MDV6-mit-yolov9-c model implementation (forward path only)."""

import torch
import torch.nn as nn

try:
    from .layers import (Conv, AConv, ELAN, RepNCSPELAN, SPPELAN, MultiheadDetection)
except ImportError:
    from layers import (Conv, AConv, ELAN, RepNCSPELAN, SPPELAN, MultiheadDetection)


class MDV6MITYOLOv9c(nn.Module):
    """
    MegaDetectorV6 MIT YOLOv9-c implementation (forward/main path only).
    
    Based on Microsoft CameraTraps implementation for wildlife detection.
    Detects 3 classes: animal, person, vehicle
    
    Architecture follows config_v9s.yaml:
    - Backbone: GELAN with ELAN and RepNCSPELAN blocks
    - Neck: SPPELAN + feature fusion
    - Head: MultiheadDetection (main path only, no auxiliary)
    
    Channel progression: 32 → 64 → 128 → 192 → 256
    """
    
    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }
    
    def __init__(self, num_classes=3, input_channels=3):
        super().__init__()
        self.num_classes = num_classes
        
        # ===== BACKBONE =====
        # Layer 0-1: Initial convolutions
        self.conv0 = Conv(input_channels, 32, 3, stride=2)  # 3 -> 32, /2
        self.conv1 = Conv(32, 64, 3, stride=2)  # 32 -> 64, /4
        
        # Layer 2: ELAN block
        self.elan2 = ELAN(64, 64, part_channels=64)  # 64 -> 64
        
        # Layer 3-4: AConv + RepNCSPELAN (B3)
        self.aconv3 = AConv(64, 128)  # 64 -> 128, /8
        self.rep_elan4 = RepNCSPELAN(128, 128, part_channels=128, 
                                     csp_args={'repeat_num': 3})  # 128 -> 128 (B3)
        
        # Layer 5-6: AConv + RepNCSPELAN (B4)
        self.aconv5 = AConv(128, 192)  # 128 -> 192, /16
        self.rep_elan6 = RepNCSPELAN(192, 192, part_channels=192,
                                     csp_args={'repeat_num': 3})  # 192 -> 192 (B4)
        
        # Layer 7-8: AConv + RepNCSPELAN (B5)
        self.aconv7 = AConv(192, 256)  # 192 -> 256, /32
        self.rep_elan8 = RepNCSPELAN(256, 256, part_channels=256,
                                     csp_args={'repeat_num': 3})  # 256 -> 256 (B5)
        
        # ===== NECK =====
        # Layer 9-11: SPPELAN + Upsample + Concat
        self.spp9 = SPPELAN(256, 256)  # 256 -> 256 (N3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Concat with B4 (192): 256 + 192 = 448
        
        # Layer 12: RepNCSPELAN
        self.rep_elan12 = RepNCSPELAN(448, 192, part_channels=192,
                                      csp_args={'repeat_num': 3})  # 448 -> 192 (N4)
        
        # ===== HEAD (Main Path) =====
        # Layer 13-15: Upsample + Concat + RepNCSPELAN
        # Concat with B3 (128): 192 + 128 = 320
        self.rep_elan15 = RepNCSPELAN(320, 128, part_channels=128,
                                      csp_args={'repeat_num': 3})  # 320 -> 128 (P3)
        
        # Layer 16-18: AConv + Concat + RepNCSPELAN
        self.aconv16 = AConv(128, 96)  # 128 -> 96
        # Concat with N4 (192): 96 + 192 = 288
        self.rep_elan18 = RepNCSPELAN(288, 192, part_channels=192,
                                      csp_args={'repeat_num': 3})  # 288 -> 192 (P4)
        
        # Layer 19-21: AConv + Concat + RepNCSPELAN
        self.aconv19 = AConv(192, 128)  # 192 -> 128
        # Concat with N3 (256): 128 + 256 = 384
        self.rep_elan21 = RepNCSPELAN(384, 256, part_channels=256,
                                      csp_args={'repeat_num': 3})  # 384 -> 256 (P5)
        
        # ===== DETECTION HEAD =====
        # MultiheadDetection for P3, P4, P5
        self.detect = MultiheadDetection(
            in_channels=[128, 192, 256],  # P3, P4, P5
            num_classes=num_classes,
            reg_max=16,
            use_group=True
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass (main path only, no auxiliary).
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            List of detection outputs for 3 scales:
            [(class_p3, anchor_p3, vector_p3), 
             (class_p4, anchor_p4, vector_p4),
             (class_p5, anchor_p5, vector_p5)]
        """
        # ===== BACKBONE =====
        x = self.conv0(x)  # (B, 32, H/2, W/2)
        x = self.conv1(x)  # (B, 64, H/4, W/4)
        x = self.elan2(x)  # (B, 64, H/4, W/4)
        
        x = self.aconv3(x)  # (B, 128, H/8, W/8)
        b3 = self.rep_elan4(x)  # (B, 128, H/8, W/8) - Save for concat
        
        x = self.aconv5(b3)  # (B, 192, H/16, W/16)
        b4 = self.rep_elan6(x)  # (B, 192, H/16, W/16) - Save for concat
        
        x = self.aconv7(b4)  # (B, 256, H/32, W/32)
        b5 = self.rep_elan8(x)  # (B, 256, H/32, W/32) - Save for concat
        
        # ===== NECK =====
        n3 = self.spp9(b5)  # (B, 256, H/32, W/32)
        
        x = self.upsample(n3)  # (B, 256, H/16, W/16)
        x = torch.cat([x, b4], dim=1)  # (B, 448, H/16, W/16)
        n4 = self.rep_elan12(x)  # (B, 192, H/16, W/16)
        
        # ===== HEAD (Main Path) =====
        # P3 branch
        x = self.upsample(n4)  # (B, 192, H/8, W/8)
        x = torch.cat([x, b3], dim=1)  # (B, 320, H/8, W/8)
        p3 = self.rep_elan15(x)  # (B, 128, H/8, W/8)
        
        # P4 branch
        x = self.aconv16(p3)  # (B, 96, H/16, W/16)
        x = torch.cat([x, n4], dim=1)  # (B, 288, H/16, W/16)
        p4 = self.rep_elan18(x)  # (B, 192, H/16, W/16)
        
        # P5 branch
        x = self.aconv19(p4)  # (B, 128, H/32, W/32)
        x = torch.cat([x, n3], dim=1)  # (B, 384, H/32, W/32)
        p5 = self.rep_elan21(x)  # (B, 256, H/32, W/32)
        
        # ===== DETECTION =====
        detections = self.detect([p3, p4, p5])
        
        return detections
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def print_model_info(self):
        """Print model information."""
        param_counts = self.count_parameters()
        print(f"\n{'='*70}")
        print(f"Model: MDV6-mit-yolov9-c (Main Path Only)")
        print(f"{'='*70}")
        print(f"Total Parameters:      {param_counts['total']:,}")
        print(f"Trainable Parameters:  {param_counts['trainable']:,}")
        print(f"{'='*70}")
        print(f"Number of Classes:     {self.num_classes}")
        print(f"Class Names:           {list(self.CLASS_NAMES.values())}")
        print(f"Channel Progression:   32 → 64 → 128 → 192 → 256")
        print(f"Detection Scales:      3 (P3, P4, P5)")
        print(f"{'='*70}\n")


def create_mdv6_mit_yolov9c(num_classes=3, pretrained=False, weights_path=None):
    """
    Create MDV6-mit-yolov9-c model.
    
    Args:
        num_classes: Number of classes (default 3 for animal/person/vehicle)
        pretrained: Whether to load pretrained weights
        weights_path: Path to pretrained weights file (.ckpt)
    
    Returns:
        MDV6MITYOLOv9c model
    """
    model = MDV6MITYOLOv9c(num_classes=num_classes)
    
    if pretrained and weights_path:
        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Extract state dict from Lightning checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'model.model.' prefix if present
            if any(k.startswith('model.model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.model.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model
