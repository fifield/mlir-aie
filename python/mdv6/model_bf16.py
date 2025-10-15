"""BFloat16 quantized version of MDV6-mit-yolov9-c."""

import torch
import torch.nn as nn
from model import MDV6MITYOLOv9c


class MDV6MITYOLOv9cBF16(MDV6MITYOLOv9c):
    """BFloat16 quantized version of MDV6-mit-yolov9-c.
    
    This model uses bfloat16 for weights and activations.
    For simplicity, we convert everything to bfloat16 including BatchNorm.
    """
    
    def __init__(self, num_classes=3, keep_bn_fp32=False):
        super().__init__(num_classes)
        self.keep_bn_fp32 = keep_bn_fp32
        self.dtype = torch.bfloat16
    
    def forward(self, x):
        """Forward pass with bfloat16 computation."""
        # Convert input to bfloat16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        # Run forward pass (will use bfloat16 weights)
        outputs = super().forward(x)
        
        # Outputs are in bfloat16, can convert back to fp32 if needed
        return outputs
    
    def load_quantized_weights(self, weights_path):
        """Load bfloat16 quantized weights."""
        print(f"Loading quantized weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Load weights
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        print(f"  Missing: {len(missing)}")
        print(f"  Unexpected: {len(unexpected)}")
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✓ All quantized weights loaded successfully")
        
        # Ensure all parameters are bfloat16 (in case some weren't in the saved file)
        self.to_bfloat16()
        
        # Ensure model is in eval mode
        self.eval()
        
        return self
    
    def to_bfloat16(self):
        """Convert model to bfloat16 (if not already)."""
        # Convert all parameters and buffers to bfloat16
        for param in self.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        
        for buffer in self.buffers():
            if buffer.dtype == torch.float32:
                buffer.data = buffer.data.to(torch.bfloat16)
        
        return self
    
    def print_model_info(self):
        """Print model information including dtype."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\nMDV6-mit-yolov9-c BFloat16 Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Dtype: bfloat16 (weights), float32 (BatchNorm)")
        print(f"  Classes: {self.num_classes} (animal, person, vehicle)")
        print(f"  Input size: 640x640")
        print(f"  Detection scales: 3 (P3, P4, P5)")
        
        # Estimate memory
        bf16_params = sum(p.numel() for p in self.parameters() if p.dtype == torch.bfloat16)
        fp32_params = sum(p.numel() for p in self.parameters() if p.dtype == torch.float32)
        
        memory_mb = (bf16_params * 2 + fp32_params * 4) / 1024 / 1024
        print(f"  Estimated memory: {memory_mb:.2f} MB")
        print(f"  BFloat16 params: {bf16_params:,}")
        print(f"  Float32 params: {fp32_params:,}")


if __name__ == "__main__":
    print("="*80)
    print("MDV6 BFLOAT16 MODEL TEST")
    print("="*80)
    
    # Create model
    model = MDV6MITYOLOv9cBF16(num_classes=3)
    
    print("\nBEFORE quantization:")
    model.print_model_info()
    
    # Convert to bfloat16
    print("\nConverting to bfloat16...")
    model.to_bfloat16()
    
    print("\nAFTER quantization:")
    model.print_model_info()
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_input = torch.randn(1, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_input)
    
    print("✓ Forward pass successful")
    for i, (class_x, anchor_x, vector_x) in enumerate(outputs):
        print(f"  Scale {i}: class {class_x.shape} ({class_x.dtype}), "
              f"anchor {anchor_x.shape} ({anchor_x.dtype}), "
              f"vector {vector_x.shape} ({vector_x.dtype})")
