"""Test MDV6-mit-yolov9-c implementation."""

import torch
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model import MDV6MITYOLOv9c


def test_model_architecture():
    """Test the MDV6-mit-yolov9-c model architecture."""
    
    print("="*80)
    print("TESTING MDV6-mit-yolov9-c IMPLEMENTATION")
    print("="*80)
    
    # Create model
    model = MDV6MITYOLOv9c(num_classes=3)
    model.eval()
    
    # Print model info
    model.print_model_info()
    
    # Test with different input sizes
    test_sizes = [320, 640]
    
    for size in test_sizes:
        print(f"\nTesting with input size {size}x{size}:")
        img = torch.randn(1, 3, size, size)
        
        try:
            with torch.no_grad():
                output = model(img)
            
            print(f"  Input shape: {img.shape}")
            print(f"  Output type: {type(output)}")
            print(f"  Number of detection scales: {len(output)}")
            
            for i, (class_x, anchor_x, vector_x) in enumerate(output):
                print(f"  Scale {i} (P{i+3}):")
                print(f"    Class output: {class_x.shape}")
                print(f"    Anchor distribution: {anchor_x.shape}")
                print(f"    Vector output: {vector_x.shape}")
            
            print(f"  ✓ Forward pass successful!")
            
        except Exception as e:
            print(f"  ✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    return True


if __name__ == "__main__":
    success = test_model_architecture()
    sys.exit(0 if success else 1)
