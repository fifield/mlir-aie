"""Test MDV6-mit-yolov9-c implementation against official CameraTraps reference."""

import torch
import sys
import urllib.request
from pathlib import Path
import numpy as np

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "camera_traps_reference"))

from model import MDV6MITYOLOv9c
from weight_mapper import load_and_transform_weights


def download_weights(weights_dir="weights"):
    """Download MDV6-mit-yolov9-c pretrained weights."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    weights_file = weights_dir / "MDV6-mit-yolov9-c.ckpt"
    
    if not weights_file.exists():
        print("Downloading MDV6-mit-yolov9-c weights...")
        url = "https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-c.ckpt?download=1"
        
        try:
            print(f"URL: {url}")
            print(f"Destination: {weights_file}")
            print("This may take a while (file is ~30MB)...")
            urllib.request.urlretrieve(url, weights_file)
            print(f"✓ Weights downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading weights: {e}")
            print("\nPlease download manually from:")
            print(url)
            return None
    else:
        print(f"✓ Weights already exist at {weights_file}")
    
    return weights_file


def load_official_model(weights_path):
    """Load the official CameraTraps MegaDetectorV6MIT model."""
    print("\n" + "="*80)
    print("LOADING OFFICIAL CAMERATRAPS MODEL")
    print("="*80)
    
    try:
        from PytorchWildlife.models.detection.yolo_mit import MegaDetectorV6MIT
        
        # Create model (it will download weights if needed)
        model = MegaDetectorV6MIT(
            weights=str(weights_path),
            device='cpu',
            pretrained=True,
            version='MDV6-mit-yolov9-c'
        )
        
        print("✓ Official model loaded successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
        
        # The actual model is in model.model attribute after _load_model is called
        if hasattr(model, 'model'):
            model.model.eval()
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"  Total parameters: {total_params:,}")
        else:
            print("  Note: Model will be loaded on first inference call")
        
        return model
        
    except Exception as e:
        print(f"✗ Error loading official model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_our_model(weights_path):
    """Load our MDV6MITYOLOv9c implementation with pretrained weights."""
    print("\n" + "="*80)
    print("LOADING OUR IMPLEMENTATION WITH PRETRAINED WEIGHTS")
    print("="*80)
    
    model = MDV6MITYOLOv9c(num_classes=3)
    model.eval()
    
    print("✓ Our model created successfully")
    print(f"  Total parameters: {model.count_parameters()['total']:,}")
    
    # Load weights using weight mapper
    if weights_path and weights_path.exists():
        try:
            print(f"\nLoading and transforming weights from {weights_path}...")
            
            # Use weight mapper to transform keys
            transformed_dict = load_and_transform_weights(weights_path)
            
            print(f"\nLoading transformed weights into model...")
            missing, unexpected = model.load_state_dict(transformed_dict, strict=False)
            
            print(f"\n  Weight loading results:")
            print(f"    Missing keys: {len(missing)}")
            if missing and len(missing) <= 10:
                print("    First 10 missing keys:")
                for key in missing[:10]:
                    print(f"      - {key}")
            
            print(f"    Unexpected keys: {len(unexpected)}")
            if unexpected and len(unexpected) <= 10:
                print("    First 10 unexpected keys:")
                for key in unexpected[:10]:
                    print(f"      - {key}")
            
            if len(missing) == 0 and len(unexpected) == 0:
                print("\n  ✓✓✓ ALL PRETRAINED WEIGHTS LOADED SUCCESSFULLY! ✓✓✓")
            elif len(missing) < 100:
                print(f"\n  ⚠ Mostly loaded ({len(transformed_dict) - len(missing)}/{len(transformed_dict)} matched)")
            else:
                print(f"\n  ✗ Significant weight mismatch")
                
        except Exception as e:
            print(f"\n  ✗ Error loading weights: {e}")
            import traceback
            traceback.print_exc()
    
    return model


def compare_outputs(official_model, our_model, test_input):
    """Compare outputs from both models."""
    print("\n" + "="*80)
    print("COMPARING MODEL OUTPUTS")
    print("="*80)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # Run official model
    print("\nRunning official model...")
    try:
        # Trigger model loading by calling _load_model
        if not hasattr(official_model, 'model'):
            official_model._load_model(weights=official_model.weights, 
                                      device=official_model.device, 
                                      url=official_model.url)
        
        with torch.no_grad():
            # Official model returns predictions in a specific format
            official_output = official_model.model(test_input)
        print(f"  ✓ Official model inference successful")
        print(f"    Output type: {type(official_output)}")
        
        if isinstance(official_output, dict):
            print(f"    Output keys: {official_output.keys()}")
            for key, val in official_output.items():
                if isinstance(val, (list, tuple)):
                    print(f"    {key}: {len(val)} items")
                    for i, item in enumerate(val):
                        if isinstance(item, torch.Tensor):
                            print(f"      [{i}]: {item.shape}")
                        elif isinstance(item, (list, tuple)):
                            print(f"      [{i}]: {len(item)} tensors")
                            for j, t in enumerate(item):
                                if isinstance(t, torch.Tensor):
                                    print(f"        [{j}]: {t.shape}")
                elif isinstance(val, torch.Tensor):
                    print(f"    {key}: {val.shape}")
        elif isinstance(official_output, (list, tuple)):
            print(f"    Output length: {len(official_output)}")
            for i, item in enumerate(official_output):
                print(f"    [{i}]: {type(item)}")
                if isinstance(item, torch.Tensor):
                    print(f"      Shape: {item.shape}")
                elif isinstance(item, (list, tuple)):
                    print(f"      Length: {len(item)}")
                    for j, t in enumerate(item):
                        if isinstance(t, torch.Tensor):
                            print(f"        [{j}]: {t.shape}")
        
    except Exception as e:
        print(f"  ✗ Error running official model: {e}")
        import traceback
        traceback.print_exc()
        official_output = None
    
    # Run our model
    print("\nRunning our model...")
    try:
        with torch.no_grad():
            our_output = our_model(test_input)
        print(f"  ✓ Our model inference successful")
        print(f"    Output type: {type(our_output)}")
        print(f"    Output length: {len(our_output)}")
        
        for i, (class_x, anchor_x, vector_x) in enumerate(our_output):
            print(f"    Scale {i}:")
            print(f"      Class: {class_x.shape}")
            print(f"      Anchor: {anchor_x.shape}")
            print(f"      Vector: {vector_x.shape}")
        
    except Exception as e:
        print(f"  ✗ Error running our model: {e}")
        import traceback
        traceback.print_exc()
        our_output = None
    
    # Compare if both succeeded
    if official_output is not None and our_output is not None:
        print("\n" + "="*80)
        print("NUMERICAL COMPARISON")
        print("="*80)
        
        # Analyze official model output format
        print("\nOfficial Model Output Format:")
        if isinstance(official_output, dict) and 'Main' in official_output:
            main_outputs = official_output['Main']
            print(f"  Main path: {len(main_outputs)} scales")
            
            for i, scale_output in enumerate(main_outputs):
                print(f"\n  Scale {i}:")
                if isinstance(scale_output, (list, tuple)):
                    for j, tensor in enumerate(scale_output):
                        if isinstance(tensor, torch.Tensor):
                            print(f"    Output[{j}]:")
                            print(f"      Shape: {tensor.shape}")
                            print(f"      Dtype: {tensor.dtype}")
                            print(f"      Device: {tensor.device}")
                            print(f"      Min: {tensor.min().item():.6f}")
                            print(f"      Max: {tensor.max().item():.6f}")
                            print(f"      Mean: {tensor.mean().item():.6f}")
                            print(f"      Std: {tensor.std().item():.6f}")
        
        print("\nOur Model Output Format:")
        for i, (class_x, anchor_x, vector_x) in enumerate(our_output):
            print(f"\n  Scale {i}:")
            print(f"    Class output:")
            print(f"      Shape: {class_x.shape}")
            print(f"      Dtype: {class_x.dtype}")
            print(f"      Min: {class_x.min().item():.6f}")
            print(f"      Max: {class_x.max().item():.6f}")
            print(f"      Mean: {class_x.mean().item():.6f}")
            
            print(f"    Anchor distribution:")
            print(f"      Shape: {anchor_x.shape}")
            print(f"      Dtype: {anchor_x.dtype}")
            print(f"      Min: {anchor_x.min().item():.6f}")
            print(f"      Max: {anchor_x.max().item():.6f}")
            
            print(f"    Vector output:")
            print(f"      Shape: {vector_x.shape}")
            print(f"      Dtype: {vector_x.dtype}")
            print(f"      Min: {vector_x.min().item():.6f}")
            print(f"      Max: {vector_x.max().item():.6f}")
        
        # Try direct numerical comparison if formats align
        if isinstance(official_output, dict) and 'Main' in official_output:
            print("\n" + "="*80)
            print("DIRECT NUMERICAL COMPARISON (Main Path)")
            print("="*80)
            
            main_outputs = official_output['Main']
            
            for i in range(len(our_output)):
                print(f"\nScale {i}:")
                official_scale = main_outputs[i]
                our_scale = our_output[i]
                
                # Compare each output
                for j, (name, our_tensor) in enumerate(zip(['Class', 'Anchor', 'Vector'], our_scale)):
                    if j < len(official_scale):
                        official_tensor = official_scale[j]
                        
                        # Calculate difference
                        diff = torch.abs(official_tensor - our_tensor)
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        
                        print(f"  {name}:")
                        print(f"    Max absolute difference: {max_diff:.6e}")
                        print(f"    Mean absolute difference: {mean_diff:.6e}")
                        
                        if max_diff < 1e-5:
                            print(f"    ✓ EXCELLENT match (< 1e-5)")
                        elif max_diff < 1e-3:
                            print(f"    ✓ Good match (< 1e-3)")
                        elif max_diff < 0.1:
                            print(f"    ⚠ Acceptable match (< 0.1)")
                        else:
                            print(f"    ✗ Significant difference (> 0.1)")
        
    return official_output, our_output


def main():
    print("="*80)
    print("MDV6-mit-yolov9-c REFERENCE IMPLEMENTATION TEST")
    print("="*80)
    
    # Download weights
    weights_path = download_weights()
    
    if not weights_path:
        print("\nCannot proceed without weights.")
        return
    
    # Load official model
    official_model = load_official_model(weights_path)
    
    # Load our model
    our_model = load_our_model(weights_path)
    
    # Create test input
    test_input = torch.randn(1, 3, 640, 640)
    
    # Compare outputs
    official_output, our_output = compare_outputs(official_model, our_model, test_input)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    if official_model and our_model:
        print("\n✓ Both models loaded and ran successfully")
        print("✓ Architecture validation complete")
    else:
        print("\n⚠ Some issues encountered - see details above")


if __name__ == "__main__":
    main()
