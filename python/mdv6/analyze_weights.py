"""Analyze MDV6-mit-yolov9-c weights structure and create key mapping."""

import torch
from pathlib import Path


def analyze_weights(weights_path):
    """Analyze the structure of MDV6-mit-yolov9-c weights."""
    
    print("="*80)
    print("ANALYZING MDV6-mit-yolov9-c WEIGHTS")
    print("="*80)
    
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    print(f"PyTorch Lightning version: {checkpoint.get('pytorch-lightning_version', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Global step: {checkpoint.get('global_step', 'N/A')}")
    
    # Extract state dict
    state_dict = checkpoint['state_dict']
    
    print(f"\nTotal parameters in state_dict: {len(state_dict)}")
    
    # Group by layer index
    layer_groups = {}
    for key in state_dict.keys():
        if key.startswith('model.model.'):
            # Extract layer number
            parts = key.split('.')
            if len(parts) >= 3 and parts[2].isdigit():
                layer_idx = int(parts[2])
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(key)
    
    print(f"\nFound {len(layer_groups)} layer groups")
    print("\nLayer structure:")
    print("-" * 80)
    
    for idx in sorted(layer_groups.keys()):
        keys = layer_groups[idx]
        print(f"\nLayer {idx}: ({len(keys)} parameters)")
        
        # Show unique parameter types
        param_types = set()
        for key in keys:
            # Get the parameter type (e.g., 'conv.weight', 'bn.bias')
            param_type = '.'.join(key.split('.')[3:])
            param_types.add(param_type)
        
        for ptype in sorted(param_types):
            # Find a key with this type and show its shape
            for key in keys:
                if key.endswith(ptype):
                    tensor = state_dict[key]
                    print(f"  {ptype:40s} {tuple(tensor.shape)}")
                    break
    
    # Analyze detection head structure
    print("\n" + "="*80)
    print("DETECTION HEAD ANALYSIS")
    print("="*80)
    
    detect_keys = [k for k in state_dict.keys() if 'detect' in k.lower() or 'head' in k.lower()]
    if detect_keys:
        print(f"\nFound {len(detect_keys)} detection-related parameters:")
        for key in detect_keys[:20]:
            tensor = state_dict[key]
            print(f"  {key:60s} {tuple(tensor.shape)}")
    else:
        print("\nNo obvious detection head keys found")
        print("Detection head might be in the numbered layers")
    
    # Count total trainable parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal parameter count: {total_params:,}")
    
    return state_dict, layer_groups


def main():
    weights_path = Path("weights/MDV6-mit-yolov9-c.ckpt")
    
    if not weights_path.exists():
        print(f"Weights not found. Run test_against_reference.py first to download.")
        return
    
    state_dict, layer_groups = analyze_weights(weights_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nThe official model uses numbered layers (0, 1, 2, ...)")
    print(f"Our model uses named layers (conv0, elan2, rep_elan4, ...)")
    print(f"\nTo load weights, we need to create a mapping between:")
    print(f"  Official: model.model.0.conv.weight")
    print(f"  Ours:     conv0.conv.weight")


if __name__ == "__main__":
    main()
