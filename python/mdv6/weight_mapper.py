"""Weight key mapping for loading official MDV6-mit-yolov9-c weights into our model."""

import torch
from pathlib import Path


def create_layer_mapping():
    """Create mapping from official numbered layers to our named layers."""
    # Based on config_v9s.yaml layer sequence
    mapping = {
        # Backbone
        '0': 'conv0',           # Conv(32)
        '1': 'conv1',           # Conv(64)
        '2': 'elan2',           # ELAN(64)
        '3': 'aconv3',          # AConv(128) - B3
        '4': 'rep_elan4',       # RepNCSPELAN(128)
        '5': 'aconv5',          # AConv(192) - B4
        '6': 'rep_elan6',       # RepNCSPELAN(192)
        '7': 'aconv7',          # AConv(256) - B5
        '8': 'rep_elan8',       # RepNCSPELAN(256)
        
        # Neck
        '9': 'spp9',            # SPPELAN(256) - N3
        # 10: UpSample (no params)
        # 11: Concat (no params)
        '12': 'rep_elan12',     # RepNCSPELAN(192) - N4
        
        # Head - Main Path
        # 13: UpSample (no params)
        # 14: Concat (no params)
        '15': 'rep_elan15',     # RepNCSPELAN(128) - P3
        '16': 'aconv16',        # AConv(96)
        # 17: Concat (no params)
        '18': 'rep_elan18',     # RepNCSPELAN(192) - P4
        '19': 'aconv19',        # AConv(128)
        # 20: Concat (no params)
        '21': 'rep_elan21',     # RepNCSPELAN(256) - P5
        '22': 'detect',         # MultiheadDetection
        
        # Layers 23-29 are auxiliary path (not in our model)
    }
    return mapping


def transform_weight_key(official_key, layer_mapping):
    """
    Transform official weight key to our model's key format.
    
    Args:
        official_key: e.g., "0.conv.weight" or "4.conv2.0.bottleneck.0.conv1.conv1.conv.weight"
        layer_mapping: Dict mapping layer numbers to names
    
    Returns:
        Transformed key: e.g., "conv0.conv.weight" or "rep_elan4.conv2.0.bottleneck.0.conv1.conv1.conv.weight"
    """
    parts = official_key.split('.')
    
    if len(parts) < 2:
        return None
    
    # Get layer number (first part)
    layer_num = parts[0]
    
    if layer_num not in layer_mapping:
        # Skip auxiliary layers or unknown layers
        return None
    
    # Replace layer number with layer name
    layer_name = layer_mapping[layer_num]
    
    # Reconstruct key
    new_key = layer_name + '.' + '.'.join(parts[1:])
    
    return new_key


def load_and_transform_weights(checkpoint_path, layer_mapping=None):
    """
    Load official checkpoint and transform keys for our model.
    
    Args:
        checkpoint_path: Path to MDV6-mit-yolov9-c.ckpt
        layer_mapping: Optional custom mapping (uses default if None)
    
    Returns:
        Transformed state dict ready for our model
    """
    if layer_mapping is None:
        layer_mapping = create_layer_mapping()
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Original state dict: {len(state_dict)} parameters")
    
    # Remove 'model.model.' prefix
    if any(k.startswith('model.model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.model.', ''): v for k, v in state_dict.items()}
        print("Removed 'model.model.' prefix")
    
    # Transform keys
    transformed_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        new_key = transform_weight_key(key, layer_mapping)
        
        if new_key is not None:
            transformed_dict[new_key] = value
        else:
            skipped_keys.append(key)
    
    print(f"\nTransformed state dict: {len(transformed_dict)} parameters")
    print(f"Skipped (auxiliary/unknown): {len(skipped_keys)} parameters")
    
    if skipped_keys and len(skipped_keys) <= 10:
        print("\nFirst few skipped keys:")
        for key in skipped_keys[:10]:
            print(f"  {key}")
    
    return transformed_dict


def test_weight_loading():
    """Test the weight transformation and loading."""
    from model import MDV6MITYOLOv9c
    
    weights_path = Path("weights/MDV6-mit-yolov9-c.ckpt")
    
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Run test_against_reference.py first to download weights.")
        return
    
    print("="*80)
    print("TESTING WEIGHT TRANSFORMATION AND LOADING")
    print("="*80)
    
    # Create model
    model = MDV6MITYOLOv9c(num_classes=3)
    print(f"\nOur model parameters: {model.count_parameters()['total']:,}")
    
    # Load and transform weights
    transformed_dict = load_and_transform_weights(weights_path)
    
    # Try to load
    print("\nAttempting to load transformed weights...")
    missing, unexpected = model.load_state_dict(transformed_dict, strict=False)
    
    print(f"\nResults:")
    print(f"  Missing keys: {len(missing)}")
    if missing and len(missing) <= 20:
        print("  First 20 missing keys:")
        for key in missing[:20]:
            print(f"    - {key}")
    
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected and len(unexpected) <= 20:
        print("  First 20 unexpected keys:")
        for key in unexpected[:20]:
            print(f"    - {key}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("\n✓ ALL WEIGHTS LOADED SUCCESSFULLY!")
    elif len(missing) < 100:
        print(f"\n⚠ Mostly loaded ({len(transformed_dict) - len(missing)}/{len(transformed_dict)} matched)")
    else:
        print(f"\n✗ Significant mismatch")
    
    return model, transformed_dict


if __name__ == "__main__":
    model, transformed_dict = test_weight_loading()
