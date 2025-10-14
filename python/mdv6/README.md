## MDV6-mit-yolov9-c Implementation

PyTorch implementation of Microsoft's MegaDetectorV6 MIT YOLOv9-c for wildlife detection.

### Overview

This is a clean-room implementation of the MDV6-mit-yolov9-c model based on Microsoft's CameraTraps repository. The model is designed for detecting animals, persons, and vehicles in camera trap images.

**Key Features:**
- ✅ Only 3 classes (animal, person, vehicle)
- ✅ Main detection path only (no auxiliary path for simpler inference)
- ✅ Based on YOLOv9-c architecture
- ✅ ~7.2M parameters
- ✅ Fully functional and tested

### Architecture

**Backbone** (GELAN):
- Conv(32) → Conv(64) → ELAN(64)
- AConv(128) → RepNCSPELAN(128) [B3]
- AConv(192) → RepNCSPELAN(192) [B4]
- AConv(256) → RepNCSPELAN(256) [B5]

**Neck**:
- SPPELAN(256) [N3]
- UpSample + Concat(B4) → RepNCSPELAN(192) [N4]

**Head** (Main Path):
- UpSample + Concat(B3) → RepNCSPELAN(128) [P3]
- AConv + Concat(N4) → RepNCSPELAN(192) [P4]
- AConv + Concat(N3) → RepNCSPELAN(256) [P5]
- MultiheadDetection([P3, P4, P5])

**Channel Progression**: 32 → 64 → 128 → 192 → 256

### Files

```
python/mdv6/
├── __init__.py          # Package initialization
├── layers.py            # Layer implementations
├── model.py             # MDV6MITYOLOv9c model
├── test_mdv6.py         # Test script
└── README.md            # This file
```

### Usage

```python
from mdv6 import MDV6MITYOLOv9c

# Create model
model = MDV6MITYOLOv9c(num_classes=3)
model.eval()

# Print model info
model.print_model_info()

# Run inference
import torch
img = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    detections = model(img)

# detections is a list of 3 tuples (one per scale):
# [(class_p3, anchor_p3, vector_p3),
#  (class_p4, anchor_p4, vector_p4),
#  (class_p5, anchor_p5, vector_p5)]

for i, (class_x, anchor_x, vector_x) in enumerate(detections):
    print(f"Scale {i}:")
    print(f"  Classes: {class_x.shape}")  # (B, 3, H, W)
    print(f"  Anchors: {anchor_x.shape}")  # (B, 16, 4, H, W)
    print(f"  Vectors: {vector_x.shape}")  # (B, 4, H, W)
```

### Testing

```bash
cd python/mdv6
python test_mdv6.py
```

### Output Format

Each detection scale returns a tuple of:

1. **class_x**: `(B, 3, H, W)` - Class probabilities for 3 classes
2. **anchor_x**: `(B, 16, 4, H, W)` - Distribution for box coordinates (DFL)
3. **vector_x**: `(B, 4, H, W)` - Expected box coordinates

### Detection Scales

- **P3**: 1/8 resolution - Small objects (animals close to camera)
- **P4**: 1/16 resolution - Medium objects
- **P5**: 1/32 resolution - Large objects (distant animals)

### Model Statistics

```
Total Parameters: 7,196,441
- Backbone: ~4.5M
- Neck: ~1.5M
- Head: ~1.2M

Input: 640x640 RGB image
Output: 3 detection scales with class + box predictions
```

### Differences from Full Implementation

This implementation includes **only the main/forward detection path**:
- ✅ Simpler architecture
- ✅ Faster inference
- ✅ Suitable for deployment
- ❌ No auxiliary path (used for PGI training)

For the full dual-path implementation with auxiliary head, see the Microsoft CameraTraps repository.

### Official Resources

- **Repository**: https://github.com/microsoft/CameraTraps
- **Weights**: https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-c.ckpt
- **Config**: https://zenodo.org/records/15178680/files/config_v9s.yaml
- **Documentation**: https://microsoft.github.io/CameraTraps/
- **License**: MIT

### Requirements

- PyTorch >= 1.10
- NumPy

### Next Steps

To use with pretrained weights:
1. Download weights from Zenodo
2. Load using `create_mdv6_mit_yolov9c(pretrained=True, weights_path='path/to/weights.ckpt')`
3. Implement NMS post-processing for final detections
4. Add visualization utilities

### Citation

If you use this implementation, please cite the original MegaDetector and PyTorch-Wildlife papers:

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation}, 
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
