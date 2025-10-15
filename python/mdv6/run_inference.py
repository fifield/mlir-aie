"""Simple script to run both models on a single image file."""

import sys
from pathlib import Path

# Add paths FIRST, before other imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "camera_traps_reference"))

import torch
from PIL import Image
import numpy as np

from model import MDV6MITYOLOv9c
from weight_mapper import load_and_transform_weights
from visualize import decode_predictions, draw_boxes


def load_image(image_path, size=640):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    img = img.resize((size, size))
    
    # Convert to tensor
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def run_our_model(image_path, weights_path, conf_threshold=0.2, iou_threshold=0.5, output_dir=None):
    """Run our MDV6 implementation."""
    print("="*80)
    print("RUNNING OUR MODEL")
    print("="*80)
    
    # Load model
    model = MDV6MITYOLOv9c(num_classes=3)
    model.eval()
    
    # Load weights
    if weights_path.exists():
        transformed_dict = load_and_transform_weights(weights_path)
        model.load_state_dict(transformed_dict, strict=False)
        print("✓ Loaded pretrained weights")
    else:
        print("⚠ Using random initialization (no weights found)")
    
    # Load image
    img_tensor = load_image(image_path)
    print(f"✓ Loaded image: {img_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"\nRaw Outputs:")
    for i, (class_x, anchor_x, vector_x) in enumerate(outputs):
        print(f"  Scale {i} (P{i+3}): class {class_x.shape}, anchor {anchor_x.shape}, vector {vector_x.shape}")
    
    # Decode to bounding boxes
    boxes, scores, classes = decode_predictions(outputs, conf_threshold, iou_threshold)
    print(f"\nDetections after NMS: {len(boxes)}")
    
    for i in range(len(boxes)):
        class_name = ['animal', 'person', 'vehicle'][classes[i]]
        print(f"  {i+1}. {class_name}: {scores[i]:.3f}")
    
    # Visualize if output_dir specified
    if output_dir and len(boxes) > 0:
        output_path = Path(output_dir) / f"our_model_{Path(image_path).name}"
        draw_boxes(image_path, boxes, scores, classes, output_path, title="Our Model")
    
    return outputs, (boxes, scores, classes)


def run_official_model(image_path, weights_path, conf_threshold=0.2, iou_threshold=0.5, output_dir=None):
    """Run official CameraTraps model."""
    print("\n" + "="*80)
    print("RUNNING OFFICIAL MODEL")
    print("="*80)
    
    try:
        from PytorchWildlife.models.detection.yolo_mit import MegaDetectorV6MIT
        
        # Load model
        model = MegaDetectorV6MIT(
            weights=str(weights_path),
            device='cpu',
            pretrained=True,
            version='MDV6-mit-yolov9-c'
        )
        print("✓ Loaded official model")
        
        # Trigger model loading
        if not hasattr(model, 'model'):
            model._load_model(weights=model.weights, device=model.device, url=model.url)
        
        # Load image
        img_tensor = load_image(image_path)
        print(f"✓ Loaded image: {img_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model.model(img_tensor)
        
        if isinstance(outputs, dict) and 'Main' in outputs:
            main_outputs = outputs['Main']
            print(f"\nRaw Outputs:")
            for i, (class_x, anchor_x, vector_x) in enumerate(main_outputs):
                print(f"  Scale {i} (P{i+3}): class {class_x.shape}, anchor {anchor_x.shape}, vector {vector_x.shape}")
            
            # Decode to bounding boxes
            boxes, scores_out, classes_out = decode_predictions(main_outputs, conf_threshold, iou_threshold)
            print(f"\nDetections after NMS: {len(boxes)}")
            
            for i in range(len(boxes)):
                class_name = ['animal', 'person', 'vehicle'][classes_out[i]]
                print(f"  {i+1}. {class_name}: {scores_out[i]:.3f}")
            
            # Visualize if output_dir specified
            if output_dir and len(boxes) > 0:
                output_path = Path(output_dir) / f"official_model_{Path(image_path).name}"
                draw_boxes(image_path, boxes, scores_out, classes_out, output_path, title="Official Model")
            
            return main_outputs, (boxes, scores_out, classes_out)
        else:
            print(f"  Unexpected output format: {type(outputs)}")
            return None, None
            
    except Exception as e:
        print(f"✗ Error running official model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MDV6 models on an image')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--weights', type=str, default='mdv6/weights/MDV6-mit-yolov9-c.ckpt',
                       help='Path to weights file')
    parser.add_argument('--official-only', action='store_true',
                       help='Run only official model')
    parser.add_argument('--ours-only', action='store_true',
                       help='Run only our model')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualization images')
    parser.add_argument('--conf-threshold', type=float, default=0.2,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IOU threshold for NMS')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    weights_path = Path(args.weights)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print("="*80)
    print(f"MDV6-mit-yolov9-c INFERENCE")
    print(f"Image: {image_path}")
    print(f"Weights: {weights_path}")
    print("="*80)
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
    else:
        output_dir = None
    
    # Run models
    our_results = None
    official_results = None
    
    if not args.official_only:
        our_results = run_our_model(image_path, weights_path, 
                                    args.conf_threshold, args.iou_threshold, output_dir)
    
    if not args.ours_only:
        official_results = run_official_model(image_path, weights_path,
                                             args.conf_threshold, args.iou_threshold, output_dir)
    
    # Compare if both ran
    if our_results is not None and official_results is not None:
        our_outputs, our_dets = our_results
        official_outputs, official_dets = official_results
        
        if our_outputs is not None and official_outputs is not None:
            print("\n" + "="*80)
            print("RAW OUTPUT COMPARISON")
            print("="*80)
            
            for i in range(len(our_outputs)):
                our_class, our_anchor, our_vector = our_outputs[i]
                off_class, off_anchor, off_vector = official_outputs[i]
                
                class_diff = torch.abs(our_class - off_class).max().item()
                anchor_diff = torch.abs(our_anchor - off_anchor).max().item()
                vector_diff = torch.abs(our_vector - off_vector).max().item()
                
                print(f"\nScale {i}:")
                print(f"  Class diff: {class_diff:.6e}")
                print(f"  Anchor diff: {anchor_diff:.6e}")
                print(f"  Vector diff: {vector_diff:.6e}")
        
        if our_dets is not None and official_dets is not None:
            our_boxes, our_scores, our_classes = our_dets
            off_boxes, off_scores, off_classes = official_dets
            
            print("\n" + "="*80)
            print("DETECTION COMPARISON")
            print("="*80)
            print(f"Our model: {len(our_boxes)} detections")
            print(f"Official model: {len(off_boxes)} detections")


if __name__ == "__main__":
    main()
