"""Visualization utilities for MDV6 detection outputs."""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Class names and colors
CLASS_NAMES = ['animal', 'person', 'vehicle']
CLASS_COLORS = [
    (255, 100, 100),  # Red for animal
    (100, 255, 100),  # Green for person
    (100, 100, 255),  # Blue for vehicle
]


def decode_predictions(outputs, conf_threshold=0.2, iou_threshold=0.5, img_size=640):
    """
    Decode model outputs to bounding boxes.
    
    Args:
        outputs: List of (class_x, anchor_x, vector_x) tuples for each scale
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        img_size: Input image size
    
    Returns:
        boxes: (N, 4) tensor of [x1, y1, x2, y2]
        scores: (N,) tensor of confidence scores
        classes: (N,) tensor of class indices
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    
    strides = [8, 16, 32]  # P3, P4, P5
    
    for scale_idx, (class_x, anchor_x, vector_x) in enumerate(outputs):
        stride = strides[scale_idx]
        B, C, H, W = class_x.shape
        
        # Get class scores and predictions
        class_scores = torch.sigmoid(class_x)  # (B, 3, H, W)
        max_scores, class_ids = class_scores.max(dim=1)  # (B, H, W)
        
        # Get box coordinates from vector_x
        # vector_x is (B, 4, H, W) - already the expected coordinates in pixels
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_y = grid_y.float()
        grid_x = grid_x.float()
        
        # Convert to absolute coordinates
        # vector_x contains distances, we need to convert to boxes
        cx = (grid_x + 0.5) * stride
        cy = (grid_y + 0.5) * stride
        
        # vector_x is (B, 4, H, W) where 4 = [left, top, right, bottom] distances
        left = vector_x[0, 0] * stride
        top = vector_x[0, 1] * stride
        right = vector_x[0, 2] * stride
        bottom = vector_x[0, 3] * stride
        
        x1 = cx - left
        y1 = cy - top
        x2 = cx + right
        y2 = cy + bottom
        
        # Filter by confidence
        mask = max_scores[0] > conf_threshold
        
        if mask.sum() > 0:
            boxes = torch.stack([
                x1[mask],
                y1[mask],
                x2[mask],
                y2[mask]
            ], dim=1)
            
            scores = max_scores[0][mask]
            classes = class_ids[0][mask]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
    
    if len(all_boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    
    # Concatenate all scales
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    classes = torch.cat(all_classes, dim=0)
    
    # Apply NMS per class
    keep_indices = []
    for class_id in range(3):
        class_mask = classes == class_id
        if class_mask.sum() > 0:
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # Simple NMS
            keep = nms(class_boxes, class_scores, iou_threshold)
            class_indices = torch.where(class_mask)[0]
            keep_indices.extend(class_indices[keep].tolist())
    
    if len(keep_indices) == 0:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    
    keep_indices = torch.tensor(keep_indices)
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]


def nms(boxes, scores, iou_threshold):
    """Simple NMS implementation."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        
        if order.numel() == 1:
            break
        
        # Compute IoU
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep, dtype=torch.int64)


def draw_boxes(image_path, boxes, scores, classes, output_path, title="Detections"):
    """
    Draw bounding boxes on image and save.
    
    Args:
        image_path: Path to original image
        boxes: (N, 4) tensor of [x1, y1, x2, y2]
        scores: (N,) tensor of confidence scores
        classes: (N,) tensor of class indices
        output_path: Where to save the visualization
        title: Title to add to image
    """
    # Load original image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Resize to 640x640 (same as model input)
    img = img.resize((640, 640))
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i].item()
        class_id = classes[i].item()
        
        x1, y1, x2, y2 = box.tolist()
        
        # Clamp to image bounds
        x1 = max(0, min(640, x1))
        y1 = max(0, min(640, y1))
        x2 = max(0, min(640, x2))
        y2 = max(0, min(640, y2))
        
        # Get color and label
        color = CLASS_COLORS[class_id]
        label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1-20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw label text
        draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)
    
    # Add title
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)  # Shadow
    
    # Save
    img.save(output_path)
    print(f"✓ Saved visualization to {output_path}")
    
    return img
