import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Local Ultralytics benchmark settings
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "yolo26x-obb.pt")

# List of image filenames to process (only these images will be read from input folder)
IMAGE_NAMES = [
    # "P0003.jpg",
    # "P0004.jpg",
    "P0007.jpg",
    # "P0019.jpg",
    # "P0027.jpg",
    # "P0047.jpg",
    # "P0053.jpg",
    # "P0056.jpg",
    # "P0060.jpg",
    # "P0075.jpg",
    # "P0079.jpg",
    # "P0081.jpg",
    # "P0086.jpg",
]

IMGSZ = 640  # Stick to 640x640 as this model is highly sensitive to scale
CONF = 0.25  # Match Triton confidence
DEVICE = "cpu"  # set to 'cpu' if GPU is busy / OOM


def load_ground_truth(gt_file_path, orig_w, orig_h):
    """Load ground truth annotations from DOTA normalized format file.
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    Returns list of dicts with line numbers and annotations.
    """
    gt_annotations = []
    if not os.path.exists(gt_file_path):
        print(f"Ground truth file not found: {gt_file_path}")
        return gt_annotations
    
    with open(gt_file_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:9]))
                
                # Convert normalized coords to absolute pixel coords
                x_coords = [coords[i] * orig_w for i in range(0, 8, 2)]
                y_coords = [coords[i] * orig_h for i in range(1, 8, 2)]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                w = x_max - x_min
                h = y_max - y_min
                x_center = x_min + w / 2.0
                y_center = y_min + h / 2.0
                
                gt_annotations.append({
                    'line_num': line_num,
                    'class_id': class_id,
                    'x_coords': x_coords,
                    'y_coords': y_coords,
                    'x': x_center,
                    'y': y_center,
                    'w': w,
                    'h': h,
                    'x_min': x_min,
                    'y_min': y_min
                })
    
    return gt_annotations


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes in xywh format."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Convert to xyxy
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Construct full image paths from input folder
    valid_images = []
    for img_name in IMAGE_NAMES:
        img_path = os.path.join(INPUT_DIR, img_name)
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not valid_images:
        raise FileNotFoundError("No valid image paths provided from input folder")

    model = YOLO(MODEL_PATH, task="obb")

    # Match Triton preprocessing: force a direct resize to 640x640 (no aspect-ratio letterbox)
    resized_sources = []
    original_shapes = []
    for p in valid_images:
        im_bgr = cv2.imread(p)
        if im_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        original_shapes.append(im_bgr.shape[:2])  # (height, width)
        # Ultralytics predict() expects BGR numpy arrays and converts to RGB internally
        im_resized = cv2.resize(im_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        resized_sources.append(im_resized)

    # Run inference with NMS settings
    inference_start = time.time()
    results = model.predict(
        source=resized_sources,
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.6,  # Higher IoU = less aggressive NMS (keeps more overlapping detections)
        max_det=300,
        device=DEVICE,
        verbose=True,
    )
    inference_time = time.time() - inference_start
    print(f"\nInference time: {inference_time:.3f}s")

    # Save annotated outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for img_idx, (img_path, r) in enumerate(zip(valid_images, results)):
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        # Get original image shape for coordinate scaling
        orig_h, orig_w = original_shapes[img_idx]
        scale_x = orig_w / IMGSZ
        scale_y = orig_h / IMGSZ
        print(f"Original image size: {orig_w}x{orig_h}")
        print(f"Resized image size: {IMGSZ}x{IMGSZ}")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
        # Load ground truth file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "label", "val", f"{base_name}.txt")
        gt_annotations = load_ground_truth(gt_file_path, orig_w, orig_h)
        
        if gt_annotations:
            print(f"\nLoaded {len(gt_annotations)} ground truth annotations from: {os.path.basename(gt_file_path)}")
            
            # Draw ground truth on a copy of the original image
            gt_img = cv2.imread(img_path)
            for gt in gt_annotations:
                pts = np.array(list(zip(gt['x_coords'], gt['y_coords'])), np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw polygon bounding box (Green)
                cv2.polylines(gt_img, [pts], True, (0, 255, 0), 2)
                # Draw class ID text
                x_min = int(gt['x_min'])
                y_min = int(gt['y_min'])
                cv2.putText(gt_img, f"cls:{gt['class_id']}", (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            gt_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_ground_truth.jpg")
            cv2.imwrite(gt_out_path, gt_img)
            print(f"Saved ground truth visualization: {gt_out_path}")
        else:
            print(f"\nNo ground truth annotations found for this image")
        
        # Print OBB detections
        if getattr(r, "obb", None) is not None and getattr(r.obb, "xywhr", None) is not None:
            obb = r.obb
            n_det = int(obb.xywhr.shape[0])
            print(f"\nTotal OBB Detections: {n_det}")
            print(f"{'-'*60}")
            
            for i in range(n_det):
                # Get bounding box in xywhr format (x_center, y_center, width, height, rotation)
                bbox = obb.xywhr[i].cpu().numpy()
                # Scale coordinates back to original image size
                bbox_scaled = bbox.copy()
                bbox_scaled[0] *= scale_x  # x_center
                bbox_scaled[1] *= scale_y  # y_center
                bbox_scaled[2] *= scale_x  # width
                bbox_scaled[3] *= scale_y  # height
                # rotation remains the same
                
                # Get class ID
                class_id = int(obb.cls[i].cpu().numpy()) if obb.cls is not None else -1
                # Get confidence score
                conf = float(obb.conf[i].cpu().numpy()) if obb.conf is not None else 0.0
                # Get class name
                class_name = model.names.get(class_id, f"Unknown({class_id})")
                
                print(f"\nDetection {i+1}:")
                print(f"  Class ID: {class_id}")
                print(f"  Class Name: {class_name}")
                print(f"  Confidence: {conf:.4f}")
                print(f"  BBox (resized xywhr): x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}, r={bbox[4]:.4f}")
                print(f"  BBox (original xywhr): x={bbox_scaled[0]:.2f}, y={bbox_scaled[1]:.2f}, w={bbox_scaled[2]:.2f}, h={bbox_scaled[3]:.2f}, r={bbox_scaled[4]:.4f}")
                
                # Check against ground truth
                if gt_annotations:
                    matched_line = None
                    max_iou = 0.0
                    matched_gt_class = -1
                    for gt in gt_annotations:
                        # We removed the strict class_id check because VisDrone classes (4=car, 5=van) 
                        # don't match DOTA classes (9=large vehicle, 10=small vehicle)
                        
                        # Calculate IoU with scaled coordinates
                        iou = calculate_iou(bbox_scaled[:4], (gt['x'], gt['y'], gt['w'], gt['h']))
                        if iou > max_iou:
                            max_iou = iou
                            if iou > 0.3:  # IoU threshold for matching
                                matched_line = gt['line_num']
                                matched_gt_class = gt['class_id']
                    
                    if matched_line:
                        print(f"  ✓ MATCHED with ground truth line {matched_line} (GT Class: {matched_gt_class}, IoU: {max_iou:.4f})")
                    else:
                        print(f"  ✗ No match in ground truth (max IoU: {max_iou:.4f})")
        
        # Print regular box detections (fallback)
        elif getattr(r, "boxes", None) is not None and getattr(r.boxes, "xyxy", None) is not None:
            boxes = r.boxes
            n_det = int(boxes.xyxy.shape[0])
            print(f"\nTotal Box Detections: {n_det}")
            print(f"{'-'*60}")
            
            for i in range(n_det):
                # Get bounding box in xyxy format (x1, y1, x2, y2)
                bbox = boxes.xyxy[i].cpu().numpy()
                # Convert to xywh for IoU calculation
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Get class ID
                class_id = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else -1
                # Get confidence score
                conf = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 0.0
                # Get class name
                class_name = model.names.get(class_id, f"Unknown({class_id})")
                
                print(f"\nDetection {i+1}:")
                print(f"  Class ID: {class_id}")
                print(f"  Class Name: {class_name}")
                print(f"  Confidence: {conf:.4f}")
                print(f"  BBox (xyxy): x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f}")
                
                # Check against ground truth
                if gt_annotations:
                    matched_line = None
                    max_iou = 0.0
                    matched_gt_class = -1
                    for gt in gt_annotations:
                        # Calculate IoU
                        iou = calculate_iou((x_center, y_center, width, height), (gt['x'], gt['y'], gt['w'], gt['h']))
                        if iou > max_iou:
                            max_iou = iou
                            if iou > 0.3:  # IoU threshold for matching
                                matched_line = gt['line_num']
                                matched_gt_class = gt['class_id']
                    
                    if matched_line:
                        print(f"  ✓ MATCHED with ground truth line {matched_line} (GT Class: {matched_gt_class}, IoU: {max_iou:.4f})")
                    else:
                        print(f"  ✗ No match in ground truth (max IoU: {max_iou:.4f})")
        else:
            print("\nNo detections found")

        annotated = r.plot()  # BGR uint8
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_ultralytics_obb.jpg")
        ok = cv2.imwrite(out_path, annotated)
        if not ok:
            raise RuntimeError(f"Failed to write output image to {out_path}")
        print(f"Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
