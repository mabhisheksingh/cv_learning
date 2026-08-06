# python3 detector.py --enable sahi --image P0007.jpg --model yolo26x-obb.pt
# python3 detector.py --enable ultralytics --image P0007.jpg --model yolo26x-obb.pt

import argparse
import os
import time

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def load_ground_truth(gt_file_path, orig_w, orig_h):
    """Load ground truth annotations from DOTA normalized format file."""
    gt_annotations = []
    if not os.path.exists(gt_file_path):
        return gt_annotations
    with open(gt_file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:9]))
                x_coords = [coords[i] * orig_w for i in range(0, 8, 2)]
                y_coords = [coords[i] * orig_h for i in range(1, 8, 2)]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                w = x_max - x_min
                h = y_max - y_min
                x_center = x_min + w / 2.0
                y_center = y_min + h / 2.0
                gt_annotations.append({
                    "line_num": line_num,
                    "class_id": class_id,
                    "x": x_center,
                    "y": y_center,
                    "w": w,
                    "h": h,
                })
    return gt_annotations


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes in xywh format."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_ground_truth(bbox, gt_annotations):
    """Return (matched_line, matched_gt_class, max_iou) for a detection bbox."""
    if not gt_annotations:
        return None, -1, 0.0
    max_iou = 0.0
    matched_line = None
    matched_gt_class = -1
    for gt in gt_annotations:
        iou = calculate_iou(bbox[:4], (gt["x"], gt["y"], gt["w"], gt["h"]))
        if iou > max_iou:
            max_iou = iou
            if iou > 0.3:
                matched_line = gt["line_num"]
                matched_gt_class = gt["class_id"]
    return matched_line, matched_gt_class, max_iou


def save_annotated_image(image_path, detections, output_dir, base_name):
    """Draw detections on image and save to output folder."""
    im_bgr = cv2.imread(image_path)
    for det in detections:
        bbox = det["bbox"]
        x_c, y_c, w, h, r = bbox
        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(im_bgr, f"{det['class_name']} {det['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_detector.jpg")
    cv2.imwrite(out_path, im_bgr)
    print(f"Saved annotated image: {out_path}")


# Default configuration constants
DEFAULT_CONF = 0.25
DEFAULT_DEVICE = "cpu"
DEFAULT_IMGSZ = 640
DEFAULT_SLICE_HEIGHT = 640
DEFAULT_SLICE_WIDTH = 640
DEFAULT_OVERLAP_HEIGHT_RATIO = 0.2
DEFAULT_OVERLAP_WIDTH_RATIO = 0.2
DEFAULT_IOU = 0.6
DEFAULT_MAX_DET = 300


class ObjectDetector:
    """Unified detector supporting Ultralytics and SAHI backends."""

    def __init__(self, model_path, conf=DEFAULT_CONF, device=DEFAULT_DEVICE, enable_sahi=False):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self.enable_sahi = enable_sahi
        self.model = YOLO(model_path, task="obb")
        if enable_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=conf,
                device=device,
            )

    def detect(self, image_path, imgsz=DEFAULT_IMGSZ, slice_h=DEFAULT_SLICE_HEIGHT, slice_w=DEFAULT_SLICE_WIDTH,
               overlap_h=DEFAULT_OVERLAP_HEIGHT_RATIO, overlap_w=DEFAULT_OVERLAP_WIDTH_RATIO):
        """Run detection. Returns list of detections in xywhr format."""
        if self.enable_sahi:
            return self._detect_sahi(
                image_path, slice_h, slice_w, overlap_h, overlap_w
            )
        return self._detect_ultralytics(image_path, imgsz)

    def _detect_ultralytics(self, image_path, imgsz):
        im_bgr = cv2.imread(image_path)
        orig_h, orig_w = im_bgr.shape[:2]
        scale_x = orig_w / imgsz
        scale_y = orig_h / imgsz

        im_resized = cv2.resize(im_bgr, (imgsz, imgsz), cv2.INTER_LINEAR)

        start = time.time()
        results = self.model.predict(
            source=[im_resized],
            imgsz=imgsz,
            conf=self.conf,
            iou=DEFAULT_IOU,
            max_det=DEFAULT_MAX_DET,
            device=self.device,
            verbose=False,
        )
        elapsed = time.time() - start

        detections = []
        r = results[0]
        if getattr(r, "obb", None) is not None and getattr(r.obb, "xywhr", None) is not None:
            obb = r.obb
            for i in range(int(obb.xywhr.shape[0])):
                bbox = obb.xywhr[i].cpu().numpy().copy()
                bbox[0] *= scale_x
                bbox[1] *= scale_y
                bbox[2] *= scale_x
                bbox[3] *= scale_y
                cls_id = int(obb.cls[i].cpu().numpy()) if obb.cls is not None else -1
                conf = float(obb.conf[i].cpu().numpy()) if obb.conf is not None else 0.0
                cls_name = self.model.names.get(cls_id, f"Unknown({cls_id})")
                detections.append({
                    "bbox": bbox,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf,
                })
        return detections, elapsed

    def _detect_sahi(self, image_path, slice_h, slice_w, overlap_h, overlap_w):
        start = time.time()
        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.sahi_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap_h,
            overlap_width_ratio=overlap_w,
        )
        elapsed = time.time() - start

        detections = []
        for pred in result.object_prediction_list:
            x, y, w, h = pred.bbox.to_xywh()
            x_c = x + w / 2
            y_c = y + h / 2
            cls_id = int(pred.category.id)
            cls_name = pred.category.name
            conf = float(pred.score.value)
            detections.append({
                "bbox": np.array([x_c, y_c, w, h, 0.0]),
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": conf,
            })
        return detections, elapsed


def _resolve_paths(base_dir, model_arg, image_arg):
    model_path = model_arg if os.path.isabs(model_arg) else os.path.join(base_dir, "model", model_arg)
    image_path = image_arg if os.path.isabs(image_arg) else os.path.join(base_dir, "data", "input", image_arg)
    return model_path, image_path


def _load_and_print_ground_truth(base_dir, image_path, orig_h, orig_w):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gt_file_path = os.path.join(base_dir, "data", "label", "val", f"{base_name}.txt")
    gt_annotations = load_ground_truth(gt_file_path, orig_w, orig_h)
    if gt_annotations:
        print(f"\nLoaded {len(gt_annotations)} ground truth annotations from: {os.path.basename(gt_file_path)}")
    else:
        print("\nNo ground truth annotations found for this image")
    return gt_annotations


def _print_detection(i, det, gt_annotations):
    bbox = det["bbox"]
    print(f"\nDetection {i+1}:")
    print(f"  Class ID: {det['class_id']}")
    print(f"  Class Name: {det['class_name']}")
    print(f"  Confidence: {det['conf']:.4f}")
    print(f"  BBox (original xywhr): x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}, r={bbox[4]:.4f}")
    matched_line, matched_gt_class, max_iou = match_ground_truth(bbox, gt_annotations)
    if matched_line:
        print(f"  \u2713 MATCHED with ground truth line {matched_line} (GT Class: {matched_gt_class}, IoU: {max_iou:.4f})")
        return True
    print(f"  \u2717 No match in ground truth (max IoU: {max_iou:.4f})")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable", choices=["sahi", "ultralytics"], default="ultralytics")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path, image_path = _resolve_paths(base_dir, args.model, args.image)

    detector = ObjectDetector(
        model_path=model_path,
        enable_sahi=(args.enable == "sahi"),
    )

    im_bgr = cv2.imread(image_path)
    orig_h, orig_w = im_bgr.shape[:2]
    print(f"Original image size: {orig_w}x{orig_h}")

    gt_annotations = _load_and_print_ground_truth(base_dir, image_path, orig_h, orig_w)

    dets, t = detector.detect(image_path)
    print(f"\nInference time: {t:.3f}s")
    print(f"\nTotal Detections: {len(dets)}")

    matched_count = sum(1 for det in dets if match_ground_truth(det["bbox"], gt_annotations)[0] is not None)
    unmatched_count = len(dets) - matched_count
    print(f"Matched: {matched_count}/{len(dets)}")
    print(f"Unmatched: {unmatched_count}/{len(dets)}")
    print(f"{'-'*60}")

    for i, det in enumerate(dets):
        _print_detection(i, det, gt_annotations)

    output_dir = os.path.join(base_dir, "data", "output")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_sahi" if args.enable == "sahi" else "_ultralytics"
    save_annotated_image(image_path, dets, output_dir, f"{base_name}{suffix}")


if __name__ == "__main__":
    main()
