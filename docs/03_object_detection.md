# Object Detection — Complete Guide

## Detection Paradigms

### Two-Stage Detectors
**Examples:** Faster R-CNN, Mask R-CNN

```
Image → Backbone → RPN (region proposals) → RoI Pooling → Classifier + Regressor
```

- Slower, more accurate (historically)
- Region Proposal Network generates candidate boxes first
- Good for small object detection

### One-Stage Detectors
**Examples:** YOLO (v1-v11), SSD, RetinaNet, FCOS

```
Image → Backbone → FPN (Feature Pyramid) → Direct predictions per cell/point
```

- Faster, slightly less accurate (closing gap)
- Each grid cell predicts boxes directly
- YOLO family: most popular for real-time use

### Transformer-based (DETR family)
**Examples:** DETR, Deformable DETR, RT-DETR, DINO, DINO-DETR

```
Image → Backbone (ResNet/ViT) → Transformer Encoder-Decoder → N object queries → Boxes + Classes
```

- No NMS needed (set prediction)
- Hungarian matching loss during training
- RT-DETR: real-time DETR, state-of-the-art speed-accuracy tradeoff

---

## DETR Family — Interview Must Know

| Model | Year | Key Innovation | Speed |
|---|---|---|---|
| DETR | 2020 | First end-to-end transformer detector, no NMS | Slow |
| Deformable DETR | 2021 | Deformable attention, 10x faster convergence | Medium |
| DINO (Detection) | 2022 | Improved matching, contrastive denoising | Fast |
| RT-DETR | 2023 | Real-time speed + transformer accuracy | Real-time |
| RT-DETRv2 | 2024 | Improved architecture, bag of tricks | Real-time |

**Why DETR needs Hungarian matching:**
- Model predicts N fixed queries (e.g., 300). Ground truth has M boxes (M << N).
- Hungarian algorithm finds the optimal 1-to-1 matching between predictions and GT.
- Unmatched predictions → no-object class.

---

## Loss Functions

### Classification
- **CrossEntropy**: standard
- **Focal Loss**: for class imbalance (background >> objects). γ=2 is typical.
  ```
  FL = -(1 - pt)^γ × log(pt)
  ```

### Bounding Box Regression
| Loss | Formula | Property |
|---|---|---|
| L1 | |pred - gt| | Scale-dependent |
| L2 / MSE | (pred - gt)² | Sensitive to outliers |
| **IoU Loss** | 1 - IoU | Scale-invariant |
| **GIoU** | IoU - gap_penalty | Works when boxes don't overlap |
| **CIoU** | GIoU - aspect_ratio_penalty | Best for training stability |

**Use CIoU or GIoU in practice.**

---

## Key Metrics

### mAP (mean Average Precision)

```
Precision = TP / (TP + FP)     # of all predicted boxes, how many are correct?
Recall    = TP / (TP + FN)     # of all GT boxes, how many did we find?
AP        = area under PR curve for one class
mAP       = mean of AP across all classes
```

| Metric | Meaning |
|---|---|
| `mAP@50` | IoU threshold = 0.50 (PASCAL VOC standard) |
| `mAP@50:95` | Average mAP from IoU 0.50 to 0.95 in 0.05 steps (COCO standard) |
| `mAP@75` | Strict IoU threshold |
| `AR@100` | Average Recall with max 100 detections per image |

### IoU (Intersection over Union)

```
IoU = Area(Intersection) / Area(Union)
IoU > 0.5 → True Positive (PASCAL standard)
IoU > 0.75 → Strict True Positive
```

---

## Data Formats

| Format | Bbox Convention | Used By |
|---|---|---|
| COCO | `[x, y, w, h]` (top-left corner) | HF, torchvision CocoDetection |
| PASCAL VOC | `[x_min, y_min, x_max, y_max]` | torchvision |
| YOLO | `[cx, cy, w, h]` normalized 0-1 | Ultralytics |

**Convert COCO → XYXY (torchvision model input):**
```python
from torchvision.ops import box_convert
boxes_xyxy = box_convert(boxes_xywh, in_fmt="xywh", out_fmt="xyxy")
```

---

## NMS (Non-Maximum Suppression)

Used by one/two-stage detectors to remove duplicate boxes for the same object.
```
1. Sort boxes by confidence score (high → low)
2. Keep highest scoring box
3. Remove all boxes with IoU > threshold (0.5) with the kept box
4. Repeat
```

DETR-family models do NOT need NMS — each query predicts exactly one object.

---

## Backbone Comparison

| Backbone | Params | ImageNet Top-1 | Good For |
|---|---|---|---|
| ResNet-50 | 25M | 76.1% | Baseline, well-understood |
| EfficientNet-B4 | 19M | 82.9% | Efficiency-accuracy balance |
| Swin-Tiny | 28M | 81.3% | DETR-based detectors |
| ViT-Base | 86M | 81.8% | Large-scale pretraining |
| ConvNeXt-T | 29M | 82.1% | CNN vs ViT benchmark |

---

## Resources

- COCO dataset: https://cocodataset.org
- DETR paper: https://arxiv.org/abs/2005.12872
- RT-DETR paper: https://arxiv.org/abs/2304.08069
- Focal Loss paper: https://arxiv.org/abs/1708.02002
- GIoU paper: https://arxiv.org/abs/1902.09630
- YOLO history overview: https://docs.ultralytics.com/models/
- Detection metrics explained: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
- Roboflow OD tutorial: https://blog.roboflow.com/object-detection/
