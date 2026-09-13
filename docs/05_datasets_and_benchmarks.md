# Datasets & Benchmarks

## Image Classification

| Dataset | Classes | Images | Standard Use |
|---|---|---|---|
| **ImageNet-1K** | 1000 | 1.28M train | Pretraining backbone, benchmark |
| **ImageNet-21K** | 21841 | 14M | Large-scale pretraining |
| CIFAR-10 | 10 | 60K | Quick experiments, teaching |
| CIFAR-100 | 100 | 60K | Harder version of CIFAR-10 |
| Oxford-IIIT Pets | 37 | 7.4K | Fine-grained classification |
| Stanford Cars | 196 | 16K | Fine-grained classification |
| iNaturalist | 8142 | 2.7M | Long-tail distribution |

**Links:**
- ImageNet: https://image-net.org/
- HuggingFace imagenet-1k: https://huggingface.co/datasets/imagenet-1k
- CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html

---

## Object Detection

| Dataset | Classes | Images | Standard Metric |
|---|---|---|---|
| **COCO** | 80 | 118K train | mAP@50:95 |
| PASCAL VOC 2012 | 20 | 11K | mAP@50 |
| Open Images v7 | 600 | 9M | mAP |
| LVIS | 1203 | 100K+ | mAP (long-tail) |
| Objects365 | 365 | 630K | Pretraining |
| VisDrone | 10 | 10.2K | Small object detection (drones) |

**Links:**
- COCO: https://cocodataset.org
- PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- Open Images: https://storage.googleapis.com/openimages/web/index.html
- LVIS: https://www.lvisdataset.org/

---

## Segmentation

| Dataset | Task | Images | Classes |
|---|---|---|---|
| **ADE20K** | Semantic | 20K | 150 |
| Cityscapes | Semantic (driving) | 5K | 19 |
| COCO-Stuff | Panoptic | 118K | 171 |
| SA-1B (SAM) | Instance masks | 11M | Open |
| BDD100K | Driving scenes | 100K | 40 |

---

## Face & Biometrics

| Dataset | Task | Size |
|---|---|---|
| LFW | Face verification | 13K |
| FER-2013 | Facial expressions | 35K |
| AffectNet | Facial expressions | 420K |
| WFLW | Facial landmarks | 10K |
| WiderFace | Face detection | 32K images |

---

## Key Leaderboards (Papers With Code)

| Task | Leaderboard URL |
|---|---|
| ImageNet Classification | https://paperswithcode.com/sota/image-classification-on-imagenet |
| COCO Object Detection | https://paperswithcode.com/sota/object-detection-on-coco |
| ADE20K Segmentation | https://paperswithcode.com/sota/semantic-segmentation-on-ade20k |
| CIFAR-10 | https://paperswithcode.com/sota/image-classification-on-cifar-10 |

---

## Dataset Sources & Tools

| Tool | Purpose | URL |
|---|---|---|
| **Roboflow** | Dataset management, augmentation, annotation | https://roboflow.com |
| **HuggingFace Datasets** | 50K+ datasets, one-line loading | https://huggingface.co/datasets |
| **Kaggle** | Competition datasets | https://kaggle.com/datasets |
| **OpenDataLab** | Chinese CV datasets | https://opendatalab.com |
| **CVAT** | Free annotation tool | https://cvat.ai |
| **Label Studio** | Open source annotation | https://labelstud.io |

---

## Data Format Cheatsheet

```python
# COCO format (JSON)
{
  "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
  "annotations": [{"image_id": 1, "bbox": [x, y, w, h], "category_id": 1}],
  "categories": [{"id": 1, "name": "dog"}]
}

# YOLO format (.txt per image)
# class_id cx cy w h  (all normalized 0-1)
0 0.5 0.4 0.3 0.2

# Pascal VOC format (.xml)
<object>
  <name>dog</name>
  <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>200</ymax></bndbox>
</object>
```

**Convert between formats:**
- Roboflow: supports all format conversions in UI
- `supervision` library: https://supervision.roboflow.com/
