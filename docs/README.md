# CV Learning — Documentation Index

Interview preparation resources for Computer Vision & ML Engineering roles.

## Documents

| File | What it covers |
|---|---|
| `01_transfer_learning.md` | All 5 TL types, when to use, common questions |
| `02_huggingface_ecosystem.md` | Trainer, datasets, PEFT, evaluate — full HF stack |
| `03_object_detection.md` | OD paradigms, DETR family, metrics, losses |
| `04_interview_prep.md` | 50+ interview Q&A for ML/CV engineering roles |
| `05_datasets_and_benchmarks.md` | Key datasets, leaderboards, standard benchmarks |

## Must-Know Topics for a Good ML/CV Developer

### Computer Vision Fundamentals
- Image classification, object detection, segmentation (semantic/instance/panoptic)
- Common backbones: ResNet, EfficientNet, ViT, Swin Transformer
- Data augmentation: random crop, flip, color jitter, mixup, cutmix
- Standard metrics: accuracy, mAP@50, mAP@50:95, IoU, F1

### Transfer Learning (this repo's focus)
- Feature extraction vs fine-tuning vs LoRA
- When to freeze vs unfreeze layers
- Learning rate schedules (warmup, cosine, linear decay)
- Knowledge distillation for model compression

### HuggingFace Ecosystem
- `Trainer` + `TrainingArguments` (main training loop)
- `AutoModel`, `AutoProcessor`, `AutoImageProcessor`
- `datasets` library (`set_transform`, `map`, `filter`)
- `evaluate` for metrics
- `peft` for LoRA and other adapters
- `transformers.pipelines` for inference

### PyTorch Core
- `Dataset` + `DataLoader` + `collate_fn`
- `torchvision.transforms.v2` with `tv_tensors`
- Custom loss functions
- Mixed precision (`torch.autocast`, `GradScaler`)
- `torch.compile` for speed

### Production & Deployment
- ONNX export
- TorchScript / `torch.jit.trace`
- Quantization (INT8, FP16)
- Batch inference optimization

### Object Detection Specific
- Anchor-based (YOLO, Faster-RCNN) vs anchor-free (FCOS, RT-DETR)
- DETR and its variants (Deformable DETR, RT-DETR, DINO)
- Loss functions: Focal Loss, GIoU, Hungarian matching
- NMS and its alternatives
