# Transfer Learning in Computer Vision

This folder contains **5 industry-standard transfer learning techniques** with working HuggingFace-based examples.

## The 5 Techniques

```
Pretrained Model (ImageNet / COCO / Web-scale)
            │
    ┌───────┼────────────────────────────────────┐
    │       │                                    │
    ▼       ▼                                    ▼
Feature  Fine-Tuning    LoRA/PEFT     Knowledge    Zero-Shot
Extract  (Full/Partial) (Adapters)   Distillation  (CLIP)
  │          │              │              │           │
Freeze    Unfreeze       Inject        Teacher→     No labels
backbone   layers       adapters       Student      needed
```

## Folder Structure

| Folder | Technique | Task | Model | Script |
|---|---|---|---|---|
| `feature_extraction/` | Frozen backbone, train head only | Classification | MobileNetV3 | `mobilenetv3_small_100_lamb_in1k.py` |
| `fine_tuning/` | Unfreeze + train all layers | Object Detection | RT-DETRv2 | `RTDETR_V2_with_transformer.py` |
| `lora/` | Parameter-efficient adapters | Classification | ViT-Base + LoRA | `vit_lora_image_classification.py` |
| `knowledge_distillation/` | Teacher → Student compression | Classification | ViT→MobileNetV3 | `mobilenet_kd_from_vit.py` |
| `zero_shot/` | No labels, text-guided inference | Classification | CLIP ViT-B/32 | `clip_zero_shot_classification.py` |

## Quick Decision Guide

```
Do you have labeled data?
├── NO  → zero_shot/     (CLIP zero-shot)
└── YES → How much?
          ├── Very little (< 500 images/class)  → lora/     (LoRA/PEFT)
          ├── Limited     (500–2k images/class)  → feature_extraction/
          ├── Medium      (2k–10k images/class)  → fine_tuning/
          └── Large       (> 10k images/class)   → fine_tuning/ (full)

Is your model too large for production?
└── YES → knowledge_distillation/  (compress into smaller student)
```

## Trainable Parameters Comparison

| Technique | % Params Trained | Relative Training Time | When to Use |
|---|---|---|---|
| Zero-Shot | 0% | Instant | No data |
| Feature Extraction | ~1-3% | 1x | Small data |
| LoRA | ~0.5-2% | 1.5x | Small-medium, large model |
| Knowledge Distillation | 100% (student only) | 2x | Compression for deploy |
| Full Fine-tuning | 100% | 3-5x | Medium-large data |

## Common Stack

- **Framework**: HuggingFace Transformers + PEFT + Datasets + Evaluate
- **Backbone zoo**: `timm`, `torchvision.models`, HF Model Hub
- **Experiment tracking**: Weights & Biases (`report_to="wandb"`)
- **Dataset source**: Roboflow (via `load_roboflow_dataset`)
