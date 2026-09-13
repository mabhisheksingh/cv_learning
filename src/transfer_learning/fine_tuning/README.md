# Fine-Tuning

## What is Fine-Tuning?

Fine-tuning unfreezes some or all layers of a pretrained model and trains them on your target dataset. Unlike feature extraction (which only trains the head), fine-tuning adapts the backbone itself to your domain.

## When to Use

- Medium-to-large dataset available (1k+ images per class)
- Target domain differs from pretraining domain (e.g., medical images, satellite imagery)
- You want maximum performance and have compute budget
- Task differs from pretraining task (e.g., object detection instead of classification)

## Variants

| Variant | What Gets Trained | Use Case |
|---|---|---|
| **Full fine-tuning** | All layers | Large dataset, enough GPU |
| **Partial fine-tuning** | Last N layers + head | Medium dataset |
| **Differential LR** | All layers but with different LRs per group | Backbone + head at different paces |
| **Gradual unfreeze** | Layers unlocked epoch by epoch | Avoid catastrophic forgetting |

## How It Works

```
Pretrained Backbone (ImageNet / COCO)
        ↓
  Unfreeze layers
        ↓
  Replace task head
        ↓
  Train with small LR (1e-5 ~ 5e-5)
        ↓
  Warmup + decay scheduler
```

## Key Hyperparameters

- **Learning rate**: Much smaller than training from scratch (1e-5 ~ 1e-4)
- **Warmup**: Prevents sudden large gradient updates at start
- **Weight decay**: Regularisation to avoid overfitting
- **Batch size**: Larger = more stable gradients

## Examples in this folder

| File | Task | Model | Library |
|---|---|---|---|
| `RTDETR_V2_with_transformer.py` | Object Detection | RT-DETRv2 R50 | HuggingFace Transformers |
| `RTDETR_V2_pure_pytorch.py` | Object Detection | RT-DETRv2 R50 | Pure PyTorch + torchvision |

## Resources

- HF Fine-tuning guide: https://huggingface.co/docs/transformers/training
- RT-DETR paper: https://arxiv.org/abs/2304.08069
- Differential LR explained: https://arxiv.org/abs/1801.06146 (ULMFiT)
- Gradual unfreezing: https://arxiv.org/abs/1801.06146
