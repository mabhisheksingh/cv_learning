# LoRA / PEFT (Parameter-Efficient Fine-Tuning)

## What is LoRA?

LoRA freezes all pretrained weights and injects small trainable **low-rank matrices** alongside the original weight matrices in attention layers. Instead of updating a full W (d×d), it learns two small matrices A (d×r) and B (r×d) where r << d.

```
W_new = W_frozen + B × A    (r = 16, vs d = 768 for ViT-Base)
Trainable params: ~0.5% instead of 100%
```

## When to Use

- Large Vision Transformer model + small/medium dataset
- Limited GPU memory (only adapter weights need gradients)
- You want to store multiple task-specific adapters cheaply
- Fast iteration on experiments (trains 3-5x faster than full FT)

## Key Concepts

| Concept | Explanation |
|---|---|
| `r` (rank) | Size of the low-rank bottleneck. r=4 (tiny) to r=64 (larger). Start with r=16 |
| `lora_alpha` | Scale factor. Effective scale = alpha/r. Usually set alpha=2×r |
| `target_modules` | Which linear layers get LoRA. For ViT: `["query", "value"]` |
| `modules_to_save` | Layers trained normally (not LoRA). Usually `["classifier"]` |
| `bias` | Whether to train bias terms. Usually `"none"` |

## LoRA vs Other PEFT Methods

| Method | Trainable Params | Best For |
|---|---|---|
| **LoRA** | ~0.5-2% | Attention layers, transformers |
| **Prefix Tuning** | <0.1% | When you can't modify weights |
| **Adapter** | ~3-5% | More capacity needed |
| **Full Fine-tuning** | 100% | Large dataset, max performance |

## Example in this folder

| File | Task | Model |
|---|---|---|
| `vit_lora_image_classification.py` | Image Classification (Facial Expressions) | ViT-Base + LoRA via PEFT |

## Resources

- LoRA paper: https://arxiv.org/abs/2106.09685
- HF PEFT docs: https://huggingface.co/docs/peft
- PEFT for vision: https://huggingface.co/docs/peft/task_guides/image_classification_lora
- LoRA explained visually: https://lightning.ai/pages/community/lora-insights/
