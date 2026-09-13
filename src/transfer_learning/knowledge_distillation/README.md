# Knowledge Distillation (Teacher → Student)

## What is Knowledge Distillation?

A large pretrained **teacher** model teaches a smaller **student** model. Instead of training the student only on hard labels (0 or 1), the student also learns from the **soft probability distributions** (dark knowledge) produced by the teacher.

```
Teacher (ViT-Base, 86M params)  →  soft probabilities  →  Student (MobileNetV3, 2.5M params)
                                                              ↑
                                               also learns from hard labels
```

## Why It Works

The teacher's output `[0.70, 0.20, 0.07, 0.03]` contains more information than the hard label `[1, 0, 0, 0]`. The student learns that class 2 looks similar to class 1 — this inter-class relationship is the "dark knowledge".

## Loss Function

```
Total Loss = α × CE(student_logits, hard_labels)
           + (1 - α) × KL(softmax(student/T), softmax(teacher/T))
```

- **α**: Balance between hard and soft loss (typically 0.3-0.7)
- **T (Temperature)**: Softens probability distributions. Higher T = softer, more info. (T=4 is common)
- **KL Divergence**: Measures how different student's distribution is from teacher's

## When to Use

- You have a large, accurate teacher model already trained
- Need a lightweight model for deployment (mobile, edge devices)
- Want better accuracy than training the student from scratch
- Latency / model size constraints in production

## Variants

| Variant | What is Distilled |
|---|---|
| **Response-based (classic)** | Final logits / probabilities |
| **Feature-based** | Intermediate feature maps |
| **Relation-based** | Relationships between data samples |

## Example in this folder

| File | Teacher | Student | Task |
|---|---|---|---|
| `mobilenet_kd_from_vit.py` | ViT-Base-224 | MobileNetV3-Small | Facial Expression Classification |

## Resources

- KD original paper (Hinton 2015): https://arxiv.org/abs/1503.02531
- TinyViT (distillation for ViT): https://arxiv.org/abs/2207.10666
- HF blog on KD: https://huggingface.co/blog/knowledge-distillation
- DistilBERT (NLP KD reference): https://arxiv.org/abs/1910.01108
