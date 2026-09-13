# Transfer Learning — Complete Guide

## The 5 Industry-Standard Techniques

### 1. Feature Extraction (Frozen Backbone)

**What:** Freeze ALL pretrained layers. Only train the classification head.

**Code pattern:**
```python
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(in_features, num_classes)  # only this trains
```

**When:** Small dataset (<500 images/class), quick prototype, domain similar to ImageNet.

**Interview Q:** *"What is catastrophic forgetting and how does feature extraction avoid it?"*
- Catastrophic forgetting = model forgets old knowledge when trained on new task.
- Feature extraction avoids it by never updating pretrained weights.

---

### 2. Full Fine-Tuning

**What:** Unfreeze all or last N layers. Train with a small learning rate.

**Key insight:** Use **differential learning rates** — backbone gets 10x smaller LR than head.
```python
optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # pretrained, careful
    {"params": model.head.parameters(),     "lr": 1e-4},  # new, can learn faster
])
```

**When:** Medium-large dataset, domain differs from ImageNet (medical, satellite).

**Interview Q:** *"Why use a smaller LR for fine-tuning vs training from scratch?"*
- Pretrained weights are already in a good region. Large LR would destroy them.
- Small LR = nudge in the right direction, not a full restart.

---

### 3. LoRA (Low-Rank Adaptation)

**What:** Add small trainable matrices (rank r) parallel to frozen weight matrices in attention layers.

```
W_new = W_frozen + scale × (B @ A)
# W: 768×768 = 589,824 params
# A: 768×16, B: 16×768 = 24,576 params  (only 4% !)
```

**Key hyperparameters:**
| Param | Typical value | Effect |
|---|---|---|
| `r` (rank) | 4, 8, 16, 32 | Higher = more capacity |
| `lora_alpha` | 2×r | Scaling factor |
| `target_modules` | `["query", "value"]` | Where to inject |

**When:** Large ViT/transformer model, limited GPU, want multiple adapters.

**Interview Q:** *"Why is r×r not the complexity but r×d + d×r?"*
- LoRA uses two matrices: A is (d×r) and B is (r×d). Complexity = 2×d×r, not r².

---

### 4. Knowledge Distillation

**What:** Large teacher model generates soft labels for training a small student model.

```
Loss = α × CE(student, hard_label) + (1-α) × KL(student_soft, teacher_soft)
```

**Temperature T:** Softens probabilities. T=1 → normal softmax. T=4 → soft, more info.
```python
soft_probs = F.softmax(logits / T, dim=-1)  # T=4 spreads probability mass
```

**When:** Model too large for production. Target: latency < X ms on device.

**Interview Q:** *"What is dark knowledge in distillation?"*
- Teacher says: class A = 0.70, class B = 0.25, class C = 0.05.
- Hard label only says: class A = 1. It throws away the B≈C similarity info.
- The soft distribution captures inter-class relationships = "dark knowledge".

---

### 5. Zero-Shot with CLIP

**What:** No training. Compare image embedding with text embeddings of class names.

```python
similarity = image_embed @ text_embeds.T   # cosine similarity
prediction = similarity.argmax()
```

**Prompt engineering matters:**
```
"cat" → bad
"a photo of a cat" → good
ensemble of 5 prompts → best
```

**When:** No labels available, new classes added frequently, baseline check.

**Interview Q:** *"How does CLIP achieve zero-shot transfer?"*
- Trained on 400M (image, text) pairs with contrastive loss.
- Learns a shared embedding space where "a photo of a dog" is close to dog images.
- At inference, class names become the "training signal".

---

## Resources

- CS231n Transfer Learning notes: https://cs231n.github.io/transfer-learning/
- HF Fine-tuning tutorial: https://huggingface.co/docs/transformers/training
- LoRA paper (Hu et al. 2021): https://arxiv.org/abs/2106.09685
- Knowledge Distillation (Hinton 2015): https://arxiv.org/abs/1503.02531
- CLIP paper (Radford 2021): https://arxiv.org/abs/2103.00020
- PEFT library: https://huggingface.co/docs/peft
- timm (1000+ pretrained models): https://huggingface.co/docs/timm
