# Zero-Shot Transfer Learning (CLIP)

## What is Zero-Shot Transfer?

The model classifies images into categories it has **never been explicitly trained on**, using only natural language descriptions of those categories. No labeled images needed at all.

```
Image of a happy face
        ↓
    CLIP Model
        ↓
Compare image embedding with text embeddings:
  "a photo of a happy face"     → similarity: 0.91  ✓ (highest)
  "a photo of a sad face"       → similarity: 0.12
  "a photo of an angry face"    → similarity: 0.08
```

## How CLIP Works

CLIP (Contrastive Language-Image Pretraining) is trained on 400M image-text pairs from the internet. It learns a shared embedding space where matching image-text pairs are close together.

```
Image Encoder (ViT)  →  image embedding (512-d)
                                ↕ cosine similarity
Text Encoder (Transformer)  →  text embedding (512-d)
```

## Zero-Shot vs Few-Shot vs Fine-tuning

| Approach | Training Images Needed | Accuracy |
|---|---|---|
| **Zero-shot** (CLIP) | 0 | Good baseline |
| **Few-shot** (1-10 per class) | Very few | Better |
| **Fine-tuning** | Many | Best |
| **Feature Extraction** | Some | Good |

## When to Use Zero-Shot

- No labeled data available at all
- Quickly evaluate if a task is solvable
- New classes added frequently (just change the text prompt)
- Prototype before investing in labeling

## Prompt Engineering Matters

```python
# Bad prompt
"angry"

# Better prompt
"a photo of an angry face"

# Even better (ensemble)
[
  "a photo of an angry person",
  "a facial expression showing anger",
  "someone who looks very angry",
]
```

## Example in this folder

| File | Model | Task |
|---|---|---|
| `clip_zero_shot_classification.py` | CLIP ViT-B/32 | Facial Expression Zero-Shot Eval |

## Resources

- CLIP paper: https://arxiv.org/abs/2103.00020
- OpenAI CLIP blog: https://openai.com/research/clip
- HF CLIP docs: https://huggingface.co/docs/transformers/model_doc/clip
- Prompt engineering for CLIP: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
