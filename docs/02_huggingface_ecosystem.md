# HuggingFace Ecosystem for Computer Vision

## Core Libraries

| Library | Purpose | Docs |
|---|---|---|
| `transformers` | Models + Trainer | https://huggingface.co/docs/transformers |
| `datasets` | Dataset loading + processing | https://huggingface.co/docs/datasets |
| `evaluate` | Metrics (accuracy, mAP, F1) | https://huggingface.co/docs/evaluate |
| `peft` | LoRA, Adapters, Prefix Tuning | https://huggingface.co/docs/peft |
| `accelerate` | Multi-GPU / distributed training | https://huggingface.co/docs/accelerate |
| `timm` | 1000+ pretrained CV models | https://huggingface.co/docs/timm |

---

## `Trainer` Class — Key Arguments

```python
Trainer(
    model=model,                    # nn.Module or HF PreTrainedModel
    args=training_args,             # TrainingArguments
    train_dataset=train_ds,         # torch Dataset or HF Dataset
    eval_dataset=valid_ds,
    data_collator=collator,         # how to batch samples
    processing_class=processor,     # image/text processor (for saving)
    compute_metrics=fn,             # evaluation metrics function
    optimizers=(optimizer, sched),  # optional custom optimizer tuple
    callbacks=[...],                # EarlyStoppingCallback, etc.
)
```

**Override `compute_loss` for custom training objectives** (e.g., Knowledge Distillation):
```python
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = my_custom_loss(outputs, inputs)
        return (loss, outputs) if return_outputs else loss
```

---

## `TrainingArguments` — Most Important Params

```python
TrainingArguments(
    output_dir="outputs/",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    warmup_steps=200,              # LR warmup
    weight_decay=1e-4,             # L2 regularisation
    eval_strategy="epoch",         # or "steps"
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,                     # mixed precision on CUDA
    bf16=False,                    # use on A100/H100
    gradient_accumulation_steps=4, # simulate larger batch
    remove_unused_columns=False,   # IMPORTANT for CV custom datasets
    report_to="wandb",             # experiment tracking
)
```

**`optim` shortcuts** (no need for custom optimizer for these):
```python
optim="adamw_torch"     # default
optim="adafactor"       # memory efficient
optim="adamw_8bit"      # bitsandbytes, GPU memory saver
```

---

## `datasets` Library Patterns

```python
from datasets import load_dataset

# Load from HF Hub
ds = load_dataset("imagenet-1k", split="train")

# Load from local folder (ImageFolder layout)
ds = load_dataset("imagefolder", data_dir="./data/my_dataset")

# On-the-fly transforms (lazy, applied at batch fetch time)
def transforms(batch):
    batch["pixel_values"] = [processor(img) for img in batch["image"]]
    return batch
ds.set_transform(transforms)    # fastest, no disk caching

# Pre-computed transforms (cached to disk)
ds = ds.map(transforms, batched=True, num_proc=4)
```

**`set_transform` vs `.map()`:**
| | `set_transform` | `.map()` |
|---|---|---|
| When applied | At batch fetch | Preprocessed once |
| Disk cache | No | Yes |
| Speed (first run) | Fast | Slow (processing) |
| Speed (repeat runs) | Same | Faster (from cache) |
| Best for | Augmentations (random) | Fixed transforms |

---

## `evaluate` Library

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **accuracy.compute(predictions=preds, references=labels),
        **f1.compute(predictions=preds, references=labels, average="weighted"),
    }
```

Available metrics: `accuracy`, `f1`, `precision`, `recall`, `mean_iou`, `coco` (for detection).

---

## PEFT Quick Reference

```python
from peft import LoraConfig, get_peft_model, PeftModel

# Apply LoRA to a model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["query","value"])
model  = get_peft_model(base_model, config)
model.print_trainable_parameters()   # shows how few params train

# Save only adapter (tiny file)
model.save_pretrained("./adapter_weights")

# Load adapter on top of base model later
model = PeftModel.from_pretrained(base_model, "./adapter_weights")

# Merge adapter into base model (for deployment, no extra latency)
model = model.merge_and_unload()
```

---

## Model Hub — Key CV Models

| Model | HF ID | Task | Params |
|---|---|---|---|
| ViT-Base | `google/vit-base-patch16-224` | Classification | 86M |
| Swin-Tiny | `microsoft/swin-tiny-patch4-window7-224` | Classification | 28M |
| MobileNetV3 | `timm/mobilenetv3_small_100.lamb_in1k` | Classification | 2.5M |
| RT-DETRv2 | `PekingU/rtdetr_r50vd` | Detection | 42M |
| CLIP | `openai/clip-vit-base-patch32` | Zero-shot | 150M |
| DINO v2 | `facebook/dinov2-base` | Features / seg | 86M |
| SAM | `facebook/sam-vit-base` | Segmentation | 91M |

---

## Resources

- HF Transformers docs: https://huggingface.co/docs/transformers
- HF Course (free): https://huggingface.co/learn/nlp-course (NLP but concepts apply)
- HF CV tasks guide: https://huggingface.co/docs/transformers/tasks/image_classification
- timm model list: https://huggingface.co/docs/timm/en/reference/models
- PEFT methods paper: https://arxiv.org/abs/2304.01933
