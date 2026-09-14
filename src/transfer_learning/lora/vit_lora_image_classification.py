import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.utils.utils import (  # noqa: E402
    get_device,
    load_classification_split,
    load_classification_split_with_hf_dataset,
    load_roboflow_dataset,
)

# ---------------------------------------------------------------------------
# 1. Constants & logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "vit_base_patch16_224"
DATASET_NAME = "Facial-Expression-Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"{MODEL_NAME}_lora"
CURRENT_DEVICE = get_device()

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Device: {CURRENT_DEVICE}")

# ---------------------------------------------------------------------------
# 2. Download / verify dataset
# ---------------------------------------------------------------------------
success, status = load_roboflow_dataset(
    download_path=str(PROJECT_ROOT / "data" / DATASET_NAME),
    project_id="facial-expression-dataset-ytgtk",
    workspace="md-likhon-islam",
    version=1,
    model_format="folder",
)
if not success:
    logger.error(f"Failed to load dataset: {status}")
    sys.exit(1)

data_dir = str(PROJECT_ROOT / "data" / DATASET_NAME)

# ---------------------------------------------------------------------------
# 3. Load splits
# ---------------------------------------------------------------------------
train_ds = load_classification_split_with_hf_dataset(data_dir, "train")
valid_ds = load_classification_split_with_hf_dataset(data_dir, "valid")
test_ds = load_classification_split_with_hf_dataset(data_dir, "test")
logger.info(
    f"Split sizes — train: {len(train_ds)}, valid: {len(valid_ds)}, test: {len(test_ds)}"
)

# Derive num_labels from ImageFolder so the model head has the correct size
_tmp = load_classification_split(data_dir, "train")
num_labels = len(_tmp.classes)
label2id = {cls: idx for idx, cls in enumerate(_tmp.classes)}
id2label = {idx: cls for idx, cls in enumerate(_tmp.classes)}
logger.info(f"Num classes: {num_labels}  Labels: {list(label2id.keys())}")

# ---------------------------------------------------------------------------
# 4. Load processor + base model
# ---------------------------------------------------------------------------
model_id = str(PROJECT_ROOT / "models" / MODEL_NAME)

image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# ---------------------------------------------------------------------------
# 5. Apply LoRA
#    - Only adds tiny trainable matrices (rank r) to the query and value
#      projection layers inside every attention block.
#    - modules_to_save=["classifier"] keeps the classification head
#      fully trainable (not through LoRA).
#    - Result: ~0.5-2% of original parameters are trained instead of 100%.
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,  # rank — higher = more capacity, more params
    lora_alpha=32,  # scaling factor (alpha/r = effective scale)
    target_modules=["query", "value"],  # which linear layers to wrap with LoRA
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],  # these are fully trained (not LoRA)
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # shows % of params being trained
model.to(CURRENT_DEVICE)


# ---------------------------------------------------------------------------
# 6. On-the-fly image transforms
# ---------------------------------------------------------------------------
def apply_transforms(batch: dict) -> dict:
    batch["pixel_values"] = [
        image_processor(img.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        for img in batch["image"]
    ]
    del batch["image"]
    return batch


train_ds.set_transform(apply_transforms)
valid_ds.set_transform(apply_transforms)
test_ds.set_transform(apply_transforms)

# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}


# ---------------------------------------------------------------------------
# 8. TrainingArguments
#    LoRA trains far fewer parameters so we can use a slightly larger LR
#    than full fine-tuning.
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,  # LoRA can tolerate higher LR than full FT
    warmup_steps=100,
    weight_decay=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=20,
    fp16=CURRENT_DEVICE == "cuda",
    bf16=False,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    seed=42,
    report_to="none",
    push_to_hub=False,
)

# ---------------------------------------------------------------------------
# 9. Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator(),
)

# ---------------------------------------------------------------------------
# 10. Train
# ---------------------------------------------------------------------------
logger.info("Starting LoRA fine-tuning…")
trainer.train()

# ---------------------------------------------------------------------------
# 11. Evaluate on test set
# ---------------------------------------------------------------------------
logger.info("Evaluating on test set…")
test_results = trainer.evaluate(test_ds)
logger.info(f"Test results: {test_results}")

# ---------------------------------------------------------------------------
# 12. Save  — saves only the LoRA adapter weights (tiny file, not full model)
# ---------------------------------------------------------------------------
best_model_dir = OUTPUT_DIR / "best_model"
trainer.save_model(str(best_model_dir))
image_processor.save_pretrained(str(best_model_dir))
logger.info(f"LoRA adapter saved to {best_model_dir}")
