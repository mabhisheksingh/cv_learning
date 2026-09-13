import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from src.utils.utils import (  # noqa: E402
    get_device,
    load_coco_split,
    load_roboflow_dataset,
)
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    Trainer,
    TrainingArguments,
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
MODEL_NAME = "rtdetr_r50vd"
DATASET_NAME = "coco_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
CURRENT_DEVICE = get_device()

logger.info(f"Project root directory: {PROJECT_ROOT}")
logger.info(f"Model Name: {MODEL_NAME}")
logger.info(f"Dataset Name: {DATASET_NAME}")
logger.info(f"Current device: {CURRENT_DEVICE}")

# ---------------------------------------------------------------------------
# 2. Download / verify dataset
# ---------------------------------------------------------------------------
success, status = load_roboflow_dataset(
    download_path=str(PROJECT_ROOT / "data" / DATASET_NAME),
    project_id="coco",
    workspace="microsoft",
    version=50,
)
if success:
    logger.info(f"Dataset ready: {status}")
else:
    logger.error(f"Failed to load dataset: {status}")
    sys.exit(1)

coco_data_dir = str(PROJECT_ROOT / "data" / DATASET_NAME)

# ---------------------------------------------------------------------------
# 3. Load splits
# ---------------------------------------------------------------------------
train_dataset = load_coco_split(coco_data_dir, "train")
valid_dataset = load_coco_split(coco_data_dir, "valid")
test_dataset = load_coco_split(coco_data_dir, "test")
logger.info(
    f"Split sizes — train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}"
)

# ---------------------------------------------------------------------------
# 4. Load processor and model
# ---------------------------------------------------------------------------
model_id = str(PROJECT_ROOT / "models" / MODEL_NAME)

image_processor = RTDetrImageProcessor.from_pretrained(model_id)
model = RTDetrForObjectDetection.from_pretrained(model_id,torch_dtype=torch.float32)
model.to(CURRENT_DEVICE)
logger.info(f"Model loaded on device: {next(model.parameters()).device}")

# ---------------------------------------------------------------------------
# 5. Data collator (COCO annotation → processor input)
# ---------------------------------------------------------------------------
def data_collator(batch):
    images = [item[0] for item in batch]
    annotations = []
    for item in batch:
        targets = [t for t in item[1] if t["category_id"] != 0]
        image_id = targets[0]["image_id"] if targets else 0
        mapped_targets = [{**t, "category_id": t["category_id"] - 1} for t in targets]
        annotations.append({"image_id": image_id, "annotations": mapped_targets})
    return image_processor(images=images, annotations=annotations, return_tensors="pt")

# ---------------------------------------------------------------------------
# 6. TrainingArguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=200,
    weight_decay=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    fp16=CURRENT_DEVICE == "cuda",
    bf16=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    seed=42,
    report_to="none",
    push_to_hub=False,
)
logger.info(f"Trainer will use device: {training_args.device}")

# ---------------------------------------------------------------------------
# 7. Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    processing_class=image_processor,
)

# ---------------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------------
logger.info("Starting fine-tuning…")
trainer.train()

# ---------------------------------------------------------------------------
# 9. Evaluate on held-out test set
# ---------------------------------------------------------------------------
logger.info("Evaluating on test set…")
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")

# ---------------------------------------------------------------------------
# 10. Save best model + processor
# ---------------------------------------------------------------------------
best_model_dir = OUTPUT_DIR / "best_model"
trainer.save_model(str(best_model_dir))
image_processor.save_pretrained(str(best_model_dir))
logger.info(f"Best model saved to {best_model_dir}")
