import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "rtdetr_r50vd"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.utils import get_device, load_coco_split, load_roboflow_dataset  # noqa: E402

# Step 1 : Set basic config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.info(f"Project root directory: {PROJECT_ROOT}")
CURRENT_DEVICE = get_device()

# Step 2 : Load dataset and model
success, status = load_roboflow_dataset(
    download_path=str(PROJECT_ROOT / "data" / "coco_data"),
    project_id="coco",
    workspace="microsoft",
)
if success:
    logger.info(f"Successfully loaded RoboFlow dataset: {status}")
else:
    logger.error(f"Failed to load RoboFlow dataset: {status}")
    sys.exit(1)

coco_data_dir = str(PROJECT_ROOT / "data" / "coco_data")

train_dateset = load_coco_split(coco_data_dir, "train")
test_dataset = load_coco_split(coco_data_dir, "test")
valid_dataset = load_coco_split(coco_data_dir, "valid")

# Load model
model_id = str(PROJECT_ROOT / "models" / MODEL_NAME)

image_processor = RTDetrImageProcessor.from_pretrained(model_id)
model = RTDetrForObjectDetection.from_pretrained(model_id,torch_dtype=torch.float32)
model.to(CURRENT_DEVICE)


def data_collator(batch):
    images = [item[0] for item in batch]
    annotations = []
    for item in batch:
        targets = [t for t in item[1] if t["category_id"] != 0]
        image_id = targets[0]["image_id"] if targets else 0
        mapped_targets = [{**t, "category_id": t["category_id"] - 1} for t in targets]
        annotations.append({"image_id": image_id, "annotations": mapped_targets})

    return image_processor(images=images, annotations=annotations, return_tensors="pt")


# Step 3: Setup training
training_arguments = TrainingArguments(
    output_dir=str(PROJECT_ROOT / "rtdetr_custom"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=1e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# Step 4: Start Training
trainer = Trainer(
    model=model,
    args=training_arguments,
    data_collator=data_collator,
    train_dataset=train_dateset,
    eval_dataset=valid_dataset,
)

trainer.train()