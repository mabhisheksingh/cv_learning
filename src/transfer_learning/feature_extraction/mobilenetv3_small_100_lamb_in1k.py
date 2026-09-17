import logging
import sys
from pathlib import Path
from time import time
from typing import Any

import albumentations as a
import evaluate
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.utils.utils import (
    get_device,
    load_roboflow_dataset,
    load_hf_dataset_dir,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "mobilenetv3_small_100_lamb_in1k"
DATASET_NAME = "Facial-Expression-Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
CURRENT_DEVICE = get_device()

# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {**acc, **f1}


# 1. Define the augmentation pipeline once outside the function
aug = a.Compose(
    [
        a.HorizontalFlip(p=0.5),
        a.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        a.RandomBrightnessContrast(p=0.2),
        a.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            p=0.3,
        ),
    ]
)


def main():
    logger.info(f"Current device: {CURRENT_DEVICE}")
    logger.info(f"Project root directory: {PROJECT_ROOT}")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Dataset Name: {DATASET_NAME}")

    # ---------------------------------------------------------------------------
    # 1. Download / verify dataset
    # ---------------------------------------------------------------------------
    success, status = load_roboflow_dataset(
        download_path=str(PROJECT_ROOT / "data" / DATASET_NAME),
        project_id="facial-expression-dataset-ytgtk",
        workspace="md-likhon-islam",
        version=1,
        model_format="folder",
    )
    if success:
        logger.info(f"Dataset ready: {status}")
    else:
        logger.error(f"Failed to load dataset: {status}")
        sys.exit(1)

    data_dir = str(PROJECT_ROOT / "data" / DATASET_NAME)

    # ---------------------------------------------------------------------------
    # 2. Load splits via ImageFolder to collect class metadata and Convert ImageFolder -> HuggingFace Dataset (image paths + labels)
    # ---------------------------------------------------------------------------

    dataset_hf = load_hf_dataset_dir(data_dir=data_dir)
    logger.info(f"data columns names :  {dataset_hf.column_names}")
    train_ds = dataset_hf["train"]
    valid_ds = dataset_hf["validation"]
    test_ds = dataset_hf["test"]

    logger.info(
        f"Split sizes — train: {len(train_ds)}, valid: {len(valid_ds)}, test: {len(test_ds)}"
    )

    # ---------------------------------------------------------------------------
    # 3. Load processor and model
    # ---------------------------------------------------------------------------
    model_id = str(PROJECT_ROOT / "models" / MODEL_NAME)

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
    )

    # ---------------------------------------------------------------------------
    # 5. Feature extraction: freeze backbone, keep classifier head trainable
    # ---------------------------------------------------------------------------
    _HEAD_KEYWORDS = {"classifier", "head", "fc", "linear"}
    for name, param in model.named_parameters():
        if not any(kw in name.lower() for kw in _HEAD_KEYWORDS):
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ---------------------------------------------------------------------------
    # 6. On-the-fly image transforms via set_transform (lazy, no caching)
    # ---------------------------------------------------------------------------
    def apply_transforms(batch: dict) -> dict:
        images = [img.convert("RGB") for img in batch["image"]]
        processed = image_processor(images=images, return_tensors="pt")
        batch["pixel_values"] = processed["pixel_values"]
        del batch["image"]
        return batch

    def custom_transform(batch: dict) -> dict[str, Any]:
        aug_images = [
            aug(image=np.array(img.convert("RGB")))["image"] for img in batch["image"]
        ]
        processed = image_processor(images=aug_images, return_tensors="pt")
        batch["pixel_values"] = processed["pixel_values"]
        del batch["image"]
        return batch

    # train_ds = train_ds.with_transform(custom_transform)
    train_ds = train_ds.with_transform(apply_transforms)
    valid_ds = valid_ds.with_transform(apply_transforms)
    test_ds = test_ds.with_transform(apply_transforms)

    # ---------------------------------------------------------------------------
    # 8. TrainingArguments
    # ---------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        num_train_epochs=15,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=1e-3,
        warmup_steps=100,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # lr_scheduler_type="cosine",
        greater_is_better=True,
        logging_steps=20,
        fp16=CURRENT_DEVICE == "cuda",
        bf16=False,
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
    logger.info("Starting feature extraction training…")
    trainer.train()

    # ---------------------------------------------------------------------------
    # 11. Evaluate on held-out test set
    # ---------------------------------------------------------------------------
    logger.info("Evaluating on test set…")
    test_results = trainer.evaluate(test_ds)
    logger.info(f"Test results: {test_results}")

    # ---------------------------------------------------------------------------
    # 12. Save best model + processor
    # ---------------------------------------------------------------------------
    best_model_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_model_dir))
    image_processor.save_pretrained(str(best_model_dir))
    logger.info(f"Best model saved to {best_model_dir}")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
