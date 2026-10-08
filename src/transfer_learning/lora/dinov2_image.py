
import logging
import sys
from collections import Counter
from pathlib import Path
from time import time

import albumentations as A
import evaluate
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from peft import LoraConfig, get_peft_model
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
MODEL_NAME = "dinov2-small"
DATASET_NAME = "Facial-Expression-Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
CURRENT_DEVICE = get_device()


def main():
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

    dataset_hf = load_hf_dataset_dir(data_dir=data_dir)
    logger.info(f"data columns names :  {dataset_hf.column_names}")
    train_ds = dataset_hf["train"]
    valid_ds = dataset_hf["validation"]
    test_ds = dataset_hf["test"]
    logger.info(
        f"Split sizes — train: {len(train_ds)}, valid: {len(valid_ds)}, test: {len(test_ds)}"
    )

    class_counts = Counter(train_ds["label"])
    logger.info(f"Train class distribution: {dict(class_counts)}")

    # ---------------------------------------------------------------------------
    # 3. Load processor and model
    # ---------------------------------------------------------------------------
    model_id = str(PROJECT_ROOT / "models" / MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=train_ds.features["label"].num_classes,
    )

    # Define lora config

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------------------
    # 5. Metrics
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

    # ---------------------------------------------------------------------------
    # 6. On-the-fly image transforms
    # ---------------------------------------------------------------------------
    # Normalization values come from the pretrained processor config
    mean = image_processor.image_mean
    std = image_processor.image_std
    size = image_processor.size["shortest_edge"]

    train_aug = A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    eval_aug = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def apply_train_transforms(batch: dict) -> dict:
        images = [np.array(img.convert("RGB")) for img in batch["image"]]
        batch["pixel_values"] = torch.stack(
            [train_aug(image=img)["image"] for img in images]
        )
        del batch["image"]
        return batch

    def apply_eval_transforms(batch: dict) -> dict:
        images = [np.array(img.convert("RGB")) for img in batch["image"]]
        batch["pixel_values"] = torch.stack(
            [eval_aug(image=img)["image"] for img in images]
        )
        del batch["image"]
        return batch

    train_ds.set_transform(apply_train_transforms)
    valid_ds.set_transform(apply_eval_transforms)
    test_ds.set_transform(apply_eval_transforms)

    # ---------------------------------------------------------------------------
    # 7. TrainingArguments (uses AdamW by default, which is Adam + weight decay)
    # ---------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        warmup_steps=100,
        weight_decay=1e-4,
        label_smoothing_factor=0.1,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        fp16=CURRENT_DEVICE == "cuda",
        bf16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=42,
        report_to="none",
        push_to_hub=False,
    )

    # ---------------------------------------------------------------------------
    # 8. Trainer
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
    # 9. Train
    # ---------------------------------------------------------------------------
    logger.info("Starting LoRA fine-tuning…")
    trainer.train()

    # ---------------------------------------------------------------------------
    # 10. Evaluate on test set
    # ---------------------------------------------------------------------------
    logger.info("Evaluating on test set…")
    test_results = trainer.evaluate(test_ds)
    logger.info(f"Test results: {test_results}")

    # ---------------------------------------------------------------------------
    # 11. Save LoRA adapter
    # ---------------------------------------------------------------------------
    best_model_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_model_dir))
    image_processor.save_pretrained(str(best_model_dir))
    logger.info(f"LoRA adapter saved to {best_model_dir}")


if __name__ == "__main__":
    start_time = time()
    logger.info(f"Current device: {CURRENT_DEVICE}")
    logger.info(f"Project root directory: {PROJECT_ROOT}")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Dataset Name: {DATASET_NAME}")
    main()
    end_time = time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")