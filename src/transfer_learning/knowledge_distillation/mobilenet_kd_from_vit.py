import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
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
TEACHER_MODEL_NAME = "vit_base_patch16_224"
STUDENT_MODEL_NAME = "mobilenetv3_small_100_lamb_in1k"
DATASET_NAME = "Facial-Expression-Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mobilenet_kd"
CURRENT_DEVICE = get_device()

KD_TEMPERATURE = 4.0  # higher → softer probability distribution from teacher
KD_ALPHA = 0.5  # 0.5 = equal weight between hard labels and teacher soft labels

logger.info(f"Device: {CURRENT_DEVICE}")
logger.info(f"Teacher: {TEACHER_MODEL_NAME}  |  Student: {STUDENT_MODEL_NAME}")

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

# Derive num_labels and label mappings from the ImageFolder layout
_tmp = load_classification_split(data_dir, "train")
num_labels = len(_tmp.classes)
label2id = {cls: idx for idx, cls in enumerate(_tmp.classes)}
id2label = {idx: cls for idx, cls in enumerate(_tmp.classes)}
logger.info(f"Num classes: {num_labels}  Labels: {list(label2id.keys())}")

# ---------------------------------------------------------------------------
# 4. Load teacher model (frozen — only used for inference during training)
#    The teacher is the larger, more accurate model that we distill from.
# ---------------------------------------------------------------------------
teacher_model_id = str(PROJECT_ROOT / "models" / TEACHER_MODEL_NAME)
teacher_processor = AutoImageProcessor.from_pretrained(teacher_model_id)
teacher_model = AutoModelForImageClassification.from_pretrained(
    teacher_model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
teacher_model.eval()
teacher_model.to(CURRENT_DEVICE)
for param in teacher_model.parameters():
    param.requires_grad = False
logger.info("Teacher model loaded and frozen.")

# ---------------------------------------------------------------------------
# 5. Load student model (trainable — this is the small model we want to train)
# ---------------------------------------------------------------------------
student_model_id = str(PROJECT_ROOT / "models" / STUDENT_MODEL_NAME)
student_processor = AutoImageProcessor.from_pretrained(student_model_id)
student_model = AutoModelForImageClassification.from_pretrained(
    student_model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
student_model.to(CURRENT_DEVICE)

trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in student_model.parameters())
logger.info(f"Student trainable params: {trainable:,} / {total:,}")


# ---------------------------------------------------------------------------
# 6. On-the-fly transforms
#    IMPORTANT: teacher and student may have different normalization stats
#    (e.g. ViT uses mean=0.5, MobileNetV3 uses ImageNet mean=0.485...).
#    We store both sets of pixel_values in the batch so each model receives
#    correctly normalized inputs. remove_unused_columns=False is required.
# ---------------------------------------------------------------------------
def apply_transforms(batch: dict) -> dict:
    batch["pixel_values"] = [
        student_processor(img.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        for img in batch["image"]
    ]
    batch["teacher_pixel_values"] = [
        teacher_processor(img.convert("RGB"), return_tensors="pt")[
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
# 7. Custom Trainer with Knowledge Distillation loss
#
#    Loss = α × CrossEntropy(student_logits, hard_labels)
#         + (1-α) × KL(softmax(student/T) || softmax(teacher/T)) × T²
#
#    The T² term compensates for the gradient scaling effect of temperature.
# ---------------------------------------------------------------------------
class KnowledgeDistillationTrainer(Trainer):
    def __init__(
        self, teacher_model: torch.nn.Module, temperature: float, alpha: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        teacher_pixel_values = inputs.pop("teacher_pixel_values", None)

        student_out = model(**inputs)
        student_logits = student_out.logits

        with torch.no_grad():
            teacher_out = self.teacher_model(pixel_values=teacher_pixel_values)
            teacher_logits = teacher_out.logits

        T = self.temperature

        hard_loss = F.cross_entropy(student_logits, labels)

        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return (loss, student_out) if return_outputs else loss


# ---------------------------------------------------------------------------
# 8. Metrics
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
# 9. TrainingArguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-3,
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
# 10. Trainer (KD version)
# ---------------------------------------------------------------------------
trainer = KnowledgeDistillationTrainer(
    teacher_model=teacher_model,
    temperature=KD_TEMPERATURE,
    alpha=KD_ALPHA,
    model=student_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=student_processor,
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator(),
)

# ---------------------------------------------------------------------------
# 11. Train
# ---------------------------------------------------------------------------
logger.info("Starting Knowledge Distillation training…")
trainer.train()

# ---------------------------------------------------------------------------
# 12. Evaluate on test set
# ---------------------------------------------------------------------------
logger.info("Evaluating on test set…")
test_results = trainer.evaluate(test_ds)
logger.info(f"Test results: {test_results}")

# ---------------------------------------------------------------------------
# 13. Save student model
# ---------------------------------------------------------------------------
best_model_dir = OUTPUT_DIR / "best_model"
trainer.save_model(str(best_model_dir))
student_processor.save_pretrained(str(best_model_dir))
logger.info(f"Student model saved to {best_model_dir}")
