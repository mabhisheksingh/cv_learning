import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchvision.transforms.v2 as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.models.detection import RTDetr_R50_VD_Weights, rtdetr_r50_vd
from torchvision.ops import box_convert

from src.utils.utils import (  # noqa: E402
    get_device,
    load_coco_split,
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
MODEL_NAME = "rtdetr_r50vd"
DATASET_NAME = "coco_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME
CURRENT_DEVICE = get_device()

# Hyperparameters  (mirror the HF TrainingArguments values)
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200
IMAGE_SIZE = 640
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Device: {CURRENT_DEVICE}")

# ---------------------------------------------------------------------------
# 2. Download / verify dataset   (same as HF version)
# ---------------------------------------------------------------------------
success, status = load_roboflow_dataset(
    download_path=str(PROJECT_ROOT / "data" / DATASET_NAME),
    project_id="coco",
    workspace="microsoft",
    version=50,
)
if not success:
    logger.error(f"Failed to load dataset: {status}")
    sys.exit(1)

coco_data_dir = str(PROJECT_ROOT / "data" / DATASET_NAME)

# ---------------------------------------------------------------------------
# 3. Load splits   (CocoDetection returns (PIL Image, List[dict]))
# ---------------------------------------------------------------------------
train_dataset = load_coco_split(coco_data_dir, "train")
valid_dataset = load_coco_split(coco_data_dir, "valid")
test_dataset = load_coco_split(coco_data_dir, "test")
logger.info(
    f"Split sizes — train: {len(train_dataset)}, "
    f"valid: {len(valid_dataset)}, test: {len(test_dataset)}"
)

# ---------------------------------------------------------------------------
# 4. Transforms
#    torchvision.transforms.v2 understands tv_tensors types, so when we
#    pass (Image, BoundingBoxes) together, Resize automatically scales
#    both the pixels AND the box coordinates. No manual math needed.
# ---------------------------------------------------------------------------
train_transforms = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

eval_transforms = T.Compose(
    [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# 5. Collate function  (HF alternative: data_collator calling image_processor)
#
#    What it does:
#      - Converts PIL images → float tensors using the given transform.
#      - Converts COCO bbox [x,y,w,h] → xyxy format expected by the model.
#      - Filters background class (category_id=0) and shifts to 0-indexed.
#      - Wraps boxes as tv_tensors.BoundingBoxes so v2 transforms scale them
#        correctly when the image is resized.
#      - Returns (List[Tensor], List[dict]) — the native torchvision API.
# ---------------------------------------------------------------------------
def make_collate_fn(transform: T.Compose):
    def collate_fn(batch):
        images, targets = [], []
        for pil_image, annotations in batch:
            orig_w, orig_h = pil_image.size

            valid_anns = [a for a in annotations if a["category_id"] != 0]
            if valid_anns:
                raw_boxes = torch.tensor(
                    [a["bbox"] for a in valid_anns], dtype=torch.float32
                )
                raw_boxes = box_convert(raw_boxes, in_fmt="xywh", out_fmt="xyxy")
                labels = torch.tensor(
                    [a["category_id"] - 1 for a in valid_anns], dtype=torch.int64
                )
            else:
                raw_boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            tv_img = tv_tensors.Image(pil_image)
            tv_boxes = tv_tensors.BoundingBoxes(
                raw_boxes, format="XYXY", canvas_size=(orig_h, orig_w)
            )

            tv_img, tv_boxes = transform(tv_img, tv_boxes)

            images.append(tv_img)
            targets.append(
                {
                    "boxes": tv_boxes.as_subclass(torch.Tensor),
                    "labels": labels,
                }
            )

        return images, targets

    return collate_fn


# ---------------------------------------------------------------------------
# 6. DataLoaders  (HF alternative: Trainer handles this internally)
# ---------------------------------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=make_collate_fn(train_transforms),
    pin_memory=False,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=make_collate_fn(eval_transforms),
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=make_collate_fn(eval_transforms),
    pin_memory=False,
)

# ---------------------------------------------------------------------------
# 7. Model  (HF alternative: RTDetrForObjectDetection.from_pretrained)
#    torchvision's rtdetr_r50_vd in train mode returns a dict of losses.
#    In eval mode it returns predictions  (same behaviour as Faster-RCNN).
# ---------------------------------------------------------------------------
model = rtdetr_r50_vd(weights=RTDetr_R50_VD_Weights.DEFAULT)
model.to(CURRENT_DEVICE)
logger.info(f"Model loaded on device: {CURRENT_DEVICE}")

# ---------------------------------------------------------------------------
# 8. Optimizer & LR scheduler  (HF alternative: Trainer sets these up)
#    - AdamW with weight-decay
#    - Linear warmup for WARMUP_STEPS, then linear decay to 0
# ---------------------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

steps_per_epoch = len(train_loader)
total_steps = NUM_EPOCHS * steps_per_epoch

warmup_sched = LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_STEPS
)
decay_sched = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.0,
    total_iters=max(total_steps - WARMUP_STEPS, 1),
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_sched, decay_sched],
    milestones=[WARMUP_STEPS],
)


# ---------------------------------------------------------------------------
# 9. Train & Eval loops  (HF alternative: Trainer.train() / Trainer.evaluate())
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 10 == 0:
            logger.info(f"  step {step}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.train()  # keep train mode so the model returns losses, not predictions
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# 10. Training loop  (HF alternative: trainer.train())
# ---------------------------------------------------------------------------
best_eval_loss = float("inf")
best_model_dir = OUTPUT_DIR / "best_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Starting fine-tuning…")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, CURRENT_DEVICE
    )
    eval_loss = evaluate(model, valid_loader, CURRENT_DEVICE)
    logger.info(
        f"Epoch {epoch}/{NUM_EPOCHS}  "
        f"train_loss={train_loss:.4f}  eval_loss={eval_loss:.4f}"
    )

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_model_dir / "model.pth")
        logger.info(f"  -> Best model saved (eval_loss={best_eval_loss:.4f})")

# ---------------------------------------------------------------------------
# 11. Test evaluation  (HF alternative: trainer.evaluate(test_dataset))
# ---------------------------------------------------------------------------
logger.info("Evaluating on test set…")
model.load_state_dict(
    torch.load(best_model_dir / "model.pth", map_location=CURRENT_DEVICE)
)
test_loss = evaluate(model, test_loader, CURRENT_DEVICE)
logger.info(f"Test loss: {test_loss:.4f}")
logger.info(f"Best model saved at: {best_model_dir}")
