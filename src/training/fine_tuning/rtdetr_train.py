import os
import glob
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    RTDetrForObjectDetection
)
from huggingface_hub import snapshot_download

RAW_MODEL_NAME = "PekingU/rtdetr_r50vd"
LOCAL_MODEL_DIR = "model/rtdetr_r50vd"


def download_model_if_not_exists(model_name: str, local_dir: str) -> str:
    """Download model from Hugging Face if not present in local directory"""
    local_path = Path(local_dir)

    if local_path.exists():
        print(f"Model already exists at: {local_path.absolute()}")
        return str(local_path.absolute())

    print(f"Downloading model {model_name} to {local_path.absolute()}...")
    try:
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Falling back to loading from Hugging Face Hub...")
        return model_name


class VisDroneYOLODataset(Dataset):
    """Dataset class for YOLO-formatted VisDrone data"""

    def __init__(self, images_dir: str, labels_dir: str, processor, split: str = "train"):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.processor = processor
        self.split = split

        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            self.image_files.extend(glob.glob(str(self.images_dir / ext)))
        self.image_files = sorted(self.image_files)

        print(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def _load_yolo_annotations(self, label_path: str, image_width: int, image_height: int) -> Dict:
        """Load YOLO format annotations and convert to absolute coordinates"""
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height

                    # Convert to [x1, y1, x2, y2] format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        # Get corresponding label file
        image_name = Path(image_path).stem
        label_path = self.labels_dir / f"{image_name}.txt"

        # Load annotations
        annotations = self._load_yolo_annotations(str(label_path), image.width, image.height)

        # Format annotations for COCO / RT-DETR processor
        # bbox format: [x, y, width, height] as plain Python floats
        ann_list = []
        for i in range(len(annotations["labels"])):
            x1 = float(annotations["boxes"][i][0])
            y1 = float(annotations["boxes"][i][1])
            x2 = float(annotations["boxes"][i][2])
            y2 = float(annotations["boxes"][i][3])
            ann_list.append({
                "id": i,
                "category_id": int(annotations["labels"][i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })

        # The processor expects a list of per-image annotation dicts
        target = {
            "image_id": idx,
            "annotations": ann_list
        }

        # Process image
        encoding = self.processor(
            images=image,
            annotations=[target],  # one element per image
            return_tensors="pt"
        )

        # Remove batch dimension added by processor
        pixel_values = encoding["pixel_values"].squeeze(0)

        # labels may be a list (per image) or a dict
        labels_raw = encoding.get("labels", encoding.get("pixel_mask", {}))
        if isinstance(labels_raw, list):
            labels = labels_raw[0]  # single image in batch
        else:
            labels = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                      for k, v in labels_raw.items()}

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad pixel values to same size
    pixel_values = torch.stack(pixel_values)

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def train_rtdetr_model():
    # Configuration
    dataset_yaml = "../dataset/VisDrone.yaml"
    project_dir = "model"
    exp_name = "visdrone_rtdetr_r50vd"

    # Training hyperparameters (from train.py)
    epochs = 100
    batch_size = 4  # RT-DETR is memory intensive
    learning_rate = 1e-4
    weight_decay = 1e-4
    warmup_ratio = 0.1
    gradient_accumulation_steps = 2
    workers = 4
    seed = 42

    # Set random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Auto-detect device: NVIDIA (CUDA) > Mac (MPS) > CPU
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Parse dataset paths from VisDrone.yaml structure
    # Script is run from src/training/fine_tuning/, dataset is at data/VisDrone_Dataset
    base_dir = Path(__file__).parent.parent.parent.parent / "data" / "VisDrone_Dataset"
    train_images = base_dir / "VisDrone2019-DET-train" / "images"
    train_labels = base_dir / "VisDrone2019-DET-train" / "labels"
    val_images = base_dir / "VisDrone2019-DET-val" / "images"
    val_labels = base_dir / "VisDrone2019-DET-val" / "labels"

    # Download model if not exists locally
    model_path = download_model_if_not_exists(RAW_MODEL_NAME, LOCAL_MODEL_DIR)

    # Load processor and model
    print(f"Loading model from: {model_path}")
    processor = AutoImageProcessor.from_pretrained(model_path,trust_remote_code=True,backend=device )
    model = RTDetrForObjectDetection.from_pretrained(model_path)

    # Update classification head for VisDrone (10 classes)
    num_classes = 10

    # RTDetrForObjectDetection uses model.model.decoder.class_labels_classifier
    classifier = None
    if hasattr(model, 'class_labels_classifier'):
        classifier = model.class_labels_classifier
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and \
            hasattr(model.model.decoder, 'class_labels_classifier'):
        classifier = model.model.decoder.class_labels_classifier

    if classifier is not None:
        in_features = classifier.in_features
        new_head = nn.Linear(in_features, num_classes)
        if hasattr(model, 'class_labels_classifier'):
            model.class_labels_classifier = new_head
        else:
            model.model.decoder.class_labels_classifier = new_head
        print(f"Replaced classifier head: {in_features} -> {num_classes} classes")
    else:
        print("Warning: Could not find classifier head. Training with original head.")

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = VisDroneYOLODataset(
        str(train_images),
        str(train_labels),
        processor,
        split="train"
    )
    val_dataset = VisDroneYOLODataset(
        str(val_images),
        str(val_labels),
        processor,
        split="val"
    )

    # Prepare dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )

    # Move model to device
    model.to(device)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Must be > 0
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Create project directory
    os.makedirs(project_dir, exist_ok=True)

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [
                {k: v.to(device) for k, v in label.items()}
                for label in batch["labels"]
            ]

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

            if step % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {k: v.to(device) for k, v in label.items()}
                    for label in batch["labels"]
                ]

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )

                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(project_dir, f"{exp_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(project_dir, f"{exp_name}_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, save_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training completed!")

    # Save final model
    final_save_path = os.path.join(project_dir, f"{exp_name}_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, final_save_path)
    print(f"Saved final model to {final_save_path}")


if __name__ == "__main__":
    train_rtdetr_model()