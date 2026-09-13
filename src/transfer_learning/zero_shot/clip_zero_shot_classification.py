"""
Zero-shot image classification with CLIP.

No training required — CLIP compares image embeddings against text prompt
embeddings to classify without ever seeing labeled examples.
"""

import logging
import sys
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from src.utils.utils import (  # noqa: E402
    get_device,
    load_classification_split,
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
MODEL_NAME   = "clip_vit_base_patch32"   # local folder name, or use HF hub ID
DATASET_NAME = "Facial-Expression-Dataset"
CURRENT_DEVICE = get_device()

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
# 3. Load CLIP model + processor
#    Use local path or fallback to HuggingFace Hub.
# ---------------------------------------------------------------------------
model_path = PROJECT_ROOT / "models" / MODEL_NAME
model_id   = str(model_path) if model_path.is_dir() else "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model     = CLIPModel.from_pretrained(model_id)
model.eval()
model.to(CURRENT_DEVICE)
logger.info(f"CLIP model loaded from: {model_id}")

# ---------------------------------------------------------------------------
# 4. Define class names and text prompts
#    Prompt engineering significantly affects zero-shot accuracy.
#    Using an ensemble of prompts per class and averaging is best practice.
# ---------------------------------------------------------------------------
CLASS_NAMES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

PROMPT_TEMPLATES = [
    "a photo of a {} face",
    "a facial expression showing {}",
    "someone who looks {}",
    "a person with a {} expression",
]

def build_text_prompts(class_names: list[str], templates: list[str]) -> list[str]:
    return [tmpl.format(cls) for cls in class_names for tmpl in templates]

text_prompts = build_text_prompts(CLASS_NAMES, PROMPT_TEMPLATES)
logger.info(f"Total text prompts: {len(text_prompts)} ({len(PROMPT_TEMPLATES)} per class)")

# Pre-compute text embeddings once (they don't change across images)
with torch.no_grad():
    text_inputs  = processor(text=text_prompts, return_tensors="pt", padding=True).to(CURRENT_DEVICE)
    text_embeds  = model.get_text_features(**text_inputs)
    text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)   # L2 normalize

    # Reshape to (num_classes, num_templates, embed_dim) and average over templates
    text_embeds  = text_embeds.view(len(CLASS_NAMES), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    text_embeds  = text_embeds / text_embeds.norm(dim=-1, keepdim=True)   # re-normalize after avg

# ---------------------------------------------------------------------------
# 5. Load test split (torchvision CocoDetection not needed — using ImageFolder)
#    load_classification_split returns ImageFolder (PIL images + int labels)
# ---------------------------------------------------------------------------
test_dataset = load_classification_split(data_dir, "test")
class_to_idx = test_dataset.class_to_idx

idx_to_class = {v: k for k, v in class_to_idx.items()}
logger.info(f"Classes: {class_to_idx}")

# ---------------------------------------------------------------------------
# 6. Zero-shot evaluation loop
# ---------------------------------------------------------------------------
correct   = 0
total     = 0
batch_size = 32

logger.info("Running zero-shot evaluation…")

for start in range(0, len(test_dataset), batch_size):
    end   = min(start + batch_size, len(test_dataset))
    batch_images = [test_dataset[i][0] for i in range(start, end)]
    batch_labels = [test_dataset[i][1] for i in range(start, end)]

    with torch.no_grad():
        image_inputs = processor(
            images=[img.convert("RGB") for img in batch_images],
            return_tensors="pt",
            padding=True,
        ).to(CURRENT_DEVICE)

        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity: (batch, num_classes)
        similarity = image_embeds @ text_embeds.T
        predictions = similarity.argmax(dim=-1).cpu().tolist()

    for pred, label in zip(predictions, batch_labels):
        predicted_class_name = CLASS_NAMES[pred]
        true_class_name      = idx_to_class[label]
        if predicted_class_name == true_class_name:
            correct += 1
        total += 1

    if (start // batch_size) % 5 == 0:
        logger.info(f"  Processed {total}/{len(test_dataset)} images")

accuracy = correct / total
logger.info(f"\nZero-Shot Accuracy on test set: {accuracy:.4f} ({correct}/{total})")
logger.info("Note: This accuracy is achieved with ZERO labeled training examples.")
