
from pathlib import Path
from accelerate import Accelerator
from transformers import RTDetrForObjectDetection

accelerate = Accelerator()
# Static
RAW_MODEL_NAME_DIR = Path(__file__).parent.parent.parent.parent / "models" / "rtdetr_r50vd"
DEVICE = accelerate.device
BASE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "VisDrone_Dataset"

#Load dataset
def load_dataset():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Dataset directory {BASE_DIR} does not exist")
    train_images = BASE_DIR / "VisDrone2019-DET-train" / "images"
    train_labels = BASE_DIR / "VisDrone2019-DET-train" / "labels"
    val_images = BASE_DIR / "VisDrone2019-DET-val" / "images"
    val_labels = BASE_DIR / "VisDrone2019-DET-val" / "labels"

    return train_images, train_labels, val_images, val_labels

def load_model():
    if not RAW_MODEL_NAME_DIR.exists():
        raise FileNotFoundError(f"Model directory {RAW_MODEL_NAME_DIR} does not exist")
    model = RTDetrForObjectDetection.from_pretrained(RAW_MODEL_NAME_DIR)
    return model




