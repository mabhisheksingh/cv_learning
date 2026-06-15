"""
Shared dataset loading utilities.
"""

from pathlib import Path
import cv2


def load_images_from_folder(folder_path: str, label: int):
    """Load images from a folder and assign label."""
    images = []
    labels = []

    folder = Path(folder_path)
    for img_path in folder.glob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            labels.append(label)

    return images, labels
