import logging
import os

import torch
from roboflow import Roboflow
from torchvision.datasets import CocoDetection

logger = logging.getLogger(__name__)
ROBOFLOW_API_KEY: str = "s2TP4YPS3Wj4VUUUTgsL"


def get_device() -> str:
    """
    Check for available device (CPU, CUDA, MPS, or XPU) and return the device type.
    :return: Device type as a string.
    """
    logger.info("Checking for available device...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    elif torch.xpu.is_available():
        device = "xpu"
    logger.info(f"Using device: {device}")
    return device


def load_coco_split(data_dir: str, split: str) -> CocoDetection:
    split_dir = os.path.join(data_dir, split)
    ann_file = os.path.join(split_dir, "_annotations.coco.json")
    dataset = CocoDetection(root=split_dir, annFile=ann_file)
    logger.info(f"Loaded {len(dataset)} of {len(dataset)} images")
    return dataset


def load_roboflow_dataset(
    *,
    download_path: str,
    project_id: str,
    workspace: str,
    enable_fresh_download: bool = False,
) -> tuple[bool, str]:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(workspace).project(project_id)

        already_present = os.path.exists(download_path) and bool(
            os.listdir(download_path)
        )
        if already_present and not enable_fresh_download:
            logger.info(
                f"Dataset already present at {download_path}; skipping download."
            )
            return True, "exists"

        # Roboflow skips downloading if the target directory already exists, even if it is empty.
        if (
            enable_fresh_download
            and os.path.exists(download_path)
            and not os.listdir(download_path)
        ):
            logger.warning(
                f"{download_path} exists but is empty; removing it to force a fresh download."
            )
            os.rmdir(download_path)

        project.version(50).download(model_format="coco", location=download_path)

        if not (os.path.exists(download_path) and os.listdir(download_path)):
            return False, "download returned but target directory is empty"

        return True, "downloaded"
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load Roboflow dataset: {e}")
        return False, str(e)
