"""Shared utilities for the cv-learning project.

This module provides thin helpers for:

* Device selection (CUDA, MPS, XPU, CPU)
* Loading COCO-style splits via :class:`torchvision.datasets.CocoDetection`
* Downloading datasets from Roboflow
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from roboflow import Roboflow
from torchvision.datasets import CocoDetection, ImageFolder

logger = logging.getLogger(__name__)

# NOTE: Prefer setting ROBOFLOW_API_KEY as an environment variable.
# The hard-coded fallback is kept only for local convenience/backwards compatibility.
_ROBOFLOW_API_KEY: str = os.environ.get(
    "ROBOFLOW_API_KEY", "s2TP4YPS3Wj4VUUUTgsL"
)
if "ROBOFLOW_API_KEY" not in os.environ:
    logger.warning(
        "ROBOFLOW_API_KEY not found in environment; using built-in fallback key. "
        "Set ROBOFLOW_API_KEY to keep secrets out of version control."
    )

__all__ = ["get_device", "load_coco_split", "load_roboflow_dataset","load_classification_split","load_classification_split_with_hf_dataset"]


def get_device() -> str:
    """Return the best available PyTorch device as a string.

    Priority order:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon GPU)
        3. XPU (Intel GPU)
        4. CPU

    Returns:
        Device string compatible with :meth:`torch.nn.Module.to`,
        e.g. ``"cuda"``, ``"mps"``, ``"cpu"``.
    """
    logger.info("Checking for available device...")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("MPS backend is available and built.")
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    return device


def load_coco_split(data_dir: str | Path, split: str) -> CocoDetection:
    """Load a single COCO split from disk.

    Expects the split directory to contain ``_annotations.coco.json`` alongside
    the images, which is the default layout produced by Roboflow's COCO export.

    Args:
        data_dir: Root directory that contains the split folders (``train``,
            ``valid``, ``test``).
        split: Name of the split to load, e.g. ``"train"``, ``"valid"``,
            ``"test"``.

    Returns:
        A :class:`torchvision.datasets.CocoDetection` instance for the requested
        split.

    Raises:
        FileNotFoundError: If the split directory or its annotation file does
            not exist.
    """
    root = Path(data_dir) / split
    ann_file = root / "_annotations.coco.json"

    if not root.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {root.resolve()}"
        )
    if not ann_file.is_file():
        raise FileNotFoundError(
            f"COCO annotation file not found: {ann_file.resolve()}"
        )

    dataset = CocoDetection(root=str(root), annFile=str(ann_file))
    logger.info(f"Loaded {len(dataset)} images from '{split}' split")
    return dataset


def load_roboflow_dataset(
    *,
    download_path: str | Path,
    project_id: str,
    workspace: str,
    version: int = 1,
    model_format: str = "coco",
    enable_fresh_download: bool = False,
) -> tuple[bool, str]:
    """Download a dataset from Roboflow if it is not already present locally.

    Args:
        download_path: Directory where the dataset should be saved.
        project_id: Roboflow project identifier, e.g. ``"coco"``.
        workspace: Roboflow workspace/owner name, e.g. ``"microsoft"``.
        version: Dataset version number to download. Defaults to ``1``.
        model_format: Export format requested from Roboflow. Defaults to
            ``"coco"``.
        enable_fresh_download: If ``True`` and ``download_path`` exists but is
            empty, remove it and re-download. Useful for recovering from a
            partial/corrupt download.

    Returns:
        A tuple ``(success, message)``. ``success`` is ``True`` when the
        dataset is present on disk (either pre-existing or freshly downloaded),
        and ``False`` when the download failed. ``message`` describes the
        outcome.
    """
    download_path = Path(download_path)

    try:
        already_present = download_path.is_dir() and bool(
            any(download_path.iterdir())
        )
        if already_present and not enable_fresh_download:
            logger.info(
                f"Dataset already present at {download_path}; skipping download."
            )
            return True, "exists"

        # Roboflow refuses to download into an existing directory, even an empty one.
        if (
            enable_fresh_download
            and download_path.is_dir()
            and not any(download_path.iterdir())
        ):
            logger.warning(
                f"{download_path} exists but is empty; removing it to force a fresh download."
            )
            download_path.rmdir()

        rf = Roboflow(api_key=_ROBOFLOW_API_KEY)
        project = rf.workspace(workspace).project(project_id)
        # # 1. Define your augmentations via JSON configuration
        # augmentation_config = {
        #     "rotation": {"degrees": 15},
        #     "flip": {"horizontal": True},
        #     "brightness": {"percent": 15}
        # }
        # new_version_num = project.generate_version(
        #     settings = {"preprocessing": {}, "augmentation": augmentation_config}
        # )
        #
        # version = new_version_num if enable_fresh_download else version

        project.version(version).download(
            model_format=model_format, location=str(download_path)
        )

        if not download_path.is_dir() or not any(download_path.iterdir()):
            return False, "download returned but target directory is empty"

        logger.info(f"Dataset downloaded successfully to {download_path}")
        return True, "downloaded"

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load Roboflow dataset: {exc}")
        return False, str(exc)

def load_classification_split(data_dir: str | Path, split: str, transform=None) -> ImageFolder:
    """Load a single classification split from disk using the ImageFolder layout with torchvision dataset.

    Expects the split directory to contain subfolders named after each class
    (e.g., 'happy', 'sad', 'neutral'), which is the default layout produced
    by Roboflow's Folder Structure export.

    Args:
        data_dir: Root directory that contains the split folders (``train``,
            ``valid``, ``test``).
        split: Name of the split to load, e.g. ``"train"``, ``"valid"``,
            ``"test"``.
        transform: A function/transform that takes in an PIL image and returns
            a transformed version (e.g., MobileNetv3 transforms).

    Returns:
        A :class:`torchvision.datasets.ImageFolder` instance for the requested
        split.

    Raises:
        FileNotFoundError: If the split directory does not exist.
    """
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {root.resolve()}"
        )
    # ImageFolder automatically looks inside the directory, treats subfolders
    # as class names, and indexes all images inside them.
    dataset = ImageFolder(root=str(root), transform=transform)

    logger.info(f"Loaded {len(dataset)} images across {len(dataset.classes)} "
            f"classes from '{split}' split"
    )
    return dataset

def load_classification_split_with_hf_dataset(
    data_dir: str | Path,
    split: str,
    transform=None,
) -> Dataset:
    """Load a single classification split from disk using the HF imagefolder builder.

    The local convention ``valid`` is mapped to HF's ``validation`` split name
    because the ``imagefolder`` builder only recognises ``train``/``validation``/``test``.
    """
    root = Path(data_dir)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir.resolve()}")

    # HF imagefolder builder expects "validation", not "valid".
    hf_split = "validation" if split == "valid" else split

    ds = load_dataset(
        "imagefolder",
        data_dir=str(root),
        split=hf_split,
        trust_remote_code=False,
    )

    if transform is not None:
        ds.set_transform(transform)

    logger.info(f"Loaded {len(ds)} images from '{split}' split via HuggingFace imagefolder")
    return ds
