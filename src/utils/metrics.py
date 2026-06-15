import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import torch


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'macro', 'micro', 'weighted')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
):
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred, target_names=target_names))


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    num_classes: int
) -> float:
    """
    Calculate Intersection over Union (IoU) for segmentation.
    
    Args:
        pred_mask: Predicted mask tensor
        gt_mask: Ground truth mask tensor
        num_classes: Number of classes
        
    Returns:
        Mean IoU
    """
    ious = []
    
    for cls in range(num_classes):
        pred = (pred_mask == cls)
        gt = (gt_mask == cls)
        
        intersection = (pred & gt).sum().float()
        union = (pred | gt).sum().float()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0.0
