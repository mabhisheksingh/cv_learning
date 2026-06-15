"""
Dog Detection ML Framework - Entry Point
"""

import sys
import os


def dev():
    """Development entry point - shows available training options."""
    print("="*60)
    print("DOG DETECTION ML FRAMEWORK")
    print("="*60)
    print("\nAvailable Training Options:")
    print("\n1. Transfer Learning (Fine-tune ResNet50):")
    print("   uv run python src/training/fine_tuning/train_transfer_learning.py")
    print("\n2. Feature Extraction + SVM Classifier:")
    print("   uv run python src/training/feature_extraction/train_classifier.py")
    print("\n3. Test Model:")
    print("   uv run python src/training/fine_tuning/test_model.py")
    print("\n" + "="*60)
    print("Configuration files in src/configs/:")
    print("  - dataset.yaml")
    print("  - feature_extraction.yaml")
    print("  - fine_tuning.yaml")
    print("="*60)


if __name__ == '__main__':
    dev()
