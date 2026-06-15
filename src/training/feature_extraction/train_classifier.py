"""
Train a classifier on CNN features for cat vs dog classification.
Uses ResNet50 as feature extractor and SVM as classifier.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models
from torchvision.models import ResNet50_Weights
from src.utils.data_loader import load_images_from_folder


def load_cat_dog_dataset(data_dir: str):
    """Load entire dataset from train and test folders."""
    print("Loading dataset...")

    # Load training data
    train_dogs, train_dog_labels = load_images_from_folder(
        f"{data_dir}/train/dogs", label=1
    )
    train_cats, train_cat_labels = load_images_from_folder(
        f"{data_dir}/train/cats", label=0
    )

    # Load test data
    test_dogs, test_dog_labels = load_images_from_folder(
        f"{data_dir}/test/dogs", label=1
    )
    test_cats, test_cat_labels = load_images_from_folder(
        f"{data_dir}/test/cats", label=0
    )

    # Combine
    train_images = train_dogs + train_cats
    train_labels = train_dog_labels + train_cat_labels

    test_images = test_dogs + test_cats
    test_labels = test_dog_labels + test_cat_labels

    print(f"Training set: {len(train_images)} images ({len(train_dogs)} dogs, {len(train_cats)} cats)")
    print(f"Test set: {len(test_images)} images ({len(test_dogs)} dogs, {len(test_cats)} cats)")

    return train_images, train_labels, test_images, test_labels


def prepare_image_for_cnn(img: np.ndarray):
    """Prepare image for CNN feature extraction."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb)
    return tensor


def extract_features_batch(images, backbone, device, batch_size=32):
    """Extract features from images in batches using backbone."""
    import torch.nn.functional as F
    print(f"Extracting features from {len(images)} images...")
    
    features_list = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Prepare batch
        tensors = []
        for img in batch_images:
            tensor = prepare_image_for_cnn(img)
            tensors.append(tensor)
        
        batch = torch.stack(tensors).to(device)
        
        # Extract features
        with torch.no_grad():
            features = backbone(batch)
            # Global average pooling if needed
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
            # Normalize
            features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-7)
        
        features_list.append(features.cpu().numpy())
        
        if (i + batch_size) % 100 == 0:
            print(f"  Processed {i + batch_size}/{len(images)} images")
    
    return np.vstack(features_list)


def train_classifier(X_train, y_train, classifier_type='svm'):
    """Train a classifier on extracted features."""
    print(f"\nTraining {classifier_type.upper()} classifier...")
    
    if classifier_type == 'svm':
        classifier = SVC(kernel='rbf', C=10.0, gamma='scale')
    elif classifier_type == 'logistic':
        classifier = LogisticRegression(max_iter=1000, C=1.0)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    classifier.fit(X_train, y_train)
    print(f"Classifier trained successfully")
    
    return classifier


def evaluate_classifier(classifier, X_test, y_test):
    """Evaluate classifier on test set."""
    print("\nEvaluating classifier...")
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, cm, y_pred


def plot_confusion_matrix(cm, class_names=['Cat', 'Dog']):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\nConfusion matrix saved to: confusion_matrix.png")
    plt.close()


def save_classifier(classifier, backbone, feature_dim, filename='classifier.pkl'):
    """Save trained classifier and feature extractor backbone."""
    model_data = {
        'classifier': classifier,
        'backbone_state_dict': backbone.state_dict(),
        'feature_dim': feature_dim
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {filename}")


def main():
    """Main training function."""
    print("="*60)
    print("CAT VS DOG CLASSIFICATION USING CNN FEATURES")
    print("="*60)
    
    # Load dataset
    data_dir = './data/raw'
    train_images, train_labels, test_images, test_labels = load_cat_dog_dataset(data_dir)
    
    # Initialize feature extractor
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load ResNet50 with torchvision pretrained weights
    print("Using torchvision pretrained weights")
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Get backbone (remove final FC layer)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = backbone.to(device)
    backbone.eval()
    
    # Extract features
    print("\nExtracting training features...")
    X_train = extract_features_batch(train_images, backbone, device, batch_size=32)
    y_train = np.array(train_labels)
    
    print("\nExtracting test features...")
    X_test = extract_features_batch(test_images, backbone, device, batch_size=32)
    y_test = np.array(test_labels)
    
    print(f"\nFeature shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Train classifier
    classifier = train_classifier(X_train, y_train, classifier_type='svm')
    
    # Evaluate
    accuracy, cm, y_pred = evaluate_classifier(classifier, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Save model
    save_classifier(classifier, backbone, X_train.shape[1])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - classifier.pkl")


if __name__ == '__main__':
    main()
