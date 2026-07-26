"""
Transfer Learning Training Script
Fine-tune ResNet50 for cat vs dog classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import models
from torchvision.models import ResNet50_Weights


class CatDogDataset(Dataset):
    """Dataset for cat vs dog classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root data directory
            split: 'train' or 'test'
            transform: Image transformations
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Load dogs (label=1)
        dog_dir = self.data_dir / 'dogs'
        for img_path in dog_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(1)
        
        # Load cats (label=0)
        cat_dir = self.data_dir / 'cats'
        for img_path in cat_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history saved to: training_history.png")
    plt.close()


def main():
    """Main training function."""
    print("="*60)
    print("TRANSFER LEARNING: CAT VS DOG CLASSIFICATION")
    print("="*60)
    
    # Configuration
    data_dir = './data/raw'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 2
    freeze_backbone = True
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CatDogDataset(data_dir, split='train', transform=train_transform)
    val_dataset = CatDogDataset(data_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load model
    print("\nLoading model...")
    print("Using torchvision pretrained weights")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze backbone if specified
    if freeze_backbone:
        print("Freezing backbone layers")
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze final layer
        for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Backbone frozen: {freeze_backbone}")
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val acc: {val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("\nGenerated files:")
    print("  - best_model.pth")
    print("  - training_history.png")


if __name__ == '__main__':
    main()
