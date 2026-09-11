# Feature Extraction (Frozen Layers)

## Overview
Feature extraction is the simplest transfer learning approach where you freeze all pre-trained layers of a model and only train the final classification head. This preserves the pre-trained features while adapting the model to your specific task.

## When to Use
- Limited training data available
- Want to preserve pre-trained features
- Quick prototyping and baseline
- Computational resources are limited

## How It Works
1. Load a pre-trained model (e.g., ResNet, VGG)
2. Freeze all layers by setting `requires_grad=False`
3. Replace the final classification layer with your own
4. Train only the final layer on your dataset

## Advantages
- Fastest training time
- Less prone to overfitting
- Minimal computational resources needed
- Good starting point for transfer learning

## Disadvantages
- Limited adaptation to new task
- May not capture domain-specific features
- Performance ceiling compared to full fine-tuning

## Example Code Structure
```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Train only the final layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001)
```

## Resources
- PyTorch Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- TensorFlow Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning
