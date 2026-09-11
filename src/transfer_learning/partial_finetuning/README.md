# Partial Fine-tuning

## Overview
Partial fine-tuning is a middle-ground approach where you unfreeze only specific layers of a pre-trained model (typically the last few layers) while keeping earlier layers frozen. This balances adaptation efficiency with computational cost.

## When to Use
- Moderate amount of training data
- Want more adaptation than feature extraction
- Limited computational resources
- Domain is somewhat similar to pre-training data
- Need faster training than full fine-tuning

## How It Works
1. Load a pre-trained model
2. Freeze early layers (feature extractors)
3. Unfreeze later layers (high-level features)
4. Replace the final classification layer
5. Train only unfrozen layers

## Common Strategies
- **Last N layers**: Unfreeze last 1-3 layers
- **Block-based**: Unfreeze last block(s) in ResNet/VGG
- **Layer-by-layer**: Gradually unfreeze during training
- **Differential learning rates**: Lower LR for early layers

## Advantages
- Good balance between performance and efficiency
- Faster than full fine-tuning
- Less risk of overfitting than full fine-tuning
- Adapts high-level features to new task
- Preserves low-level feature extractors

## Disadvantages
- More complex to implement than feature extraction
- Requires deciding which layers to unfreeze
- May not adapt as well as full fine-tuning
- Some hyperparameter tuning needed

## Example Code Structure
```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)

# Freeze early layers, unfreeze last 2 layers
layers_to_freeze = list(model.children())[:-2]
for layer in layers_to_freeze:
    for param in layer.parameters():
        param.requires_grad = False

# Train unfrozen layers
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
)
```

## Best Practices
- Start with last 1-2 layers, unfreeze more if needed
- Use lower learning rates for earlier unfrozen layers
- Monitor validation performance to decide on unfreezing strategy
- Consider gradual unfreezing during training

## Resources
- PyTorch Layer Freezing: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- TensorFlow Fine-tuning: https://www.tensorflow.org/tutorials/images/transfer_learning
