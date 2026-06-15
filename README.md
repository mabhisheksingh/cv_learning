# Dog Detection ML Framework

A PyTorch-based machine learning framework for dog detection with support for multiple feature extraction methods and training approaches.

## Features

- **Multiple Feature Extraction Methods**:
  - HOG (Histogram of Oriented Gradients)
  - SIFT (Scale-Invariant Feature Transform)
  - CNN-based features (using pretrained models)

- **Multiple Training Approaches**:
  - Custom CNN architectures
  - Pretrained model fine-tuning (ResNet, EfficientNet, VGG, MobileNet)
  - Traditional ML classifiers (SVM, Random Forest)
  - Hybrid approaches (CNN features + SVM)

- **Training Utilities**:
  - Flexible configuration management
  - Checkpoint saving and loading
  - Training logging
  - Fine-tuning strategies (gradual unfreeze, differential learning rates)

## Project Structure

```
cv-learning/
├── data/                       # Dataset (gitignored)
│   └── raw/                    # Raw dataset images
│       ├── train/              # Training data (cats, dogs)
│       └── test/               # Test data (cats, dogs)
├── models/                     # Downloaded models (gitignored)
│   └── resnet50/               # Local ResNet50 model
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   └── transforms.py      # PyTorch transforms/augmentation
│   ├── training/
│   │   ├── feature_extraction/ # Feature extraction training
│   │   │   ├── transfer_learning.py # Transfer learning model
│   │   │   ├── train_classifier.py  # Train SVM on CNN features
│   │   │   └── traditional_trainer.py  # Traditional ML training
│   │   └── fine_tuning/        # Fine-tuning and testing
│   │       ├── train_transfer_learning.py  # Fine-tune ResNet50
│   │       ├── fine_tuning.py  # Fine-tuning pretrained models
│   │       ├── trainer.py      # PyTorch training loop
│   │       └── test_model.py   # Test model on images
│   ├── models/
│   │   ├── custom_cnn.py       # Custom CNN architecture
│   │   ├── pretrained.py       # Pretrained models
│   │   └── traditional.py      # SVM, Random Forest wrappers
│   ├── configs/                # Configuration files
│   │   ├── dataset.yaml        # Dataset configuration
│   │   ├── feature_extraction.yaml  # Feature extraction config
│   │   ├── fine_tuning.yaml    # Fine-tuning config
│   │   └── config.yaml         # General configuration
│   └── utils/                  # All utilities
│       ├── config.py           # Configuration management
│       ├── metrics.py          # Evaluation metrics
│       └── logger.py           # Training logger
├── logs/                       # Training logs (gitignored)
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv sync
```

## Download Required Data and Models

### 1. Download Dataset from Kaggle

Install Kaggle API:
```bash
pip install kaggle
```

Download the Cats vs Dogs dataset:
```bash
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
unzip cats-and-dogs-image-classification.zip -d data/raw/
```

Or manually download and extract to `data/raw/` with this structure:
```
data/raw/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

### 2. Download Pretrained Models from Hugging Face

Install Hugging Face CLI:
```bash
pip install huggingface_hub
```

Download models to `models/` folder:
```bash
# ResNet50
hf download microsoft/resnet-50 --local-dir ./models/resnet50

# EfficientNet-B0
hf download google/efficientnet-b0 --local-dir ./models/efficientnet_b0

# Swin Transformer
hf download microsoft/swin-tiny-patch4-window7-224 --local-dir ./models/swin_tiny

# MobileNet V2
hf download google/mobilenet_v2_1.0_224 --local-dir ./models/mobilenet_v2
```

**Note**: Models will be automatically downloaded by PyTorch on first use if not manually downloaded.

## Usage

### Training

**Transfer Learning (Fine-tune ResNet50):**
```bash
uv run transfer_learning
```

**Feature Extraction + SVM Classifier:**
```bash
uv run feature_extraction
```

**Test Model:**
```bash
uv run test_model
```

### Configuration

Edit `configs/config.yaml` to customize training parameters:

```yaml
data:
  image_size: 224
  batch_size: 32
  num_workers: 4

model:
  type: custom_cnn
  num_classes: 2
  dropout: 0.5

training:
  num_epochs: 50
  learning_rate: 0.001
```

### Data Preparation

The dataset should be organized in `data/raw/` with the following structure:

```
data/raw/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Update the data loading code in `main.py` to load your images from these folders.

## Supported Models

### Custom CNN
- `CustomCNN`: 4-layer CNN with batch normalization
- `SimpleCNN`: Lightweight 3-layer CNN

### Pretrained Models
- ResNet (18, 50, 101)
- EfficientNet (B0, B3)
- VGG16
- MobileNet V2
- DenseNet121

### Traditional Classifiers
- SVM (with RBF kernel)
- Random Forest
- Logistic Regression

## Fine-Tuning Strategies

The framework supports multiple fine-tuning strategies:

1. **Freeze Backbone**: Train only the classifier head
2. **Gradual Unfreeze**: Unfreeze layers in stages
3. **Differential Learning Rates**: Different LRs for different layer groups

See `src/training/fine_tuning.py` for implementation details.

## License

MIT License
