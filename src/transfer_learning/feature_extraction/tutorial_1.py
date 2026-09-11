"""
Feature Extraction Tutorial using Local ResNet50 Model
This tutorial demonstrates how to use a pre-trained ResNet50 model for feature extraction
using the Hugging Face Transformers library (no manual PyTorch training loops).
"""

import os

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


class ImageClassificationDataset:
    """Dataset class using Hugging Face Datasets library"""

    def __init__(self, image_dir, processor, label_to_id=None):
        self.image_dir = image_dir
        self.processor = processor
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        if os.path.exists(image_dir):
            for class_name in sorted(os.listdir(image_dir)):
                class_path = os.path.join(image_dir, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.image_paths.append(os.path.join(class_path, img_name))
                            self.labels.append(class_name)

        # Create label mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_id = (
            {label: idx for idx, label in enumerate(unique_labels)}
            if label_to_id is None
            else label_to_id
        )
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        print(
            f"Found {len(self.image_paths)} images across {len(unique_labels)} classes"
        )
        print(f"Classes: {unique_labels}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        # Process image for the model
        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": self.label_to_id[label],
        }

    def to_hf_dataset(self):
        """Convert to Hugging Face Dataset format"""
        data = {"pixel_values": [], "labels": []}

        for idx in range(len(self)):
            item = self[idx]
            data["pixel_values"].append(item["pixel_values"].numpy())
            data["labels"].append(item["labels"])

        return HFDataset.from_dict(data)


def load_model_for_feature_extraction(model_path, num_classes=None):
    """
    Load pre-trained ResNet50 model for feature extraction

    Args:
        model_path: Path to the local model directory
        num_classes: Number of output classes (None keeps original)

    Returns:
        model: Modified model for feature extraction
        processor: Image processor for preprocessing
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    print(f"Loading model from: {model_path}")

    # Load the pre-trained model and processor
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)

    print(f"Original model architecture: {model}")

    # Freeze all layers for feature extraction
    for param in model.parameters():
        param.requires_grad = False

    print("All layers frozen for feature extraction")

    # Create label mappings
    if num_classes is not None:
        label2id = {f"class_{i}": i for i in range(num_classes)}
        id2label = {i: f"class_{i}" for i in range(num_classes)}

        # Configure model for new number of classes
        model.num_labels = num_classes
        model.label2id = label2id
        model.id2label = id2label

        # Replace the classifier head
        if hasattr(model, "classifier"):
            # Handle Sequential classifier (Flatten + Linear)
            if isinstance(model.classifier, torch.nn.Sequential):
                # The Linear layer is typically the last layer in the Sequential
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)

        print(f"Classifier replaced with {num_classes} classes")
    else:
        # Use original label mappings
        label2id = model.config.id2label
        id2label = model.config.label2id
        if isinstance(label2id, dict):
            label2id = {v: k for k, v in id2label.items()}

    return model, processor, label2id, id2label


def extract_features(model, processor, image_path):
    """
    Extract features from a single image

    Args:
        model: The feature extraction model
        processor: Image processor
        image_path: Path to the image

    Returns:
        features: Extracted feature vector
    """
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Extract features (forward pass without classifier)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[-1]  # Last hidden layer

    return features


def main():
    """Main function to demonstrate feature extraction using Transformers Trainer"""

    # Path to your local ResNet50 model
    MODEL_PATH = "/Users/abhishek/PycharmProjects/cv-learning/models/resnet50"

    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        return

    # Example: 10 classes (modify based on your dataset)
    NUM_CLASSES = 10

    # Load model for feature extraction
    model, processor, label2id, id2label = load_model_for_feature_extraction(
        MODEL_PATH, num_classes=NUM_CLASSES
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    # Example: Create dataset and train using Transformers Trainer
    # Uncomment and modify with your dataset path
    """
    train_dataset = ImageClassificationDataset("path/to/train/data", processor)
    train_hf_dataset = train_dataset.to_hf_dataset()

    eval_dataset = ImageClassificationDataset(
        "path/to/val/data", processor, label_to_id=train_dataset.label_to_id
    )
    eval_hf_dataset = eval_dataset.to_hf_dataset()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=0.001,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=eval_hf_dataset,
        data_collator=DefaultDataCollator(),
        tokenizer=processor,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    """

    print("\nFeature extraction setup complete!")
    print("To use this with your own data:")
    print("1. Prepare your dataset in a folder structure (class_name/image_files)")
    print("2. Update the NUM_CLASSES variable")
    print("3. Uncomment the dataset loading code")
    print("4. Run the trainer")


if __name__ == "__main__":
    main()
