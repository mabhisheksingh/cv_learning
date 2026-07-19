"""RF-DETR Object Detection with Supervision

This script demonstrates how to use the RF-DETR model from Hugging Face
with supervision for object detection and annotation.
"""

from transformers import AutoImageProcessor, RfDetrForObjectDetection
import torch
from PIL import Image
import requests
import supervision as sv
import numpy as np

# Load image from URL
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load model and processor from local path
processor = AutoImageProcessor.from_pretrained("/Users/abhishek/PycharmProjects/cv-learning/models/rf-detr-large")
model = RfDetrForObjectDetection.from_pretrained("/Users/abhishek/PycharmProjects/cv-learning/models/rf-detr-large")

# Run inference
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert raw outputs to supervision Detections format
target_sizes = torch.tensor([image.size[::-1]])  # (width, height)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.35)[0]

detections = sv.Detections.from_transformers(
    transformers_results=results,
    id2label=model.config.id2label,
)

# Print detection results
for i, (score, label_id, box) in enumerate(zip(detections.confidence, detections.class_id, detections.xyxy)):
    print(
        f"Detected {model.config.id2label[label_id]} with confidence "
        f"{round(score, 3)} at location {box.round(2)}"
    )

# Annotate and save image
image_np = np.array(image)  # Convert PIL to numpy array
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{model.config.id2label[class_id]} {confidence:0.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated_image = box_annotator.annotate(scene=image_np.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

with sv.ImageSink(target_dir_path=".") as sink:
    sink.save_image(image=annotated_image, image_name="rf_detr_result.jpg")
print("Annotated image saved to rf_detr_result.jpg")
