import time
import os

from datasets import DownloadConfig, load_dataset

HF_TOKEN = os.environ.get("HF_TOKEN")
# Load the dataset from local cache (no internet needed)
start_time = time.time()
dataset = load_dataset(
    path="uoft-cs/cifar10",
    cache_dir="/home/abhishek/Desktop/Personal/cv_learning/data",
    token=HF_TOKEN,
    download_config=DownloadConfig(local_files_only=True),
)
end_time = time.time()
print(f"Time taken to load dataset: {end_time - start_time:.2f} seconds")
print(dataset)
print(f"Dataset info: {dataset}")

# View sample images
print("\nSample images from dataset:")
sample = dataset["train"][0]
print(f"Image shape: {sample['img'].size}")
print(f"Label: {sample['label']}")

# # Display first 5 images
# for i in range(5):
#     example = dataset["train"][i]
#     print(f"Image {i}: label={example['label']}, size={example['img'].size}")
#     example['img'].show()  # Opens image in default image viewer
