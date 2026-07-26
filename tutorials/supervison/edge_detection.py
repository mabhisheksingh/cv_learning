import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ek Line Mein Summary
# Sobel: Best combination hai noise control aur edge detection ka.
# Prewitt: Sobel jaisa hi hai par thoda simple aur sasta padta hai.
# Roberts: Super fast hai par kachre (noise) se darr lagta hai isse.
# Canny: Sabse best aur khubsoorat edges deta hai kyunki yeh noise saaf karke, edges ko patla karke filter karta hai.

# Load image in grayscale - use existing image path
image_path = '/Users/abhishek/PycharmProjects/cv-learning/data/raw/test/cats/cat_88.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at path: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image. Please check the path.")
    exit(1)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# 1. Sobel Edge Detection
# Sobel uses Gaussian smoothing and differentiation to reduce noise sensitivity
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)

# 2. Prewitt Edge Detection
# Prewitt is similar to Sobel but uses simpler kernels
prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_x = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_kernel_x)
prewitt_y = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_kernel_y)
prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
prewitt_magnitude = np.uint8(prewitt_magnitude / prewitt_magnitude.max() * 255)

# 3. Roberts Edge Detection
# Roberts cross-gradient operator - very fast but sensitive to noise
roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_kernel_x)
roberts_y = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_kernel_y)
roberts_magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
roberts_magnitude = np.uint8(roberts_magnitude / roberts_magnitude.max() * 255)

# 4. Canny Edge Detection
# Multi-stage algorithm with noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding
canny_edges = cv2.Canny(blurred_image, 100, 200)

# Resize all images to the same height for horizontal stacking
target_height = 300
original_resized = cv2.resize(image, (int(image.shape[1] * target_height / image.shape[0]), target_height))
sobel_resized = cv2.resize(sobel_magnitude, (int(sobel_magnitude.shape[1] * target_height / sobel_magnitude.shape[0]), target_height))
prewitt_resized = cv2.resize(prewitt_magnitude, (int(prewitt_magnitude.shape[1] * target_height / prewitt_magnitude.shape[0]), target_height))
roberts_resized = cv2.resize(roberts_magnitude, (int(roberts_magnitude.shape[1] * target_height / roberts_magnitude.shape[0]), target_height))
canny_resized = cv2.resize(canny_edges, (int(canny_edges.shape[1] * target_height / canny_edges.shape[0]), target_height))

# Combine all images horizontally
combined_image = np.hstack([original_resized, sobel_resized, prewitt_resized, roberts_resized, canny_resized])

# Add labels to the combined image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined_image, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, 'Sobel', (original_resized.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, 'Prewitt', (original_resized.shape[1] + sobel_resized.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, 'Roberts', (original_resized.shape[1] + sobel_resized.shape[1] + prewitt_resized.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, 'Canny', (original_resized.shape[1] + sobel_resized.shape[1] + prewitt_resized.shape[1] + roberts_resized.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)

# Display the combined image
plt.figure(figsize=(20, 6))
plt.imshow(combined_image, cmap='gray')
plt.title('Edge Detection Comparison: Sobel | Prewitt | Roberts | Canny')
plt.axis('off')
plt.tight_layout()
plt.show()

# Optional: Save the combined image
cv2.imwrite('/Users/abhishek/PycharmProjects/cv-learning/tutorials/supervison/edge_detection_comparison.png', combined_image)
print("Combined image saved as edge_detection_comparison.png")
