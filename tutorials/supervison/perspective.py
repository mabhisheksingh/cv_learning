import  cv2
import numpy as np
from pathlib import Path

image_paths = [
    # "88B888.png",
    "KL10AZ3739.png"
]

def load_image(path):
    image = cv2.imread(path)
    return image

def load_all_images()->list[np.ndarray]:
    images = []
    base_dir:Path = Path(__file__).parent

    for path in image_paths:
        path = base_dir / path
        images.append(load_image(path))
    return images

def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(point_sum)]  # top-left
    rect[2] = points[np.argmax(point_sum)]  # bottom-right
    rect[1] = points[np.argmin(point_diff)]  # top-right
    rect[3] = points[np.argmax(point_diff)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def select_points(image: np.ndarray, window_name: str = "Select 4 corners") -> np.ndarray:
    points: list[list[int]] = []
    preview = image.copy()

    def handle_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(window_name, preview)

    cv2.imshow(window_name, preview)
    cv2.setMouseCallback(window_name, handle_click)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) >= 4 or key == ord("q"):
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype="float32")


def detect_plate_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < image_area * 0.01:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 2.0 <= aspect_ratio <= 6.5:
            return approx.reshape(4, 2)

    return None


def enhance_plate(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def executor():
    images = load_all_images()
    
    if not images or images[0] is None:
        print("Error: Could not load images")
        return
    
    image = images[0]
    height, width = image.shape[:2]
    print(f"Image shape: {image.shape}")
    
    plate_points = detect_plate_corners(image)
    if plate_points is None:
        print("Automatic detection failed. Click 4 corners of the number plate, then press 'q'.")
        plate_points = select_points(image)

    if plate_points is None or plate_points.shape[0] != 4:
        print("Error: Please select exactly 4 points.")
        return

    transformed = four_point_transform(image, plate_points)
    transformed = cv2.resize(
        transformed,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )
    # enhanced = enhance_plate(transformed)
    
    # Display the original and transformed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Transformed Image", transformed)
    # cv2.imshow("Enhanced Plate", enhanced)
    
    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the transformed image
    output_path = Path(__file__).parent / "transformed_output.png"
    # cv2.imwrite(str(output_path), enhanced)
    print(f"Transformed image saved to: {output_path}")

if __name__ == "__main__":
    print("Perspective Transform")
    executor()