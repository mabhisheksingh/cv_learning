# Speed Tracking with IPM Calibration

A production-grade vehicle speed tracking system using YOLO object detection, ByteTrack tracking, and Inverse Perspective Mapping (IPM) for accurate speed estimation.

## Features

- **Object Detection**: YOLOv8 for vehicle detection
- **Object Tracking**: ByteTrack for consistent vehicle tracking across frames
- **IPM Calibration**: Interactive perspective transform to eliminate camera angle distortion
- **Bird's Eye View**: Real-time top-down view for accurate distance measurement
- **Speed Calculation**: Smoothed speed estimation with validation
- **Track Lifecycle**: Automatic cleanup of stale tracks to prevent memory leaks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Mode (No Calibration)
```bash
python speed_tracking.py --source-path your_video.mp4 --pixels-per-meter 20
```

### Calibration Mode (Recommended for Accuracy)
```bash
python speed_tracking.py --source-path your_video.mp4 --calibrate
```

**Calibration Steps:**
1. First frame will pause
2. Click 4 points on the road plane in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
3. Press `c` to confirm, `r` to reset, `q` to cancel
4. Preview of bird's eye view will show for 2 seconds
5. Processing starts with perspective-transformed coordinates

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--source-path` | Yes | Path to the video file |
| `--pixels-per-meter` | No | Calibration parameter: pixels per meter (default: 20) |
| `--calibrate` | No | Enable interactive IPM calibration mode |

## Calibration Benefits

**Without IPM:**
- `pixels_per_meter` varies by depth (perspective distortion)
- Objects far away appear slower even at same speed
- ~60% accuracy

**With IPM:**
- Constant scale across entire scene
- Accurate distance measurement
- ~85% accuracy

## How to Calibrate `pixels_per_meter`

1. Run with `--calibrate`
2. In the bird's eye view preview, measure a known distance (e.g., lane marking = 3 meters)
3. Count pixels for that distance in the warped view
4. Calculate: `pixels_per_meter = pixel_distance / real_distance`
5. Run with `--pixels-per-meter <your_value>`

## Architecture

### `speed_tracking.py`
Main execution script for video processing.

### `speed_utils.py`
Speed calculation utilities with:
- Track history management
- Smoothing window (5 frames)
- Speed validation (max 300 km/h)
- Track lifecycle cleanup (20 second timeout)

## Performance Considerations

- **Memory**: Automatic track cleanup prevents memory leaks on long videos
- **Latency**: Optimized frame reading avoids double-reading
- **Accuracy**: IPM calibration eliminates perspective distortion

## Requirements

- Python 3.12+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Supervision
- ByteTrack

## Supported Video Formats

mp4, avi, mov, mkv, webm

## License

MIT
