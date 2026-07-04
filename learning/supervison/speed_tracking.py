from __future__ import annotations

import argparse
import logging
from typing import Generator
import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2
from trackers import ByteTrackTracker

from learning.supervison.speed_utils import SpeedUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress noisy library loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("supervision").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

supported_formats:list = ["mp4", "avi", "mov", "mkv", "webm"]

# ---------------------------------------------------------------------------
# Speed Calculation Utilities
# ---------------------------------------------------------------------------

def parse_arguments()-> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description= "Speed tracking using Supervision"
    )
    parser.add_argument(
        "--source-path",
        type=str,
        required=True,
        help="Path to the video file"
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="Calibration parameter: pixels per meter for speed calculation (optional)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration mode to select 4 points for perspective transform"
    )
    return parser.parse_args()

def get_video_frames_and_info(source_path:str):
    """Get video frames and info with error handling."""
    try:
        if not source_path:
            raise ValueError("Source path cannot be empty")

        if source_path.split('.')[-1].lower() not in supported_formats:
            raise ValueError(f"Not a valid format, only support {supported_formats}")

        frame_generator : Generator[np.ndarray, None, None] = sv.get_video_frames_generator(source_path=source_path)
        video_info = sv.VideoInfo.from_video_path(video_path=source_path)

        logger.info(f"Loaded video: {source_path}, Resolution: {video_info.resolution_wh}, FPS: {video_info.fps}")
        return frame_generator, video_info

    except Exception as e:
        logger.error(f"Error loading video {source_path}: {e}")
        raise

def get_polygone_zone_and_polygone(video_info:sv.VideoInfo):
    w,h = video_info.resolution_wh
    polygone = np.array([[0,h//2],[w,h//2],[w,h],[0,h]], dtype=np.int32)
    polyzone = sv.PolygonZone(polygon=polygone)
    return polygone, polyzone

def load_model(device="mps", model_path: str = "/Users/abhishek/PycharmProjects/cv-learning/learning/supervison/yolov8n.pt"):
    """Load YOLO model with error handling."""
    try:
        if not model_path:
            raise ValueError("Model path cannot be empty")

        model = YOLO(model_path)
        model.to(device)
        logger.info(f"Loaded model from {model_path} on device {device}")
        return model

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def calibrate_perspective(frame, dst_size=(1000, 1000)):
    """Interactive calibration: click 4 points to define road plane."""
    points = []
    temp_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 4:
                logger.info("Already have 4 points. Press 'c' to confirm or 'r' to reset.")
                return
            points.append([x, y])
            temp_frame = frame.copy()
            
            # Draw points and lines
            for i, pt in enumerate(points):
                color = (0, 255, 0) if i == len(points) - 1 else (0, 0, 255)
                cv2.circle(temp_frame, tuple(pt), 5, color, -1)
                cv2.putText(temp_frame, f"{i+1}", (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw lines between points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(temp_frame, tuple(points[i]), tuple(points[i+1]), (255, 255, 0), 2)
            
            cv2.imshow("Calibration - Click 4 points (TL, TR, BR, BL)", temp_frame)
            
            if len(points) == 4:
                logger.info("4 points selected. Press 'c' to confirm or 'r' to reset.")
    
    cv2.namedWindow("Calibration - Click 4 points (TL, TR, BR, BL)")
    cv2.setMouseCallback("Calibration - Click 4 points (TL, TR, BR, BL)", mouse_callback)
    
    cv2.imshow("Calibration - Click 4 points (TL, TR, BR, BL)", temp_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None, None
        elif key == ord('r'):
            points = []
            temp_frame = frame.copy()
            cv2.imshow("Calibration - Click 4 points (TL, TR, BR, BL)", temp_frame)
        elif key == ord('c') and len(points) == 4:
            cv2.destroyAllWindows()
            break
    
    # Compute homography
    src = np.array(points, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [dst_size[0], 0],
        [dst_size[0], dst_size[1]],
        [0, dst_size[1]]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Show preview
    warped = cv2.warpPerspective(frame, M, dst_size)
    cv2.imshow("Bird's Eye View Preview", warped)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    logger.info("Calibration complete. Homography matrix computed.")
    return M, dst_size

def executor(source_path:str, pixels_per_meter: float | None = None, calibrate: bool = False):
    """Main executor function with error handling and logging."""
    try:
        frames, video_info = get_video_frames_and_info(source_path=source_path)
        logger.info(f"Starting processing for {source_path}")
        model = load_model()

        # Calibration mode
        M = None
        dst_size = None
        if calibrate:
            # Get first frame separately for calibration (without consuming main generator)
            cap = cv2.VideoCapture(source_path)
            ret, first_frame = cap.read()
            cap.release()
            if not ret:
                logger.error("Failed to read first frame for calibration")
                return
            
            logger.info("Calibration mode enabled. Please select 4 points on the road plane.")
            logger.info("Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
            M, dst_size = calibrate_perspective(first_frame)
            if M is None:
                logger.warning("Calibration cancelled. Using normal mode.")
            else:
                logger.info("Using perspective transform for accurate speed calculation.")

        thickness = sv.calculate_optimal_line_thickness(resolution_wh= video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh= video_info.resolution_wh)
        box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.TRACK
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            color_lookup=sv.ColorLookup.TRACK,
            text_position=sv.Position.BOTTOM_CENTER,
            text_thickness=thickness,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
            trace_length= int(video_info.fps * 2),
        )

        byte_track =  ByteTrackTracker()
        polygone, polyzone = get_polygone_zone_and_polygone(video_info=video_info)
        speed_utils = SpeedUtils()
        frame_id = 0
        total_frames = 0
        
        logger.info("Starting frame processing...")
        
        for frame in frames:
            try:
                total_frames += 1
                if total_frames % 30 == 0:
                    logger.info("Progress | frame=%d fps=%.1f", total_frames, video_info.fps)
                    speed_utils.cleanup_stale_tracks(current_time=frame_id/video_info.fps)
                result = model.predict(source=frame,device="mps",conf=.5)[0]
                annotated_frame = frame.copy()
                detections = sv.Detections.from_ultralytics(result)

                detections = byte_track.update(detections=detections)
                detections = detections[polyzone.trigger(detections=detections)] ## apply polyzone filter

                # Filter out unassigned tracks (tracker_id=-1)
                mask = detections.tracker_id != -1
                detections = detections[mask]

                if len(detections) > 0 and detections.tracker_id is not None:
                    label = []
                    for i, (tracker_id, class_name, confidence) in enumerate(zip(detections.tracker_id,detections['class_name'], detections.confidence)):
                        try:
                            # Calculate center point of bounding box
                            bottom_center = detections[i].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                            logger.debug("Bottom center | track_id=%s bottom_center=%s", tracker_id, bottom_center)
                            
                            # Transform point using homography if calibration enabled
                            if M is not None:
                                point_2d = np.array([bottom_center], dtype=np.float32).reshape(-1, 1, 2)
                                transformed = cv2.perspectiveTransform(point_2d, M)[0][0]
                                center = (transformed[0], transformed[1])
                            else:
                                center = (bottom_center[0], bottom_center[1])
                            
                            # Calculate speed
                            velocity = speed_utils.calculate_speed_in_kmph(
                                track_id=tracker_id,
                                center=center,
                                curr_time=frame_id/video_info.fps,
                                pixels_per_meter=pixels_per_meter if pixels_per_meter else 20.0,
                            )

                            # Create label with speed only
                            label_text = f"{tracker_id}: {velocity} kmph"
                            label.append(label_text)

                        except Exception as e:
                            logger.error("Detection processing failed | track_id=%s frame=%d err=%s",
                                         tracker_id, total_frames, e, exc_info=True)
                            # Add fallback label to prevent mismatch
                            label.append(f"{tracker_id}: -- kmph")
                            continue
                    
                    annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=detections,labels=label)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame,detections=detections)

                #debug purpose only
                annotated_frame = trace_annotator.annotate(scene=annotated_frame,detections=detections)
                annotated_frame = sv.draw_polygon(scene=annotated_frame,polygon=polygone,color=sv.Color.RED,thickness=thickness)

                # Display bird's eye view if calibration enabled
                if M is not None:
                    warped = cv2.warpPerspective(annotated_frame, M, dst_size)
                    warped_resized = cv2.resize(warped, (video_info.resolution_wh[0] // 2, video_info.resolution_wh[1]))
                    combined = np.hstack([annotated_frame, warped_resized])
                    cv2.imshow("Original + Bird's Eye View", combined)
                else:
                    cv2.imshow("Frame", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested to quit")
                    break

                frame_id += 1
                
            except Exception as e:
                logger.error("Frame processing failed | frame=%d err=%s", total_frames, e, exc_info=True)
                continue
                
        cv2.destroyAllWindows()
        logger.info(f"Processing complete. Total frames processed: {total_frames}")
        
    except Exception as e:
        logger.error("Fatal error in executor | err=%s", e, exc_info=True)
        cv2.destroyAllWindows()
        raise

if __name__ == "__main__":
    logger.info("Starting speed tracking...")
    args = parse_arguments()
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        executor(args.source_path, args.pixels_per_meter, args.calibrate)
    except Exception as e:
        logger.error("Application failed | err=%s", e, exc_info=True)
        raise