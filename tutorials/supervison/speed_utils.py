from __future__ import annotations
import logging
import math
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)
MAX_SMOOTHING_WINDOW = 5  # More frames for smoother speed (0.4s at 25fps)
MAX_REASONABLE_SPEED_KMPH = 300.0  # Reject speeds above this (adjust per use case)
TIME_TO_DELETE_TRACK_ID = 20 # seconds

class SpeedUtils:
    def __init__(self):
        logger.info("SpeedUtils initialized | smoothing_window=%d max_speed=%.1f_kmph",
                    MAX_SMOOTHING_WINDOW, MAX_REASONABLE_SPEED_KMPH)
        self.map = defaultdict(list)
        self.last_scene:dict[int,float] = {}

    def add(self,track_id:int,center:tuple[int,int],time:float)->None:
        is_new = track_id not in self.map
        self.map[track_id].append((center,time))
        self.last_scene[track_id] = time
        if is_new:
            logger.info("New track started | track_id=%s center=%s time=%.4f", track_id, center, time)
        logger.debug("Position added | track_id=%s center=%s time=%.4f history_len=%d",
                     track_id, center, time, len(self.map[track_id]))

    def cleanup_stale_tracks(self,current_time:float)->None:
        stale = [tid for tid,last in self.last_scene.items()
                 if current_time - last > TIME_TO_DELETE_TRACK_ID]
        for tid in stale:
            logger.info("Track expired | track_id=%s time=%.4f", tid, current_time)
            self.map.pop(tid)
            self.last_scene.pop(tid)

    def get(self,track_id:int)->list|None:
        if track_id not in self.map:
            logger.debug("Track not found | track_id=%s", track_id)
            return None
        return self.map[track_id]

    @staticmethod
    def validate_speed(speed: float) -> bool:
        """Validate if speed is within reasonable bounds."""
        return 0 <= speed <= MAX_REASONABLE_SPEED_KMPH

    def calculate_speed_in_kmph(self,track_id:int,center:tuple[int,int],curr_time:float,pixels_per_meter:float=20)->float:
        history = self.get(track_id)
        self.add(track_id,center,curr_time)
        if history is None or len(history) < 2:
            logger.debug("Insufficient history for speed calc | track_id=%s history_len=%d",
                         track_id, len(history) if history else 0)
            return 0.0
        if len(history) >= MAX_SMOOTHING_WINDOW:
            logger.debug("Trimming history | track_id=%s old_len=%d new_len=%d",
                         track_id, len(history), MAX_SMOOTHING_WINDOW)
            self.map[track_id] = history[-MAX_SMOOTHING_WINDOW:]
            history = self.map[track_id]

        oldest_center, oldest_time = history[0]
        latest_center, latest_time = history[-1]

        time_delta = latest_time - oldest_time
        if time_delta <= 0:
            logger.warning("Non-positive time delta | track_id=%s delta=%.6f oldest=%.4f latest=%.4f",
                           track_id, time_delta, oldest_time, latest_time)
            return 0.0

        total_pixel_distance = sum(math.dist(history[i][0], history[i+1][0]) for i in range(len(history)-1))

        pixel_distance_in_meters = total_pixel_distance / pixels_per_meter
        speed_in_kmph = (pixel_distance_in_meters / time_delta)*3.6

        if not self.validate_speed(speed_in_kmph):
            logger.warning("Speed out of bounds | track_id=%s speed=%.2f_kmph max=%.1f_kmph",
                          track_id, speed_in_kmph, MAX_REASONABLE_SPEED_KMPH)
            return 0.0

        logger.debug("Speed computed | track_id=%s speed=%.2f_kmph px_dist=%.1f m_dist=%.3f dt=%.4f ppm=%.1f",
                     track_id, speed_in_kmph, total_pixel_distance, pixel_distance_in_meters,
                     time_delta, pixels_per_meter)

        return round(speed_in_kmph,2)


    def get_bird_eye_view(self,frame,src_points,dst_size):
        # Source points: 4 corners of the road plane in original image
        src = np.array(src_points, dtype=np.float32)

        # Destination points: rectangle in output
        dst = np.array([
            [0, 0],
            [dst_size[0], 0],
            [dst_size[0], dst_size[1]],
            [0, dst_size[1]]
        ], dtype=np.float32)
        # Compute homography
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp image
        warped = cv2.warpPerspective(frame, M, dst_size)
        return warped, M

