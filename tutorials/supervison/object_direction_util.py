from __future__ import annotations

import logging
from collections import deque, defaultdict
from enum import Enum
import numpy as np

# Suppress noisy library loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("supervision").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ValidDirections(Enum):
    """Valid direction types for object movement."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"

MAX_SMOOTHING_WINDOW: int = 5
MIN_DISPLACEMENT_PX: int = 3
TIME_TO_DELETE_TRACK_ID: int = 20

class ObjectDirectionUtil:
    def __init__(self):
        logger.info("ObjectDirectionUtil initialized")
        self.position_history = defaultdict(lambda: deque(maxlen=MAX_SMOOTHING_WINDOW))
        self.last_seen: dict[int, float] = {}
        self.last_direction: dict[int, ValidDirections] = {}

    def cleanup_stale_tracks(self, current_time: float) -> None:
        stale = [tid for tid, last in self.last_seen.items() if current_time - last > TIME_TO_DELETE_TRACK_ID]
        for tid in stale:
            self.position_history.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.last_direction.pop(tid, None)


    def get_direction(self, track_id: int, bottom_center: tuple[float, float], curr_video_time: float) -> ValidDirections:
        self.last_seen[track_id] = curr_video_time

        if len(self.position_history[track_id]) < MAX_SMOOTHING_WINDOW:
            logger.debug("Warming up track | track_id=%s center=%s time=%.4f", track_id, bottom_center, curr_video_time)
            self.position_history[track_id].append((bottom_center, curr_video_time))
            return ValidDirections.UNKNOWN

        avg_bottom_center = np.median([pos[0] for pos in self.position_history[track_id]], axis=0)
        dx = bottom_center[0] - avg_bottom_center[0]
        dy = bottom_center[1] - avg_bottom_center[1]
        logger.debug(
            "Direction analysis | track_id=%s current=%s avg=%s dx=%.4f dy=%.4f |dx|=%.4f |dy|=%.4f",
            track_id, bottom_center, avg_bottom_center, dx, dy, abs(dx), abs(dy)
        )
        self.position_history[track_id].append((bottom_center, curr_video_time))

        if abs(dx) < MIN_DISPLACEMENT_PX and abs(dy) < MIN_DISPLACEMENT_PX:
            direction = ValidDirections.STATIONARY
        elif abs(dx) > abs(dy):
            direction = ValidDirections.RIGHT if dx > 0 else ValidDirections.LEFT
        else:
            direction = ValidDirections.DOWN if dy > 0 else ValidDirections.UP

        if self.last_direction.get(track_id) != direction:
            logger.info("Direction changed | track_id=%s direction=%s", track_id, direction.value)
            self.last_direction[track_id] = direction

        logger.debug("Direction result | track_id=%s chosen=%s", track_id, direction.value)
        return direction
