from .preprocessing import preprocess_image
from .perspective_transform import PerspectiveTransformer
from .lane_tracker import HybridLaneTracker
from dataclasses import dataclass
import numpy as np

@dataclass
class LaneResult:
    warped_binary: np.ndarray
    lane_dbg: np.ndarray
    sl: object
    sr: object
    target_x: float
    anchor: str
    lane_width_px: float
    curvature: float
    y_eval: float = 400.0
    lateral_error_px: float = 0.0

class LaneDetector:
    def __init__(self):
        self.transformer = PerspectiveTransformer()
        self.tracker = HybridLaneTracker()
        self.lost_frames = 0
        self.last_target_x = 320.0
        self._target_ema = 320.0

    def process(self, frame, dt=0.033):
        # 1. Perspective Transform
        warped_colour = self.transformer.warp(frame)
        
        # 2. Preprocessing
        warped_binary = preprocess_image(warped_colour)
        
        # 3. Lane Tracking
        sl, sr, lane_dbg, mode_label = self.tracker.update(warped_binary)
        
        # 4. Target Calculation
        y_eval = 320.0
        lw = self.tracker.estimated_lane_width
        
        target_x, anchor = self.tracker.get_target_x(y_eval, lw)
        
        if target_x is None:
            self.lost_frames += 1
            self.tracker.dead_reckoner.accumulate(dt)
            target_x = self.last_target_x
        else:
            self.lost_frames = 0
            delta = abs(target_x - self._target_ema)
            alpha = 1.0 if delta > 5.0 else 0.8
            self._target_ema = (1.0 - alpha) * self._target_ema + alpha * target_x
            target_x = self._target_ema
            self.last_target_x = target_x
            
        curv = self.tracker.get_curvature(y_eval)
        lateral_error = target_x - 320.0

        return LaneResult(
            warped_binary=warped_binary,
            lane_dbg=lane_dbg,
            sl=sl,
            sr=sr,
            target_x=target_x,
            anchor=anchor,
            lane_width_px=lw,
            curvature=curv,
            y_eval=y_eval,
            lateral_error_px=lateral_error
        )
