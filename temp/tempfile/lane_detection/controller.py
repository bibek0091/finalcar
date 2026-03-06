import math
import numpy as np
from dataclasses import dataclass

@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm:       float
    target_x:        float
    anchor:          str

class StanleyController:
    def __init__(self, k: float = 1.2, ks: float = 0.2, wheelbase_m: float = 0.23):
        self.k  = k
        self.ks = ks
        self.L  = wheelbase_m

    def compute(self, target_x_px: float, heading_rad: float, velocity_ms: float, lane_width_px: float):
        ppm  = max(lane_width_px, 50) / 0.35    # pixels per metre
        ce_m = (320.0 - target_x_px) / ppm      # cross-track error (metres)
        k_eff = self.k * min(1.0, velocity_ms / 0.25)
        reactive_rad = heading_rad + math.atan2(k_eff * ce_m, velocity_ms + self.ks)
        return math.degrees(reactive_rad)

class DividerGuard:
    DIVIDER_SAFE_PX = 130   
    EDGE_SAFE_PX    = 100   
    GAIN            = 0.35
    MAX_CORR        = 25.0
    DEADBAND_PX     =  2

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale = 0.0, 1.0
        div_corr = edge_corr = 0.0

        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))

        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale

class Controller:
    MAX_STEER      = 25.0
    MAX_STEER_RATE = 40.0

    def __init__(self):
        self.prev_steer = 0.0
        self.guard      = DividerGuard()
        self.stanley    = StanleyController(k=2.5, ks=0.1, wheelbase_m=0.23)

    def compute(self, lane_result, velocity_ms: float = 0.0, base_speed: float = 50.0) -> ControlOutput:
        
        # 1. Stanley Steering 
        # (Assuming no heading logic implemented in detector yet, we will fallback to 0.0 heading error)
        heading_rad = getattr(lane_result, 'heading_rad', 0.0)
        raw_steer = self.stanley.compute(
            lane_result.target_x, heading_rad,
            velocity_ms, lane_result.lane_width_px)

        # 2. Hardware Rate Limiting
        rate_delta  = max(-self.MAX_STEER_RATE,
                          min(self.MAX_STEER_RATE, raw_steer - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        self.prev_steer = steer_angle

        # 3. Divider Guard
        steer_guarded, guard_spd_mult = self.guard.apply(
            steer_angle, lane_result.sl, lane_result.sr, y_eval=lane_result.y_eval)
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_guarded))

        # 4. Final Speed
        speed = float(base_speed)
        
        if "DEAD_RECKONING" in lane_result.anchor:
            try:
                dr_conf = float(lane_result.anchor.split("_")[2])
            except Exception:
                dr_conf = 0.5
            speed *= (0.4 + 0.4 * dr_conf)

        if lane_result.anchor == "DIVIDER_FOLLOW":
            speed *= 0.75

        final_speed = speed * guard_spd_mult

        MINIMUM_DRIVE_PWM = 18.0
        if final_speed > 0:
            final_speed = max(final_speed, MINIMUM_DRIVE_PWM)

        return ControlOutput(
            steer_angle_deg = steer_angle,
            speed_pwm       = final_speed,
            target_x        = lane_result.target_x,
            anchor          = lane_result.anchor,
        )
