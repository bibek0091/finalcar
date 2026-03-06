import cv2
import numpy as np
import math
import sys
import warnings
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PerceptionResult:
    warped_binary: np.ndarray
    lane_dbg: np.ndarray
    sl: object
    sr: object
    target_x: float
    anchor: str
    confidence: float
    lane_width_px: float
    curvature: float
    heading_rad: float
    y_eval: float = 400.0

@dataclass
class ControlOutput:
    steer_angle_deg: float
    speed_pwm: float
    target_x: float
    anchor: str

# ══════════════════════════════════════════════════════════════════════════════
# PERCEPTION: PURE BEV & LANE TRACKING
# ══════════════════════════════════════════════════════════════════════════════
class HybridLaneTracker:
    NWINDOWS = 9
    SW_MARGIN = 60
    MINPIX = 50
    MIN_PIX_OK = 200
    EMA_ALPHA = 0.55
    STALE_FIT_FRAMES = 12

    WIDE_ROAD_PX = 420
    RIGHT_LANE_BIAS_PX = 0
    DIVIDER_FOLLOW_OFFSET_PX = 90

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode = "SEARCH"
        self.sl = None
        self.sr = None
        self.left_stale = 0
        self.right_stale = 0
        self.estimated_lane_width = 280.0
        self.last_target_x = 320.0

    def update(self, warped_binary):
        nz = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy)
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy)

        has_l = len(li) >= self.MIN_PIX_OK
        has_r = len(ri) >= self.MIN_PIX_OK

        # Safe polyfit with try-except to catch singular matrices on perfectly straight noisy lines
        if has_l:
            try:
                fl = np.polyfit(nzy[li], nzx[li], 2)
                self.sl = self._ema(self.sl, fl, self.EMA_ALPHA)
                self.left_stale = 0
            except Exception:
                has_l = False
                self.left_stale += 1

        if not has_l:
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.sl = None
            else:
                self.left_stale += 1

        if has_r:
            try:
                fr = np.polyfit(nzy[ri], nzx[ri], 2)
                self.sr = self._ema(self.sr, fr, self.EMA_ALPHA)
                self.right_stale = 0
            except Exception:
                has_r = False
                self.right_stale += 1
                
        if not has_r:
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.sr = None
            else:
                self.right_stale += 1

        if has_l and has_r:
            y_positions = [100, 200, 300, 400]
            widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y) for y in y_positions]
            weighted_avg_width = np.average(widths, weights=[4, 3, 2, 1])
            self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg

    def get_target_x(self, y_eval):
        hw = self.estimated_lane_width / 2.0
        def ev(fit): return float(np.polyval(fit, y_eval))

        has_right = (self.sr is not None)
        has_left  = (self.sl is not None)

        if not has_right and not has_left:
            return self.last_target_x, "DEAD_RECKONING (FALLBACK)"

        if has_right:
            if has_left:
                base_x = (ev(self.sl) + ev(self.sr)) / 2.0 + self.RIGHT_LANE_BIAS_PX
                anchor = "RL_DUAL"
            else:
                base_x = ev(self.sr) - hw + self.RIGHT_LANE_BIAS_PX
                anchor = "RL_FROM_EDGE"
        else:
            base_x = ev(self.sl) + self.DIVIDER_FOLLOW_OFFSET_PX
            anchor = "DIVIDER_FOLLOW"

        self.last_target_x = base_x
        return base_x, anchor

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid, margin = int(self.w * 0.40), self.SW_MARGIN

        l_lo, l_hi = margin, mid - margin
        r_lo, r_hi = mid + margin, self.w - margin

        lb = int(np.argmax(hist[l_lo:l_hi])) + l_lo if l_hi > l_lo else margin
        rb = int(np.argmax(hist[r_lo:r_hi])) + r_lo if r_hi > r_lo else mid + margin

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo, y_hi = self.h - (win + 1) * wh, self.h - win * wh
            xl0, xl1 = max(0, lx - self.SW_MARGIN), min(self.w, lx + self.SW_MARGIN)
            xr0, xr1 = max(0, rx - self.SW_MARGIN), min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]
            li.append(gl); ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li, ri = np.concatenate(li), np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = 80 # Poly margin
        def band(fit): return ((nzx > np.polyval(fit, nzy) - m) & (nzx < np.polyval(fit, nzy) + m)).nonzero()[0]
        
        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _ema(self, prev, new, alpha):
        if prev is None: return new.copy()
        return alpha * new + (1.0 - alpha) * prev

# ══════════════════════════════════════════════════════════════════════════════
# CONTROL: STANLEY & DIVIDER GUARD
# ══════════════════════════════════════════════════════════════════════════════
class StanleyController:
    def __init__(self, k=1.2, ks=0.2):
        self.k = k
        self.ks = ks

    def compute(self, target_x_px, heading_rad, velocity_ms, lane_width_px):
        ppm = max(lane_width_px, 50) / 0.35  # pixels per metre
        ce_m = (320.0 - target_x_px) / ppm   # cross-track error (metres)
        
        k_eff = self.k * min(1.0, velocity_ms / 0.25) if velocity_ms > 0 else self.k
        reactive_rad = heading_rad + math.atan2(k_eff * ce_m, velocity_ms + self.ks)
        return math.degrees(reactive_rad)

class DividerGuard:
    DIVIDER_SAFE_PX = 130
    EDGE_SAFE_PX = 100
    GAIN = 0.35
    MAX_CORR = 25.0

    def apply(self, steer_angle, left_fit, right_fit, car_x=320, y_eval=440):
        correction = 0.0
        div_corr = edge_corr = 0.0
        speed_scale = 1.0

        if left_fit is not None:
            gap = car_x - float(np.polyval(left_fit, y_eval))
            if gap < self.DIVIDER_SAFE_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min((self.GAIN * 3.0) * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.2, 1.0 - err / 60.0))

        if right_fit is not None:
            gap = float(np.polyval(right_fit, y_eval)) - car_x
            if gap < self.EDGE_SAFE_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR * 0.4)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 100.0))

        correction = div_corr - edge_corr
        return steer_angle + correction, speed_scale

class Controller:
    MAX_STEER = 45.0
    MAX_STEER_RATE = 20.0

    def __init__(self):
        self.prev_steer = 0.0
        self.guard = DividerGuard()
        self.stanley = StanleyController()
        self.base_speed = 50.0

    def compute(self, perc_res, velocity_ms=0.5):
        raw_steer = self.stanley.compute(
            perc_res.target_x, perc_res.heading_rad, velocity_ms, perc_res.lane_width_px)

        rate_delta = max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, raw_steer - self.prev_steer))
        steer_angle = self.prev_steer + rate_delta
        
        alpha = 0.7 
        steer_angle = alpha * self.prev_steer + (1 - alpha) * steer_angle
        self.prev_steer = steer_angle

        steer_guarded, guard_spd_mult = self.guard.apply(
            steer_angle, perc_res.sl, perc_res.sr, y_eval=perc_res.y_eval)
        steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_guarded))

        speed = float(self.base_speed)
        if perc_res.anchor == "DIVIDER_FOLLOW":
            speed *= 0.75

        final_speed = speed * guard_spd_mult
        return ControlOutput(steer_angle, final_speed, perc_res.target_x, perc_res.anchor)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP & TUNER UI
# ══════════════════════════════════════════════════════════════════════════════
def nothing(x): pass

def main():
    # Suppress numpy RankWarnings which flood the console when polyfit is run on straight/noisy lines
    warnings.simplefilter('ignore', np.RankWarning)
    
    print("[INFO] Initializing Camera...")
    cap = cv2.VideoCapture(0) # 0 for webcam, or change to a path like "video.mp4"
    
    if not cap.isOpened():
        print("[ERROR] Could not open video source! Check /dev/video0 or your webcam connection.")
        sys.exit(1)

    print("[INFO] Camera opened successfully. Building UI...")

    cv2.namedWindow("Tuner")
    
    # [FIX] Draw a slightly larger dummy image with text to ensure the window manager renders it correctly
    dummy_bg = np.zeros((50, 500), dtype=np.uint8)
    cv2.putText(dummy_bg, "Trackbars for tuning are below.", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Tuner", dummy_bg)
    cv2.waitKey(1)

    # Perception Tuners
    cv2.createTrackbar("Src Top Width", "Tuner", 240, 640, nothing)
    cv2.createTrackbar("Src Top Y", "Tuner", 260, 480, nothing)
    cv2.createTrackbar("CLAHE Limit", "Tuner", 30, 100, nothing) 
    cv2.createTrackbar("Threshold Block", "Tuner", 31, 101, nothing) 
    cv2.createTrackbar("Threshold C", "Tuner", 15, 50, nothing)
    cv2.createTrackbar("EMA Alpha", "Tuner", 55, 100, nothing) 
    
    # Control Tuners
    cv2.createTrackbar("Base Speed", "Tuner", 50, 100, nothing)
    cv2.createTrackbar("Stanley K", "Tuner", 12, 50, nothing) 
    cv2.createTrackbar("Stanley Ks", "Tuner", 20, 100, nothing) 
    cv2.createTrackbar("Divider Safe PX", "Tuner", 130, 300, nothing)
    cv2.createTrackbar("Edge Safe PX", "Tuner", 100, 300, nothing)

    tracker = HybridLaneTracker()
    controller = Controller()

    print("[INFO] Starting tracking loop. Press 'q' in any window to quit.")

    is_live = (cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0)

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret: 
                if is_live:
                    print("[WARNING] Frame dropped from live camera. Exiting loop.")
                    break
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    continue
                
            frame = cv2.resize(frame, (640, 480))

            try:
                t_width = cv2.getTrackbarPos("Src Top Width", "Tuner")
                t_y = cv2.getTrackbarPos("Src Top Y", "Tuner")
                clahe_lim = max(0.1, cv2.getTrackbarPos("CLAHE Limit", "Tuner") / 10.0)
                
                th_block = cv2.getTrackbarPos("Threshold Block", "Tuner")
                th_block = max(3, th_block if th_block % 2 != 0 else th_block + 1)
                
                th_c = cv2.getTrackbarPos("Threshold C", "Tuner")
                
                tracker.EMA_ALPHA = cv2.getTrackbarPos("EMA Alpha", "Tuner") / 100.0
                controller.base_speed = cv2.getTrackbarPos("Base Speed", "Tuner")
                controller.stanley.k = cv2.getTrackbarPos("Stanley K", "Tuner") / 10.0
                controller.stanley.ks = cv2.getTrackbarPos("Stanley Ks", "Tuner") / 100.0
                controller.guard.DIVIDER_SAFE_PX = cv2.getTrackbarPos("Divider Safe PX", "Tuner")
                controller.guard.EDGE_SAFE_PX = cv2.getTrackbarPos("Edge Safe PX", "Tuner")
            except cv2.error:
                t_width, t_y, clahe_lim, th_block, th_c = 240, 260, 3.0, 31, 15

            # --- BEV Transform ---
            src_pts = np.float32([[320-t_width//2, t_y], [320+t_width//2, t_y], [40, 450], [600, 450]])
            dst_pts = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, M, (640, 480))

            # --- Image Processing ---
            lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clahe_lim, tileGridSize=(8, 8))
            L = clahe.apply(lab[:, :, 0])
            
            binary = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, th_block, th_c)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

            # --- Tracking ---
            sl, sr, dbg = tracker.update(binary)
            target_x, anchor = tracker.get_target_x(y_eval=400.0)
            
            # Calculate heading
            heading_rad = 0.0
            def _lane_heading(fit, y): return math.atan2(np.polyval(fit, y - 50) - np.polyval(fit, y), 50)
            if sl is not None and sr is not None:
                heading_rad = (_lane_heading(sl, 400) + _lane_heading(sr, 400)) / 2.0
            elif sl is not None: heading_rad = _lane_heading(sl, 400)
            elif sr is not None: heading_rad = _lane_heading(sr, 400)

            conf = 1.0 if (sl is not None and sr is not None) else 0.5 if (sl is not None or sr is not None) else 0.0

            perc_res = PerceptionResult(
                warped_binary=binary, lane_dbg=dbg, sl=sl, sr=sr, 
                target_x=target_x, anchor=anchor, confidence=conf, 
                lane_width_px=tracker.estimated_lane_width, 
                curvature=tracker.get_curvature(400.0), heading_rad=heading_rad)

            # --- Control ---
            ctrl_res = controller.compute(perc_res, velocity_ms=0.5)

            # --- Visual Debug ---
            cv2.putText(dbg, f"Steer: {ctrl_res.steer_angle_deg:+.1f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(dbg, f"PWM: {ctrl_res.speed_pwm:.0f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(dbg, f"Anchor: {ctrl_res.anchor}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            tx = int(np.clip(target_x, 0, 639))
            cv2.line(dbg, (tx, 380), (tx, 420), (0, 255, 255), 2)

            cv2.polylines(frame, [src_pts.astype(np.int32)], True, (0,0,255), 2)

            cv2.imshow("Original", frame)
            cv2.imshow("Lane Tracking BEV", dbg)
            
            cv2.imshow("Tuner", dummy_bg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # [FIX] Ensures hardware releases lock on camera even if code crashes.
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up camera resources and closed windows.")

if __name__ == "__main__":
    main()