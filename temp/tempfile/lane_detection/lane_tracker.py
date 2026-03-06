import numpy as np
import cv2

class DeadReckoningNavigator:
    def __init__(self):
        self.last_valid_target    = 320.0
        self.last_valid_curvature = 0.0
        self._lost_time_s         = 0.0

    def reset_lost_timer(self):
        self._lost_time_s = 0.0

    def accumulate(self, dt: float):
        self._lost_time_s += dt

    def predict_target(self, last_speed=0.0, last_steering=0.0):
        t = max(0.0, self._lost_time_s)
        lateral_drift   = last_steering * 2.0 * t  # Approximation
        predicted_target = self.last_valid_target + lateral_drift
        if abs(self.last_valid_curvature) > 0.001:
            predicted_target += self.last_valid_curvature * 5000 * t
        predicted_target = float(np.clip(predicted_target, 150, 490))
        confidence       = max(0.0, 1.0 - t / 2.0)
        return predicted_target, confidence

class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.55
    EMA_ALPHA_TURN   = 0.75
    STALE_FIT_FRAMES = 12

    WIDE_ROAD_PX             = 420
    SINGLE_LANE_PX           = 200
    RIGHT_LANE_BIAS_PX       = 0
    DIVIDER_FOLLOW_OFFSET_PX = 90

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None
        self.sr         = None
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0
        self.dead_reckoner = DeadReckoningNavigator()
        self.estimated_lane_width = 280.0

    def update(self, warped_binary):
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy)
            mode_label  = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit  = fl
            curv_now = self.get_curvature(self.h // 2)
            alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
            self.sl        = self._ema(self.sl, fl, alpha)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit, self.sl = None, None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            curv_now = self.get_curvature(self.h // 2)
            alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
            self.sr         = self._ema(self.sr, fr, alpha)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit, self.sr = None, None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit, self.sl, self.left_stale, has_l = None, None, self.STALE_FIT_FRAMES, False
                else:
                    self.right_fit, self.sr, self.right_stale, has_r = None, None, self.STALE_FIT_FRAMES, False
            else:
                y_positions = [100, 200, 300, 400]
                widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y) for y in y_positions]
                weighted_avg_width = np.average(widths, weights=[4, 3, 2, 1])
                self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, last_speed=0.0, last_steering=0.0):
        sl, sr = self.sl, self.sr
        hw = lane_width_px / 2.0

        def ev(fit): return float(np.polyval(fit, y_eval))

        has_right = (sr is not None)
        has_left  = (sl is not None)

        if not has_right and not has_left:
            predicted_x, conf = self.dead_reckoner.predict_target(last_speed, last_steering)
            return predicted_x + extra_offset_px, f"DEAD_RECKONING_{conf:.2f}"

        if has_left:
            # Maintain a safe distance from the lane divider 
            base_x = ev(sl) + hw
            if has_right:
                anchor = "DIVIDER_AND_EDGE"
            else:
                anchor = "DIVIDER_ONLY"
        elif has_right:
            base_x = ev(sr) - hw
            anchor = "EDGE_ONLY"

        self.dead_reckoner.last_valid_target    = base_x
        self.dead_reckoner.last_valid_curvature = self.get_curvature(y_eval)
        self.dead_reckoner.reset_lost_timer()
        return base_x + extra_offset_px, anchor

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return abs(2.0 * a) / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid, margin = int(self.w * 0.40), self.SW_MARGIN

        l_lo = margin
        l_hi = mid - margin
        r_lo = mid + margin
        r_hi = self.w - margin

        lb = int(np.argmax(hist[l_lo:l_hi])) + l_lo if l_hi > l_lo else margin
        rb = int(np.argmax(hist[r_lo:r_hi])) + r_lo if r_hi > r_lo else mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode='same')
            p1 = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1-40):min(self.w, p1+40)] = 0
            p2 = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

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

        li, ri = np.concatenate(li) if len(li) else np.array([]), np.concatenate(ri) if len(ri) else np.array([])
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)

        def band(fit): 
            if fit is None: return np.array([], dtype=int)
            return ((nzx > np.polyval(fit, nzy) - m) & (nzx < np.polyval(fit, nzy) + m)).nonzero()[0]

        li = band(self.sl)
        ri = band(self.sr)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 180 < w < 420

    def _ema(self, prev, new, alpha=None):
        if alpha is None:
            alpha = self.EMA_ALPHA
        if prev is None: return new.copy()
        return alpha * new + (1.0 - alpha) * prev
