"""
lane_tracker.py
================
BFMC Autonomous Module – Headless Lane Tracking Utilities

Extracted from backup1-rightlane driving/lane.py (HybridLaneTracker + JunctionDetector).
All cv2.imshow(), trackbar, and Pilot GUI logic has been removed.

Contains:
  - HybridLaneTracker  : BEV sliding-window / polynomial lane tracker with EMA smoothing
  - JunctionDetector   : Evidence-based state machine (NORMAL <-> JUNCTION)
  - get_bev()          : Bird's-eye view preprocessing helper
  - pure_pursuit()     : Converts lateral error to steering angle (degrees)
"""

import cv2
import numpy as np
import math
import logging

log = logging.getLogger(__name__)

# ===========================================================================
# PHYSICAL CONSTANTS
# ===========================================================================
WHEELBASE_M          = 0.23    # front-to-rear axle distance in metres
LANE_WIDTH_M         = 0.35    # one-lane physical width in metres

# ===========================================================================
# CAMERA – Bird's Eye View calibration
# SRC: 4 points in raw camera image  [TL, TR, BL, BR]
# DST: corresponding points in BEV   [TL, TR, BL, BR]
# ===========================================================================
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# LANE OFFSET CONSTANTS
# ===========================================================================
RIGHT_LANE_OFFSET_PX = 30   # default right-lane target offset from image centre
DUAL_OFFSET_PX       = -10  # slight left bias when both lines are found


# ===========================================================================
# BEV PREPROCESSING HELPER
# ===========================================================================
def get_bev(frame, M=None, clahe=None):
    """
    Convert a raw BGR camera frame to a bird's-eye-view binary image
    ready for lane detection.

    Parameters
    ----------
    frame  : np.ndarray  – raw BGR frame (480 x 640)
    M      : np.ndarray  – 3x3 perspective transform matrix (computed once)
    clahe  : cv2.CLAHE   – pre-created CLAHE object (computed once)

    Returns
    -------
    binary : np.ndarray  – thresholded BEV binary (uint8, single channel)
    """
    if M is None:
        M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    warped_colour = cv2.warpPerspective(frame, M, (640, 480))
    hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
    L   = clahe.apply(hls[:, :, 1])

    binary = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ===========================================================================
# PURE PURSUIT STEERING
# ===========================================================================
def pure_pursuit(target_x, look_ahead_px, lane_width_px, image_width=640):
    """
    Compute steering angle (degrees) using the pure-pursuit formula.

    Parameters
    ----------
    target_x      : float  – target BEV x-pixel the car should steer toward
    look_ahead_px : float  – longitudinal look-ahead distance in pixels
    lane_width_px : float  – calibrated lane width in pixels
    image_width   : int    – width of the BEV image (default 640)

    Returns
    -------
    steer_deg : float  – steering angle in degrees (+ = right, - = left)
    """
    lane_width_px = max(lane_width_px, 50)
    ppm    = lane_width_px / LANE_WIDTH_M      # pixels-per-metre
    cx     = image_width / 2.0
    dx     = target_x - cx
    dy     = max(float(look_ahead_px), 1.0)
    ld     = math.sqrt(dx * dx + dy * dy)
    alpha  = math.atan2(dx, dy)
    wb_px  = WHEELBASE_M * ppm
    steer  = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
    return math.degrees(steer)


# ===========================================================================
# HYBRID LANE TRACKER
# ===========================================================================
class HybridLaneTracker:
    """
    Two-stage BEV lane tracker:
      SEARCH  -> sliding window histogram approach
      TRACKING -> polynomial-band search (faster, more stable)

    Maintains EMA-smoothed polynomials (sl = left/centre-divider,
    sr = right/outer-edge) across frames.
    """

    NWINDOWS         = 9
    SW_MARGIN        = 60     # sliding-window half-width (px)
    MINPIX           = 50     # min pixels to recenter a window
    POLY_MARGIN_BASE = 60     # normal search band around polynomial (px)
    POLY_MARGIN_CURV = 120    # wider band when curvature is high
    MIN_PIX_OK       = 200    # min pixels to declare a line "found"
    EMA_ALPHA        = 0.50   # polynomial EMA weight (higher = more responsive)
    STALE_FIT_FRAMES = 5      # frames before discarding a stale polynomial

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None   # EMA-smoothed left polynomial
        self.sr         = None   # EMA-smoothed right polynomial
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def update(self, warped_binary):
        """
        Process one BEV binary frame and update internal polynomial state.

        Returns
        -------
        sl         : np.ndarray | None  – smoothed left polynomial coefficients
        sr         : np.ndarray | None  – smoothed right polynomial coefficients
        mode_label : str                – "POLY" or "SLIDE"
        """
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri = self._poly_search(warped_binary, nzx, nzy, curvature=curv)
            mode_label = "POLY"
        else:
            li, ri = self._sliding_window(warped_binary, nzx, nzy)
            mode_label = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit   = fl
            self.sl         = self._ema(self.sl, fl)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit   = fr
            self.sr          = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        # Sanity: if both found, verify lane width is plausible
        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                # Drop the less-confident line
                if self.left_conf < self.right_conf:
                    self.left_fit   = None
                    self.sl         = None
                    self.left_stale = self.STALE_FIT_FRAMES
                    has_l           = False
                else:
                    self.right_fit   = None
                    self.sr          = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r            = False

        self.mode = (
            "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None)
            else "SEARCH"
        )
        return self.sl, self.sr, mode_label

    def get_target_x(self, y_eval, lane_width_px,
                     extra_offset_px=RIGHT_LANE_OFFSET_PX,
                     nav_state="NORMAL"):
        """
        Compute the BEV x-pixel the car should steer toward.

        RIGHT-LANE CONVENTION:
          sl (left_fit)  = white centre divider
          sr (right_fit) = white outer edge

        Returns
        -------
        target_x : float | None  – x-pixel target (None = lost)
        anchor   : str           – description of which lines were used
        """
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit):
            return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None:
                return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None:
                return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr is not None:
                return ev(sr) - hw + extra_offset_px, "JCT_EDGE"
            if sl is not None:
                return ev(sl) + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        # NORMAL right-lane driving
        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "DUAL"

        # Ghost-line extrapolation (only one line visible)
        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "GHOST_L"
        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + DUAL_OFFSET_PX, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval):
        """Return approximate curvature (1/px) of the best available polynomial."""
        fit = self.sr if self.sr is not None else self.sl
        if fit is None:
            return 0.0
        a, b   = fit[0], fit[1]
        num    = abs(2.0 * a)
        denom  = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def reset(self):
        """Hard-reset all tracking state (call at start of each maneuver)."""
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None
        self.sr         = None
        self.left_stale  = 0
        self.right_stale = 0

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _sliding_window(self, warped, nzx, nzy):
        """Full sliding-window search. Returns index arrays (li, ri)."""
        hist = np.sum(warped[self.h // 2:, :], axis=0)
        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN

        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float),
                                   np.ones(20) / 20, mode='same')
            p1  = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1 - 40):min(self.w, p1 + 40)] = 0
            p2  = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh

            xl0 = max(0, lx - self.SW_MARGIN)
            xl1 = min(self.w, lx + self.SW_MARGIN)
            xr0 = max(0, rx - self.SW_MARGIN)
            xr1 = min(self.w, rx + self.SW_MARGIN)

            gl = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) &
                  (nzx >= xr0) & (nzx < xr1)).nonzero()[0]

            li.append(gl)
            ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        return np.concatenate(li), np.concatenate(ri)

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        """Polynomial-band search (fast tracking mode). Falls back to sliding window."""
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015
             else self.POLY_MARGIN_BASE)

        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        return li, ri

    def _width_sane(self, lf, rf, y=400):
        """Return True if lane width at row y is within plausible range."""
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 80 < w < 560

    def _ema(self, prev, new):
        """Exponential moving average for polynomial coefficient arrays."""
        if prev is None:
            return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


# ===========================================================================
# JUNCTION DETECTOR
# ===========================================================================
class JunctionDetector:
    """
    Evidence-based state machine that detects road junctions from the BEV
    binary image.

    States
    ------
    NORMAL   – driving on a regular lane section
    JUNCTION – intersection area detected; triggers dead-reckoning handoff

    Evidence signals
    ----------------
    both_lost    : both lane lines have < 200 pixels
    cross_energy : top-half of BEV has MORE energy than bottom (crossbar)
    wide_lane    : measured lane width exceeds WIDTH_RATIO_HIGH * expected
    """

    ENTRY_FRAMES       = 5    # consecutive evidence frames to enter JUNCTION
    EXIT_FRAMES        = 8    # consecutive clear frames to exit JUNCTION
    CROSS_ENERGY_RATIO = 1.4  # top/bottom energy ratio threshold
    WIDTH_RATIO_HIGH   = 1.6  # measured/expected width ratio for wide-lane trigger
    MIN_BOT_ENERGY     = 500  # minimum bottom-half energy (ignore noise)

    def __init__(self):
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0

    def update(self, warped_binary, left_conf, right_conf,
               left_fit, right_fit, lane_width_px):
        """
        Evaluate one BEV frame and update junction state.

        Parameters
        ----------
        warped_binary : np.ndarray  – BEV binary image
        left_conf     : int          – pixel count for left line
        right_conf    : int          – pixel count for right line
        left_fit      : array | None – left polynomial
        right_fit     : array | None – right polynomial
        lane_width_px : float        – calibrated lane width in pixels

        Returns
        -------
        state : str  – "NORMAL" or "JUNCTION"
        """
        h, w = warped_binary.shape

        both_lost    = (left_conf < 200) and (right_conf < 200)
        hist_top     = float(np.sum(warped_binary[:h // 2, :]))
        hist_bot     = float(np.sum(warped_binary[h // 2:, :]))

        cross_energy = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_energy = (hist_top / hist_bot) > self.CROSS_ENERGY_RATIO

        wide_lane = False
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit,  h - 50)
            rx = np.polyval(right_fit, h - 50)
            if (rx - lx) > lane_width_px * self.WIDTH_RATIO_HIGH:
                wide_lane = True

        evidence = both_lost or cross_energy or wide_lane

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state         = "JUNCTION"
                self.exit_count    = 0
                self.frames_in_jct = 0
                log.info("[JunctionDetector] Junction ENTERED")

        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15:
                self.state       = "NORMAL"
                self.entry_count = 0
                log.info("[JunctionDetector] Junction EXITED")

        return self.state
