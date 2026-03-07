# main.py
import tkinter as tk
from tkinter import messagebox
import time
import math
import numpy as np
import cv2
import csv
import json
from PIL import Image, ImageTk

from config import *
from map_engine import MapEngine
from dashboard_ui import DashboardUI
from adas_vision_utils import JunctionDetector, RoundaboutNavigator, annotate_bev

# --- AUTOMATION IMPORTS ---
try:
    from perception import VisionPipeline
    from control import Controller
    _AUTO_DRIVE_AVAILABLE = True
except ImportError:
    _AUTO_DRIVE_AVAILABLE = False

try:
    from traffic_module import TrafficDecisionEngine, ThreadedYOLODetector
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False

try:
    from serial_handler import STM32_SerialHandler
except ImportError:
    class STM32_SerialHandler:
        def connect(self): return True
        def disconnect(self): pass
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def set_light_state(self, state, on): pass

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  OPENCV TRAFFIC LIGHT COLOR DETECTION  (no external dependency)
# ─────────────────────────────────────────────────────────────
def opencv_detect_traffic_light_color(bgr_frame, x1, y1, x2, y2):
    """
    Pure OpenCV HSV-based traffic-light colour detection.
    Crops the bounding-box region, converts to HSV, and checks
    which colour (red / yellow / green) has the highest saturated-
    pixel count.  Returns "RED", "YELLOW", "GREEN", or "NONE".
    """
    h_img, w_img = bgr_frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img, x2), min(h_img, y2)
    if x2c <= x1c or y2c <= y1c:
        return "NONE"

    roi = bgr_frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return "NONE"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── RED  (wraps around 0°/180°) ──────────────────────────
    mask_r1 = cv2.inRange(hsv, np.array([0,  120, 100]), np.array([10,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160,120, 100]), np.array([180, 255, 255]))
    red_px   = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)

    # ── YELLOW  (15°–35°) ─────────────────────────────────────
    mask_y  = cv2.inRange(hsv, np.array([15, 120, 100]), np.array([35,  255, 255]))
    yel_px  = cv2.countNonZero(mask_y)

    # ── GREEN  (40°–85°) ──────────────────────────────────────
    mask_g  = cv2.inRange(hsv, np.array([40, 120, 100]), np.array([85,  255, 255]))
    grn_px  = cv2.countNonZero(mask_g)

    # Need at least 8 matching pixels to avoid noise
    MIN_PX = 8
    best = max(red_px, yel_px, grn_px)
    if best < MIN_PX:
        return "NONE"
    if best == red_px:
        return "RED"
    if best == yel_px:
        return "YELLOW"
    return "GREEN"


# ─────────────────────────────────────────────────────────────
class MockCtrl: pass

# ─────────────────────────────────────────────────────────────
#  SIGN DETECTION DISTANCES  (defaults – overridden by UI sliders)
# ─────────────────────────────────────────────────────────────
SIGN_DETECT_DIST_DEFAULT  = 5.0   # m – detection window
SIGN_TRIGGER_DIST_DEFAULT = 2.0   # m – action trigger
SIGN_MIN_BBOX_H_PX        = 20    # px min bbox height

# ─────────────────────────────────────────────────────────────
#  ACCURATE PINHOLE DISTANCE MODEL
#  dist_m = (real_height_m * CAMERA_FOCAL_LENGTH_PX) / bbox_height_px
#
#  Real heights measured for BFMC signs (approximate, tune on track):
#    - Traffic signs (stop, crosswalk, priority…) : 0.08 m  (from config)
#    - Traffic light housing                       : 0.20 m
#    - Car (sedan profile)                         : 0.35 m  (door height)
#    - Pedestrian (standing)                       : 0.50 m  (torso visible)
#
#  Cross-check: also estimate from bbox width when available.
#  Final estimate = weighted average(height_est, width_est).
# ─────────────────────────────────────────────────────────────
_SIGN_REAL_HEIGHTS = {
    "stop-sign":          REAL_SIGN_HEIGHT_M,   # 0.08 m from config
    "crosswalk-sign":     REAL_SIGN_HEIGHT_M,
    "priority-sign":      REAL_SIGN_HEIGHT_M,
    "parking-sign":       REAL_SIGN_HEIGHT_M,
    "highway-entry-sign": REAL_SIGN_HEIGHT_M,
    "highway-exit-sign":  REAL_SIGN_HEIGHT_M,
    "roundabout-sign":    REAL_SIGN_HEIGHT_M,
    "oneway-sign":        REAL_SIGN_HEIGHT_M,
    "noentry-sign":       REAL_SIGN_HEIGHT_M,
    "traffic-light":      0.20,   # taller housing
    "car":                0.35,   # door/profile height
    "pedestrian":         0.50,   # visible torso height
}
_SIGN_REAL_WIDTHS = {
    "stop-sign":          REAL_SIGN_HEIGHT_M,   # square signs
    "crosswalk-sign":     REAL_SIGN_HEIGHT_M,
    "priority-sign":      REAL_SIGN_HEIGHT_M,
    "parking-sign":       REAL_SIGN_HEIGHT_M,
    "highway-entry-sign": REAL_SIGN_HEIGHT_M,
    "highway-exit-sign":  REAL_SIGN_HEIGHT_M,
    "roundabout-sign":    REAL_SIGN_HEIGHT_M,
    "oneway-sign":        REAL_SIGN_HEIGHT_M * 2.0,  # wider
    "noentry-sign":       REAL_SIGN_HEIGHT_M,
    "traffic-light":      0.10,   # narrow housing
    "car":                0.70,   # sedan width
    "pedestrian":         0.30,   # shoulder width
}

def estimate_distance_m(label: str, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Accurate single-camera distance estimate using the pinhole model.
    Uses per-class real heights AND widths, cross-checks both, returns
    a confidence-weighted average.

    dist = (real_size_m * CAMERA_FOCAL_LENGTH_PX) / bbox_size_px

    When both h and w estimates are available they are averaged with
    equal weight.  If one axis is implausibly small (<5 px) it is
    excluded to avoid divide-by-near-zero errors.
    """
    lbl = label.lower()
    real_h = _SIGN_REAL_HEIGHTS.get(lbl, REAL_SIGN_HEIGHT_M)
    real_w = _SIGN_REAL_WIDTHS.get(lbl, REAL_SIGN_HEIGHT_M)
    f      = CAMERA_FOCAL_LENGTH_PX

    box_h = max(y2 - y1, 1)
    box_w = max(x2 - x1, 1)

    estimates = []
    if box_h >= 5:
        estimates.append(real_h * f / box_h)
    if box_w >= 5:
        estimates.append(real_w * f / box_w)

    if not estimates:
        return 99.0
    return float(sum(estimates) / len(estimates))

# ─────────────────────────────────────────────────────────────
#  LANE-CENTRE TRIM  (pixels added to extra_offset_px)
#  0 = perfect mathematical centre between sl and sr.
#  Positive = nudge right inside the lane (buffer from left divider).
#  Negative = nudge left (buffer from right outer edge).
#  Tune via the Lane Offset slider.  Start at 0.
# ─────────────────────────────────────────────────────────────
RIGHT_LANE_OFFSET_DEFAULT = 0     # px trim from lane centre


class BFMC_App:
    def __init__(self, root):
        self.root = root
        self.root.title("BFMC ADAS Command Center")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        self.root.configure(bg=THEME["bg"])

        self.ui = DashboardUI(self.root, self)
        self.map_engine = MapEngine()

        self.handler = STM32_SerialHandler()
        self.is_connected = False

        # Physics State
        self.car_x, self.car_y, self.car_yaw = 0.5, 0.5, 0.0
        self.current_speed, self.current_steer = 0.0, 0.0
        self.steer_ema = 0.0
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.last_ctrl_time = time.time()
        self.current_hz = 0.0
        self.STEER_STEP = 5.0
        self.is_calibrating = False  # FIX: declared in __init__ to avoid AttributeError

        # ADAS State
        self.adas_enabled = True
        self.speed_limit = 150
        self.in_highway_mode = False
        self.ai_override_speed = None
        self.stop_until = 0.0
        self.slow_until = 0.0
        self.acted_signs = set()
        self.last_logged_adas = ""
        self.current_light_status = "NONE"
        self._last_traffic_mult   = 1.0    # updated each frame from TrafficResult
        self._pedestrian_blocking = False   # updated each frame from TrafficResult
        self._slow_speed          = 80.0   # updated each cycle to 80% of base_speed
        # FIX: highway flag preserved by reading it BEFORE the global reset each cycle
        self.active_state_flags = {
            "stop_sign": False, "no_entry": False, "pedestrian": False,
            "red_light": False, "yellow_light": False, "green_light": False,
            "caution": False, "highway": False, "park": False, "overtake": False
        }

        # Routing State
        self.mode = "DRIVE"
        self.start_node = None; self.end_node = None; self.pass_node = None
        self.path = []
        self.visited_path_nodes = set()
        self.path_signs = []
        self.visible_signs = {}
        self.current_acting_sign = None; self.acting_sign_end_time = 0.0; self.parking_sign_acting = None
        self.has_driven = False

        # Parking/Playback State
        self.is_playing_back = False
        self.is_parking_reverse_mode = False
        self.is_waiting_to_reverse = False
        self.parking_wait_until = 0.0
        self.playback_queue = []; self.playback_cmd = None; self.playback_frames = 0

        # Overtake Logic State
        self.overtake_state = "IDLE"; self.overtake_timer = 0.0
        self.last_car_dist = 999.0; self.last_car_time = time.time()

        # Pipelines (Lane Detection)
        self.is_auto_mode    = False
        self.auto_start_time = 0.0
        self.auto_base_speed = None   # locked when autonomous starts; None = use slider
        self.vision = VisionPipeline() if _AUTO_DRIVE_AVAILABLE else None
        self.controller = Controller() if _AUTO_DRIVE_AVAILABLE else None
        self.jct = JunctionDetector(); self.rbt = RoundaboutNavigator()

        self.traffic_engine, self.yolo = None, None
        if _AI_AVAILABLE:
            self.yolo = ThreadedYOLODetector(YOLO_MODEL_FILE)
            self.traffic_engine = TrafficDecisionEngine(self.yolo)

        self.picam2 = None
        self.latest_frame = None
        self.latest_raw_bgr = None

        # ── ADAS Distance & Lane Offset sliders ─────────────────
        self.var_detect_dist  = tk.DoubleVar(value=SIGN_DETECT_DIST_DEFAULT)
        self.var_trigger_dist = tk.DoubleVar(value=SIGN_TRIGGER_DIST_DEFAULT)
        self.var_lane_offset  = tk.DoubleVar(value=RIGHT_LANE_OFFSET_DEFAULT)
        self._build_distance_panel()

        # Set initial base speed to 50
        try:
            self.ui.slider_base_speed.set(50)
        except Exception:
            pass

        self.load_config()
        self.set_mode("DRIVE")
        if _CAM_AVAILABLE:
            self._init_camera()

        # Bindings & Loops
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.update_camera()
        self.control_loop()
        self.render_map()

    # ─────────────────────────────────────────────────────────
    # ADAS DISTANCE & LANE OFFSET PANEL
    # ─────────────────────────────────────────────────────────
    def _build_distance_panel(self):
        # Compact slider bar at bottom: Detect Dist / Trigger Dist / Lane Offset
        bg   = THEME["panel"]
        fg   = THEME["fg"]
        acc  = THEME["accent"]
        font = THEME.get("font_p", ("Helvetica", 9))

        frame = tk.Frame(self.root, bg=bg, bd=1, relief=tk.SUNKEN)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

        tk.Label(frame, text="ADAS DISTANCES & LANE",
                 bg=bg, fg=acc,
                 font=THEME.get("font_h", ("Helvetica", 10, "bold"))
                 ).pack(side=tk.LEFT, padx=8)

        # Detect Distance (1–10 m)
        tk.Label(frame, text="Detect (m):", bg=bg, fg=fg, font=font
                 ).pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_detect = tk.Label(frame, text=f"{SIGN_DETECT_DIST_DEFAULT:.1f}",
                                   bg=bg, fg="cyan", font=font, width=4)
        self.lbl_detect.pack(side=tk.LEFT)
        tk.Scale(frame, variable=self.var_detect_dist,
                 from_=1.0, to=10.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=130, bg=bg, fg=fg,
                 highlightthickness=0, troughcolor="#333",
                 command=lambda v: self.lbl_detect.config(text=f"{float(v):.1f}")
                 ).pack(side=tk.LEFT, padx=2)

        tk.Frame(frame, bg="#444", width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Trigger Distance (0.5–5 m)
        tk.Label(frame, text="Trigger (m):", bg=bg, fg=fg, font=font
                 ).pack(side=tk.LEFT, padx=(4, 0))
        self.lbl_trigger = tk.Label(frame, text=f"{SIGN_TRIGGER_DIST_DEFAULT:.1f}",
                                    bg=bg, fg="cyan", font=font, width=4)
        self.lbl_trigger.pack(side=tk.LEFT)
        tk.Scale(frame, variable=self.var_trigger_dist,
                 from_=0.5, to=5.0, resolution=0.5,
                 orient=tk.HORIZONTAL, length=130, bg=bg, fg=fg,
                 highlightthickness=0, troughcolor="#333",
                 command=lambda v: self.lbl_trigger.config(text=f"{float(v):.1f}")
                 ).pack(side=tk.LEFT, padx=2)

        tk.Frame(frame, bg="#444", width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Lane Offset (-100 to +100 px; positive = shift target right = stay right lane)
        tk.Label(frame, text="Lane Offset (px):", bg=bg, fg=fg, font=font
                 ).pack(side=tk.LEFT, padx=(4, 0))
        self.lbl_lane_offset = tk.Label(frame,
                                        text=f"{RIGHT_LANE_OFFSET_DEFAULT:+.0f}",
                                        bg=bg, fg="yellow", font=font, width=5)
        self.lbl_lane_offset.pack(side=tk.LEFT)
        tk.Scale(frame, variable=self.var_lane_offset,
                 from_=-100, to=100, resolution=5,
                 orient=tk.HORIZONTAL, length=150, bg=bg, fg=fg,
                 highlightthickness=0, troughcolor="#333",
                 command=lambda v: self.lbl_lane_offset.config(text=f"{float(v):+.0f}")
                 ).pack(side=tk.LEFT, padx=2)

        # Reset button
        tk.Button(frame, text="Reset",
                  bg="#333", fg=fg, font=font,
                  command=self._reset_distance_sliders
                  ).pack(side=tk.LEFT, padx=8)

    def _reset_distance_sliders(self):
        self.var_detect_dist.set(SIGN_DETECT_DIST_DEFAULT)
        self.var_trigger_dist.set(SIGN_TRIGGER_DIST_DEFAULT)
        self.var_lane_offset.set(RIGHT_LANE_OFFSET_DEFAULT)
        self.lbl_detect.config(text=f"{SIGN_DETECT_DIST_DEFAULT:.1f}")
        self.lbl_trigger.config(text=f"{SIGN_TRIGGER_DIST_DEFAULT:.1f}")
        self.lbl_lane_offset.config(text=f"{RIGHT_LANE_OFFSET_DEFAULT:+.0f}")
        self.ui.log_event(
            f"Distances reset – detect={SIGN_DETECT_DIST_DEFAULT}m "
            f"trigger={SIGN_TRIGGER_DIST_DEFAULT}m "
            f"lane_offset={RIGHT_LANE_OFFSET_DEFAULT:+}px", "INFO")

    # ─────────────────────────────────────────────────────────
    # HARDWARE AND PIPELINE METHODS
    # ─────────────────────────────────────────────────────────
    def _init_camera(self):
        for attempt in range(3):
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_video_configuration(
                    main={"size": (640, 480), "format": "XRGB8888"}))
                self.picam2.start()
                self.ui.log_event("Camera initialized successfully.", "SUCCESS")
                return
            except Exception as e:
                if self.picam2:
                    try: self.picam2.stop(); self.picam2.close()
                    except: pass
                self.picam2 = None
                time.sleep(1.5)
        self.ui.log_event("CAMERA RESOURCE BUSY!", "CRITICAL")

    def toggle_connection(self):
        if not self.is_connected:
            if self.handler.connect():
                self.is_connected = True
                self.ui.lbl_conn.config(text="🟢 CONNECTED", fg=THEME["success"])
                self.ui.btn_connect.config(text="DISCONNECT", bg=THEME["danger"])
                self.ui.log_event("🔗 Connected to STM32 Hardware successfully.", "SUCCESS")
                if self.has_driven and messagebox.askyesno("Resume Run", "Resume from car's last position?"):
                    self.align_car_to_path()
                else:
                    self.in_highway_mode = False; self.ai_override_speed = None
                    self.stop_until = 0.0; self.slow_until = 0.0
                    self.acted_signs.clear(); self.is_playing_back = False
                    self.is_parking_reverse_mode = False; self.is_waiting_to_reverse = False
                    self.playback_queue = []; self.overtake_state = "IDLE"; self.last_car_dist = 999.0
                    self.visited_path_nodes.clear(); self.has_driven = True
                    self.current_acting_sign = None; self.parking_sign_acting = None
                    for s in self.path_signs:
                        if s.get('status') in ["✅ CONFIRMED", "⚠️ MISSED", "🟢 ACTING"]:
                            s['status'] = "⏳ PENDING"
                    self.ui.update_sign_table(self.path_signs)

                    if self.start_node:
                        self.car_x = float(self.map_engine.G.nodes[self.start_node].get('x', DEFAULT_START_X))
                        self.car_y = float(self.map_engine.G.nodes[self.start_node].get('y', DEFAULT_START_Y))
                        if self.path and len(self.path) >= 2:
                            x1, y1 = float(self.map_engine.G.nodes[self.path[0]].get('x', 0)), float(self.map_engine.G.nodes[self.path[0]].get('y', 0))
                            x2, y2 = float(self.map_engine.G.nodes[self.path[1]].get('x', 0)), float(self.map_engine.G.nodes[self.path[1]].get('y', 0))
                            self.car_yaw = math.atan2(y2 - y1, x2 - x1)
                        else:
                            self.car_yaw = 0.0
                    else:
                        self.car_x, self.car_y, self.car_yaw = DEFAULT_START_X, DEFAULT_START_Y, 0.0
                self.current_speed, self.current_steer = 0.0, 0.0
                self.render_map()
        else:
            self.handler.disconnect(); self.is_connected = False
            self.ui.log_event("🔌 Disconnected from STM32.", "WARN")
            self.ui.lbl_conn.config(text="⚫ DISCONNECTED", fg=THEME["danger"])
            self.ui.btn_connect.config(text="CONNECT CAR", bg=THEME["accent"])

    def toggle_auto_mode(self):
        self.is_auto_mode = not self.is_auto_mode
        if self.is_auto_mode:
            # Snapshot slider value once as the autonomous base speed.
            # This value is used for the lifetime of this autonomous run.
            # The slider cannot change it while auto is active.
            self.auto_base_speed = self.ui.slider_base_speed.get()
            self.auto_start_time = time.time()
            self.is_calibrating  = True
            self.steer_ema       = 0.0
            # Force all keys to False so no residual keypress carries into auto
            for k in self.keys: self.keys[k] = False
            self.ui.btn_auto.config(text="MODE: AUTONOMOUS", bg="#9b59b6")
            self.ui.log_event(
                f"🤖 Switched to AUTONOMOUS. Base speed locked at {self.auto_base_speed}. "
                f"Slider & keyboard DISABLED. Calibrating 5s …", "SUCCESS")
        else:
            self.is_calibrating  = False
            self.auto_base_speed = None
            # Release any stuck keys
            for k in self.keys: self.keys[k] = False
            self.ui.btn_auto.config(text="MODE: MANUAL", bg="#444")
            self.ui.log_event("🖐 Switched to MANUAL mode. Keyboard & slider active.", "WARN")
            self.ui.log_event("Lane-following DISABLED. Keyboard control active.", "INFO")

    def toggle_adas_mode(self):
        self.adas_enabled = not self.adas_enabled
        if self.adas_enabled:
            self.ui.btn_adas.config(text="ADAS ASSIST: ON", bg="#9b59b6")
            self.ui.log_event("✅ ADAS ASSIST enabled – sign & obstacle responses active.", "SUCCESS")
        else:
            self.ui.btn_adas.config(text="ADAS ASSIST: OFF", bg="#444")
            self.ui.log_event("⚠️ ADAS ASSIST DISABLED – no automatic responses.", "WARN")

    # ─────────────────────────────────────────────────────────
    # CSV PLAYBACK
    # ─────────────────────────────────────────────────────────
    def start_csv_playback(self, filename="car_actions.csv"):
        if self.is_playing_back or self.is_waiting_to_reverse:
            return
        try:
            self.playback_queue = []
            with open(filename, mode='r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        s = float(parts[0])
                        self.playback_queue.append({
                            "speed": s,
                            "steering": float(parts[1]),
                            "pwm": int(float(parts[2])),
                            "direction": int(float(parts[3])),
                            "duration": int(float(parts[4]))
                        })
            # Hard-stop appended so the car halts cleanly at end of sequence
            self.playback_queue.append({
                "speed": 0, "steering": 0, "pwm": 0, "direction": 0, "duration": 3
            })
            self.is_playing_back = True; self.is_parking_reverse_mode = False
            self.playback_cmd = None; self.playback_frames = 0
            self.ui.log_event("Commencing Auto-Park.", "SUCCESS")
        except Exception as e:
            self.ui.log_event(f"Auto-Park failed: {e}", "CRITICAL")
            self.is_playing_back = False

    def start_reverse_parking_exit(self):
        filename = "car_actions.csv"
        try:
            self.playback_queue = []
            with open(filename, mode='r') as f:
                lines = f.readlines()
                for line in reversed(lines[1:]):
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        s = float(parts[0])
                        # Speed negated (reverse), steering negated (opposite lock)
                        self.playback_queue.append({
                            "speed":    -abs(s),          # always reverse
                            "steering": -float(parts[1]), # mirror steer for reverse
                            "pwm":      int(float(parts[2])),
                            "direction": -1,
                            "duration": int(float(parts[4]))
                        })
            # Append a hard-stop command at the end so the car halts cleanly
            self.playback_queue.append({
                "speed": 0, "steering": 0, "pwm": 0, "direction": 0, "duration": 3
            })
            self.is_playing_back = True; self.is_parking_reverse_mode = True
            self.playback_cmd = None; self.playback_frames = 0
            self.ui.log_event("Executing Reverse Auto-Park Exit sequence.", "WARN")
        except Exception as e:
            self.ui.log_event(f"Reverse exit failed: {e}", "CRITICAL")

    # ─────────────────────────────────────────────────────────
    # MAP & ROUTING
    # ─────────────────────────────────────────────────────────
    def set_mode(self, m):
        self.mode = m
        self.ui.var_main_mode.set(m)
        for w in self.ui.tool_frame.winfo_children():
            w.destroy()
        if m == "NAV":
            self.ui.log_event("🗺 Mode: NAVIGATION – select route on map.", "INFO")
            self.ui.build_nav_tools(self)
        elif m == "SIGN":
            self.ui.log_event("🚧 Mode: SIGN EDITOR – place/delete signs on map.", "INFO")
            self.ui.build_sign_tools(self)
        else:
            self.ui.log_event("🚗 Mode: DRIVE – click map to teleport car.", "INFO")
            tk.Label(self.ui.tool_frame,
                     text="Drive Mode Active - Click Map to Teleport Digital Twin",
                     bg=THEME["panel"], fg=THEME["success"],
                     font=THEME["font_h"]).pack(side=tk.LEFT, padx=10, pady=5)

    def on_map_click(self, event):
        canvas_x = self.ui.map_canvas.canvasx(event.x)
        canvas_y = self.ui.map_canvas.canvasy(event.y)
        mx, my = self.map_engine.to_meter(canvas_x, canvas_y)
        click_point = np.array([canvas_x, canvas_y])

        if self.mode == "DRIVE":
            self.car_x, self.car_y = mx, my
            self.current_speed = 0
            self.render_map()
            return

        closest = None; min_d = 60.0
        for n, p in self.map_engine.node_pixels.items():
            d = np.linalg.norm(np.array(p) - click_point)
            if d < min_d:
                min_d, closest = d, n

        if self.mode == "NAV" and closest:
            t = self.ui.var_path.get()
            if t == "START":   self.start_node = closest
            elif t == "END":   self.end_node = closest; self.calc_path()
            elif t == "PASS":  self.pass_node = closest
            self.render_map()

        elif self.mode == "SIGN":
            if self.ui.chk_del.get():
                orig_count = len(self.map_engine.signs)
                self.map_engine.signs = [
                    s for s in self.map_engine.signs
                    if np.linalg.norm(np.array(self.map_engine.node_pixels[s['node']]) - click_point) > 60
                ]
                if len(self.map_engine.signs) < orig_count:
                    self.ui.log_event("Deleted sign(s) at click location.", "WARN")
            elif closest:
                if not any(s['node'] == closest for s in self.map_engine.signs):
                    self.map_engine.signs.append({
                        "node": closest,
                        "type": self.ui.var_sign.get(),
                        "x": self.map_engine.G.nodes[closest].get('x', mx),
                        "y": self.map_engine.G.nodes[closest].get('y', my),
                        "status": "⏳ PENDING"
                    })
                    self.ui.log_event(f"Placed '{self.ui.var_sign.get()}' at node {closest}.", "SUCCESS")
            self.calc_path()
            self.render_map()

    def calc_path(self):
        if not self.start_node or not self.end_node:
            return
        self.path_signs = []; self.visited_path_nodes.clear()
        self.path = self.map_engine.calc_path_nodes(self.start_node, self.end_node, self.pass_node)

        if len(self.path) >= 2:
            n1, n2 = self.path[0], self.path[1]
            x1, y1 = float(self.map_engine.G.nodes[n1].get('x', 0)), float(self.map_engine.G.nodes[n1].get('y', 0))
            x2, y2 = float(self.map_engine.G.nodes[n2].get('x', 0)), float(self.map_engine.G.nodes[n2].get('y', 0))
            self.car_yaw = math.atan2(y2 - y1, x2 - x1)
            self.car_x, self.car_y = x1, y1

        for node in self.path:
            for s in self.map_engine.signs:
                if s['node'] == node:
                    s['status'] = "⏳ PENDING"
                    self.path_signs.append(s)
        self.ui.update_sign_table(self.path_signs)

    def align_car_to_path(self):
        if not self.path or len(self.path) < 2:
            return
        min_dist = 999.0; best_dx, best_dy = 0, 0
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            ux, uy = float(self.map_engine.G.nodes[u].get('x', 0)), float(self.map_engine.G.nodes[u].get('y', 0))
            vx, vy = float(self.map_engine.G.nodes[v].get('x', 0)), float(self.map_engine.G.nodes[v].get('y', 0))
            dx, dy = vx - ux, vy - uy
            l2 = dx * dx + dy * dy
            if l2 == 0:
                continue
            t = max(0, min(1, ((self.car_x - ux) * dx + (self.car_y - uy) * dy) / l2))
            proj_x, proj_y = ux + t * dx, uy + t * dy
            d = math.hypot(self.car_x - proj_x, self.car_y - proj_y)
            if d < min_dist:
                min_dist, best_dx, best_dy = d, dx, dy
        if best_dx != 0 or best_dy != 0:
            self.car_yaw = math.atan2(best_dy, best_dx)

    def clear_route(self):
        self.start_node = None; self.end_node = None; self.pass_node = None; self.path = []
        self.visited_path_nodes.clear(); self.acted_signs.clear()
        for item in self.ui.tree.get_children():
            self.ui.tree.delete(item)
        self.render_map(); self.ui.log_event("🗑 Route & sign history cleared. Ready for new run.", "WARN")

    # ─────────────────────────────────────────────────────────
    # STATE UPDATES AND LOOPS
    # ─────────────────────────────────────────────────────────
    def render_map(self):
        pil = self.map_engine.render_map(
            self.car_x, self.car_y, self.car_yaw,
            self.path, self.visited_path_nodes, self.path_signs,
            True, self.start_node, self.pass_node, self.end_node
        )
        self.tk_map = ImageTk.PhotoImage(pil)
        self.ui.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map)
        self.ui.map_canvas.config(scrollregion=self.ui.map_canvas.bbox(tk.ALL))

    # ─────────────────────────────────────────────────────────
    # CAMERA UPDATE  — AI detects traffic-light bbox only;
    #                  OpenCV determines the colour inside it.
    # ─────────────────────────────────────────────────────────
    def update_camera(self):
        frame = None
        if self.picam2:
            try:
                frame = self.picam2.capture_array()
            except Exception:
                pass
        else:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "NO CAM", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if frame is not None:
            # Normalise to BGR
            if frame.ndim == 3 and frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.latest_raw_bgr = bgr.copy()

            if _AI_AVAILABLE and self.traffic_engine:
                res = self.traffic_engine.process(bgr)

                # ── Store traffic_mult for controller.compute() ───────────
                self._last_traffic_mult = res.speed_multiplier

                # ── Reset per-frame detection state ───────────────────────
                prev_visible = set(self.visible_signs.keys())
                self.visible_signs.clear()
                self.active_labels_current = list(res.active_labels)
                if hasattr(self, '_det_log_set'):
                    self._det_log_set -= prev_visible

                # ── Traffic light colour from traffic_module ──────────────
                # res.light_status: "[RED] HALT" / "[YELLOW] CAUTION" /
                #                   "[GREEN] GO"  / "NONE"
                # Keep last known colour when light is not seen (don't reset)
                # so the car stays stopped at a red light between frames.
                if "RED" in res.light_status:
                    self.current_light_status = "RED"
                elif "YELLOW" in res.light_status:
                    self.current_light_status = "YELLOW"
                elif "GREEN" in res.light_status:
                    self.current_light_status = "GREEN"

                # ── Pedestrian state from traffic_module ──────────────────
                self._pedestrian_blocking = res.pedestrian_blocking

                # ── Populate visible_signs from raw YOLO bbox detections ──
                # Uses accurate per-class pinhole model (estimate_distance_m).
                # Cross-checks both bbox height and width; weighted average.
                raw_dets = self.yolo.get_detections() if self.yolo else []
                for det in raw_dets:
                    lbl = det.get("label", "").lower()
                    if not lbl:
                        continue
                    x1, y1, x2, y2 = det["bbox"]
                    box_h = max(y2 - y1, 1)

                    # Accurate distance: pinhole model with per-class real sizes
                    dist_m = estimate_distance_m(lbl, x1, y1, x2, y2)

                    # Skip tiny detections
                    if box_h < SIGN_MIN_BBOX_H_PX:
                        continue

                    # Only register within detection window
                    if dist_m <= self.var_detect_dist.get():
                        if lbl not in self.visible_signs or dist_m < self.visible_signs[lbl]['dist']:
                            self.visible_signs[lbl] = {
                                'h': box_h, 'dist': dist_m,
                                'bbox': (x1, y1, x2, y2)
                            }
                            if not hasattr(self, '_det_log_set'):
                                self._det_log_set = set()
                            if lbl not in self._det_log_set:
                                self._det_log_set.add(lbl)
                                self.ui.log_event(
                                    f"👁 [{lbl}] entered zone at "
                                    f"~{dist_m:.2f}m (h={box_h}px w={x2-x1}px)", "INFO")

                # ── Nearest sign / traffic state log (throttled 2 s) ─────
                if res.sign_approach_m < 99.0:
                    if not hasattr(self, '_sign_dist_log_t') or \
                       time.time() - self._sign_dist_log_t > 2.0:
                        self._sign_dist_log_t = time.time()
                        self.ui.log_event(
                            f"🚦 TDE: {res.state} | {res.reason} | "
                            f"nearest={res.sign_approach_m:.1f}m "
                            f"mult={res.speed_multiplier:.2f}", "INFO")

                final_img = res.yolo_debug_frame
                self.ui.lbl_ai.config(
                    text=f"AI: {res.state} | {res.reason[:20]}", fg="cyan")
            else:
                final_img = frame
                self.ui.lbl_ai.config(text="AI: N/A")

            if final_img is not None:
                final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                cw, ch = self.ui.cam_label.winfo_width(), self.ui.cam_label.winfo_height()
                img = Image.fromarray(final_img).resize(
                    (cw if cw > 20 else 440, ch if ch > 20 else 330))
                imgtk = ImageTk.PhotoImage(image=img)
                self.ui.cam_label.imgtk = imgtk
                self.ui.cam_label.configure(image=imgtk, text="")

        self.root.after(50, self.update_camera)

    # ─────────────────────────────────────────────────────────
    # ADAS RULE ENGINE
    # ─────────────────────────────────────────────────────────
    def _set_sign_status(self, path_sign, new_status):
        """
        Central helper: update a sign's status, refresh the UI table,
        and write a log entry for every status transition.
        Status values:
            ⏳ PENDING   – on path, not yet encountered
            🟡 APPROACH  – detected in camera, distance > trigger
            🟢 ACTING    – within trigger distance, response active
            ✅ CONFIRMED – response fully completed
            ⚠️ MISSED    – passed without detection
        """
        old = path_sign.get('status', '')
        if old == new_status:
            return  # no change – skip update & log spam
        path_sign['status'] = new_status
        sign_type = path_sign.get('type', '?')
        node      = path_sign.get('node', '?')

        status_msgs = {
            "⏳ PENDING":   ("INFO",    f"[{sign_type}] @ node {node} → PENDING"),
            "🔵 DETECTING": ("INFO",    f"[{sign_type}] @ node {node} → DETECTING at {path_sign.get('_dist','?')}m (watching…)"),
            "🟡 APPROACH":  ("INFO",    f"[{sign_type}] @ node {node} → APPROACHING at {path_sign.get('_dist','?')}m"),
            "🟢 ACTING":    ("WARN",    f"[{sign_type}] @ node {node} → ACTING (response engaged)"),
            "✅ CONFIRMED": ("SUCCESS", f"[{sign_type}] @ node {node} → CONFIRMED ✅"),
            "⚠️ MISSED":    ("WARN",    f"[{sign_type}] @ node {node} → MISSED ⚠️"),
        }
        if new_status in status_msgs:
            lvl, msg = status_msgs[new_status]
            self.ui.log_event(msg, lvl)
        self.ui.update_sign_table(self.path_signs)

    def apply_adas_rules(self):
        """
        ADAS Rule Engine – v3
        ══════════════════════════════════════════════════════════
        • Indicators (active_state_flags) are set AND held every
          cycle for as long as the condition is true – they are
          never prematurely cleared by the rule engine.
        • Sign status transitions: PENDING → APPROACH → ACTING →
          CONFIRMED, logged at every step via _set_sign_status().
        • Traffic light: RED holds full stop every cycle until
          OpenCV sees GREEN.  YELLOW slows every cycle.
        • All overrides apply in both MANUAL and AUTONOMOUS modes.
        ══════════════════════════════════════════════════════════
        """
        current_time = time.time()
        self.ai_override_speed = None

        # ══════════════════════════════════════════════════════
        # 1. TIMED HARD-STOP  (stop-sign / noentry-sign timer)
        #    Keep the indicator ON for the full duration of the timer.
        # ══════════════════════════════════════════════════════
        if current_time < self.stop_until:
            self.ai_override_speed = 0
            # Decide which indicator to light based on what triggered the stop
            if "noentry-stop" in self.acted_signs:
                self.active_state_flags["no_entry"] = True
            else:
                self.active_state_flags["stop_sign"] = True
            remaining = self.stop_until - current_time
            # Log a countdown tick every full second
            tick = int(remaining)
            tick_key = f"stop_tick_{tick}"
            if tick_key not in self.acted_signs and tick > 0:
                self.acted_signs.add(tick_key)
                self.ui.log_event(f"⏱ Stopped – resuming in {tick}s …", "INFO")
            return

        # ══════════════════════════════════════════════════════
        # 2. SLOW TIMER  (crosswalk / priority / roundabout)
        #    Keep caution indicator ON while the timer runs.
        # ══════════════════════════════════════════════════════
        if current_time < self.slow_until:
            self.ai_override_speed = getattr(self, '_slow_speed', 80)
            self.active_state_flags["caution"] = True
            remaining = self.slow_until - current_time
            tick = int(remaining)
            tick_key = f"slow_tick_{tick}"
            if tick_key not in self.acted_signs and tick > 0:
                self.acted_signs.add(tick_key)
                self.ui.log_event(f"⏱ Caution speed – resuming full speed in {tick}s …", "INFO")

        # ══════════════════════════════════════════════════════
        # 3. DYNAMIC OBSTACLES  (bypass map path – always active)
        # ══════════════════════════════════════════════════════

        # — Pedestrian: combine traffic_module's PedestrianCrosswalkMonitor
        #   (_pedestrian_blocking from res.pedestrian_blocking) with direct
        #   bbox distance check as backup. Either triggers a full stop.
        ped_close    = ("pedestrian" in self.visible_signs and
                        self.visible_signs["pedestrian"]['dist'] <= self.var_trigger_dist.get())
        ped_blocking = getattr(self, '_pedestrian_blocking', False) or ped_close
        if ped_blocking:
            self.ai_override_speed = 0
            self.active_state_flags["pedestrian"] = True
            if "pedestrian" not in self.acted_signs:
                self.ui.log_event("🚶 PEDESTRIAN detected! Full stop until clear.", "CRITICAL")
                self.acted_signs.add("pedestrian")
            return  # Pedestrian always highest priority – skip everything else
        else:
            if "pedestrian" in self.acted_signs:
                self.ui.log_event("🚶 Pedestrian cleared. Resuming.", "SUCCESS")
            self.acted_signs.discard("pedestrian")

        # — Car ahead: slow every cycle while in frame —
        if "car" in self.visible_signs and self.visible_signs["car"]['dist'] <= self.var_trigger_dist.get():
            _slow = getattr(self, '_slow_speed', 80)
            if self.ai_override_speed is None or self.ai_override_speed > _slow:
                self.ai_override_speed = _slow
            self.active_state_flags["caution"] = True
            if "car" not in self.acted_signs:
                self.ui.log_event(f"🚗 Car ahead at {self.visible_signs['car']['dist']:.1f}m – slowing to {getattr(self,'_slow_speed',80):.0f}.", "WARN")
                self.acted_signs.add("car")
        else:
            if "car" in self.acted_signs:
                self.ui.log_event("🚗 Car obstacle cleared. Resuming speed.", "INFO")
            self.acted_signs.discard("car")

        if not self.visible_signs:
            return

        # ══════════════════════════════════════════════════════
        # 4. TRAFFIC LIGHT  (persistent per-colour check)
        #    Indicator and override are re-applied EVERY cycle
        #    so the car cannot creep forward between frames.
        # ══════════════════════════════════════════════════════
        if "traffic-light" in self.visible_signs:
            tl_dist = self.visible_signs["traffic-light"]['dist']
            if tl_dist <= self.var_trigger_dist.get():

                if self.current_light_status == "RED":
                    self.ai_override_speed = 0
                    self.active_state_flags["red_light"]  = True
                    self.active_state_flags["caution"]    = True
                    if "tl-red" not in self.acted_signs:
                        self.ui.log_event(f"🔴 RED light at {tl_dist:.1f}m – FULL STOP until GREEN.", "CRITICAL")
                        self.acted_signs.add("tl-red")
                        self.acted_signs.discard("tl-yellow")
                        self.acted_signs.discard("tl-green")
                        # Update traffic-light sign status to ACTING
                        for s in self.path_signs:
                            if s['type'].lower() == "traffic-light":
                                s['_dist'] = f"{tl_dist:.1f}"
                                self._set_sign_status(s, "🟢 ACTING")
                    return  # RED overrides everything below

                elif self.current_light_status == "YELLOW":
                    _slow = getattr(self, '_slow_speed', 80)
                    if self.ai_override_speed is None or self.ai_override_speed > _slow:
                        self.ai_override_speed = _slow
                    self.active_state_flags["yellow_light"] = True
                    self.active_state_flags["caution"]      = True
                    if "tl-yellow" not in self.acted_signs:
                        self.ui.log_event(f"🟡 YELLOW light at {tl_dist:.1f}m – slowing to {getattr(self,'_slow_speed',80):.0f}.", "WARN")
                        self.acted_signs.add("tl-yellow")
                        self.acted_signs.discard("tl-red")
                        self.acted_signs.discard("tl-green")
                        for s in self.path_signs:
                            if s['type'].lower() == "traffic-light":
                                s['_dist'] = f"{tl_dist:.1f}"
                                self._set_sign_status(s, "🟢 ACTING")

                elif self.current_light_status == "GREEN":
                    self.active_state_flags["green_light"] = True
                    if "tl-green" not in self.acted_signs:
                        self.ui.log_event(f"🟢 GREEN light – proceeding at full speed.", "SUCCESS")
                        self.acted_signs.add("tl-green")
                        self.acted_signs.discard("tl-red")
                        self.acted_signs.discard("tl-yellow")
                        for s in self.path_signs:
                            if s['type'].lower() == "traffic-light" and s.get('status') != "✅ CONFIRMED":
                                self._set_sign_status(s, "✅ CONFIRMED")

        # ══════════════════════════════════════════════════════
        # 5. MAP-BASED PATH SIGNS
        #    Full status lifecycle: PENDING → APPROACH → ACTING → CONFIRMED
        # ══════════════════════════════════════════════════════
        pending_signs = [s for s in self.path_signs
                         if s.get('status') not in ["✅ CONFIRMED", "⚠️ MISSED"]]

        for path_sign in pending_signs:
            sign_type = path_sign['type'].lower()

            if sign_type == "traffic-light":
                continue  # handled entirely in section 4

            # ── Distance-gated status phases ─────────────────────────────
            # Signs here are within detect distance (var_detect_dist).
            if sign_type in self.visible_signs:
                dist_m = self.visible_signs[sign_type]['dist']
                path_sign['_dist'] = f"{dist_m:.1f}"

                if dist_m > self.var_trigger_dist.get():
                    # Between 5 m and 2 m: DETECTING – sign seen, not yet active
                    cur_status = path_sign.get('status', '')
                    if cur_status in ["⏳ PENDING", "🔵 DETECTING"]:
                        self._set_sign_status(path_sign, "🔵 DETECTING")
                    elif cur_status == "🟡 APPROACH":
                        pass  # already advanced – keep it
                    self.ui.log_event(
                        f"👁 [{sign_type}] detected at {dist_m:.1f}m – monitoring …", "INFO"
                    ) if path_sign.get('_last_detect_log', 999) - dist_m > 0.5 else None
                    path_sign['_last_detect_log'] = dist_m
                    continue  # not close enough to act yet

                # ── Within trigger distance → ACTING ────────────────────
                sign_node_key = f"act_{sign_type}_{path_sign.get('node','?')}"

                if path_sign.get('status') not in ["🟢 ACTING", "✅ CONFIRMED"]:
                    self._set_sign_status(path_sign, "🟢 ACTING")

                self.active_state_flags["caution"] = True

                # ── STOP SIGN ───────────────────────────────────────────
                if sign_type == "stop-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.stop_until = current_time + 3.0
                        self.ai_override_speed = 0
                        self.active_state_flags["stop_sign"] = True
                        self.ui.log_event(f"🛑 STOP SIGN at {dist_m:.1f}m – halting for 3 seconds.", "WARN")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        # Timer is running – keep indicator on
                        self.ai_override_speed = 0
                        self.active_state_flags["stop_sign"] = True

                # ── PARKING SIGN ─────────────────────────────────────────
                elif sign_type == "parking-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.ui.log_event(f"🅿️ PARKING SIGN at {dist_m:.1f}m – starting Auto-Park.", "SUCCESS")
                        self.start_csv_playback("car_actions.csv")
                        self.active_state_flags["park"] = True
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        self.active_state_flags["park"] = True

                # ── CROSSWALK ────────────────────────────────────────────
                elif sign_type == "crosswalk-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.slow_until = current_time + 5.0
                        self.ai_override_speed = getattr(self, '_slow_speed', 80)
                        self.ui.log_event(f"🚸 CROSSWALK at {dist_m:.1f}m – slowing for 5s.", "WARN")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        _slow = getattr(self, '_slow_speed', 80)
                        if self.ai_override_speed is None or self.ai_override_speed > _slow:
                            self.ai_override_speed = _slow

                # ── PRIORITY SIGN ────────────────────────────────────────
                elif sign_type == "priority-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.slow_until = current_time + 5.0
                        self.ai_override_speed = getattr(self, '_slow_speed', 80)
                        self.ui.log_event(f"⚠️ PRIORITY SIGN at {dist_m:.1f}m – slowing for 5s.", "WARN")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        _slow = getattr(self, '_slow_speed', 80)
                        if self.ai_override_speed is None or self.ai_override_speed > _slow:
                            self.ai_override_speed = _slow

                # ── ROUNDABOUT ───────────────────────────────────────────
                elif sign_type == "roundabout-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.slow_until = current_time + 10.0
                        self.ai_override_speed = getattr(self, '_slow_speed', 80)
                        self.ui.log_event(f"🔄 ROUNDABOUT SIGN at {dist_m:.1f}m – slowing for 10s.", "INFO")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        _slow = getattr(self, '_slow_speed', 80)
                        if self.ai_override_speed is None or self.ai_override_speed > _slow:
                            self.ai_override_speed = _slow

                # ── NO ENTRY ─────────────────────────────────────────────
                elif sign_type == "noentry-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.acted_signs.add("noentry-stop")  # tracks which stop type
                        self.stop_until = current_time + 30.0
                        self.ai_override_speed = 0
                        self.active_state_flags["no_entry"] = True
                        self.ui.log_event(f"⛔ NO ENTRY at {dist_m:.1f}m – stopping for 30s.", "CRITICAL")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        self.ai_override_speed = 0
                        self.active_state_flags["no_entry"] = True

                # ── HIGHWAY ENTRY ────────────────────────────────────────
                elif sign_type == "highway-entry-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.in_highway_mode = True
                        self.active_state_flags["highway"] = True
                        self.ui.log_event(f"🛣️ HIGHWAY ENTRY at {dist_m:.1f}m – speed +20% activated.", "SUCCESS")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")
                    else:
                        self.active_state_flags["highway"] = True  # keep indicator live

                # ── HIGHWAY EXIT ─────────────────────────────────────────
                elif sign_type == "highway-exit-sign":
                    if sign_node_key not in self.acted_signs:
                        self.acted_signs.add(sign_node_key)
                        self.in_highway_mode = False
                        self.active_state_flags["highway"] = False
                        self.ui.log_event(f"🏁 HIGHWAY EXIT at {dist_m:.1f}m – speed normalised.", "WARN")
                        self._set_sign_status(path_sign, "✅ CONFIRMED")

            else:
                # Sign is NOT visible this frame but was previously approaching
                # Don't reset to PENDING – just leave it until confirmed or missed
                pass

    # ─────────────────────────────────────────────────────────
    # CONTROL LOOP  (20 Hz)
    # ─────────────────────────────────────────────────────────
    def control_loop(self):
        now = time.time()
        dt = max(now - self.last_ctrl_time, 0.001)
        self.last_ctrl_time = now

        # ── Base speed ────────────────────────────────────────
        # In AUTONOMOUS mode: use the locked snapshot taken when auto started.
        # The dashboard slider is IGNORED while auto is active so the operator
        # cannot accidentally change the speed mid-run.
        # In MANUAL mode: use the live slider as normal.
        if self.is_auto_mode and self.auto_base_speed is not None:
            base_speed = self.auto_base_speed
        else:
            base_speed = self.ui.slider_base_speed.get()

        # Highway mode: +20% of current base_speed (relative, not fixed)
        if self.in_highway_mode:
            base_speed = base_speed * 1.20

        # Slow-zone speed = 80% of base_speed (−20%). Used by all caution responses.
        slow_speed = max(base_speed * 0.80, 10)
        self._slow_speed = slow_speed   # available to apply_adas_rules()

        map_sim_multiplier = self.ui.slider_sim_speed.get()
        steer_multiplier   = self.ui.slider_steer_mult.get()

        # Reset ALL flags to False each cycle.
        # apply_adas_rules() (called below) re-sets whichever flags are
        # currently active.  Highway is a persistent vehicle state so
        # restore it immediately so the indicator stays lit between cycles.
        for key in self.active_state_flags:
            self.active_state_flags[key] = False
        if self.in_highway_mode:
            self.active_state_flags["highway"] = True

        target_speed, target_steer = 0, 0
        perc = None
        ctrl = MockCtrl()
        ctrl.anchor = "MANUAL"; ctrl.steer_angle_deg = 0.0
        ctrl.lookahead_px = 150; ctrl.target_x = 320.0

        # ── CONTINUOUS PERCEPTION (lane detection) ────────────
        # vision.process() is called with extra_offset_px=0 so perception
        # computes the raw lane fits without any pre-baked offset.
        # We then override target_x ourselves using right-line priority:
        #
        #   TWO lines visible  → true centre = (sl_x + sr_x) / 2
        #   RIGHT line only    → centre from right = sr_x − lane_width/2
        #                        (car stays safely off the right line)
        #   LEFT line only     → keep perception's own estimate (no sr anchor)
        #   Both lost          → keep perception's dead-reckoning estimate
        #
        # Lane-width is tracked with a slow EMA so reconstruction stays
        # stable when one line is temporarily missing.
        # Lane Offset slider adds a fine trim on top (default 0 = exact centre).
        if self.latest_raw_bgr is not None and self.vision:
            try:
                perc = self.vision.process(
                    self.latest_raw_bgr,
                    dt=dt,
                    extra_offset_px=0.0,   # we apply offset ourselves below
                    velocity_ms=max(self.current_speed / 1000.0, 0.0),
                    last_steering=self.current_steer,
                )
            except Exception:
                perc = None

            if perc is not None:
                # ── RIGHT-LINE PRIORITY TARGETING ──────────────────────
                Y_EVAL = 400.0

                def _ev(fit):
                    return float(np.polyval(fit, Y_EVAL))

                sr = perc.sr   # right solid line  (primary anchor)
                sl = perc.sl   # left line (solid on normal road, dashed on highway)
                lw = perc.lane_width_px

                # Update stable lane-width EMA whenever both lines are visible
                if sr is not None and sl is not None:
                    measured_w = _ev(sr) - _ev(sl)
                    if 150 < measured_w < 500:   # sanity bounds in BEV pixels
                        if not hasattr(self, '_lw_ema'):
                            self._lw_ema = measured_w
                        self._lw_ema = 0.9 * self._lw_ema + 0.1 * measured_w

                lw_stable = getattr(self, '_lw_ema', lw)

                # Fine-trim offset from slider (default 0)
                lane_offset_px = (self.var_lane_offset.get()
                                  if hasattr(self, 'var_lane_offset')
                                  else RIGHT_LANE_OFFSET_DEFAULT)

                if sr is not None and sl is not None:
                    # TWO lines: perfect mathematical centre
                    new_target = (_ev(sl) + _ev(sr)) * 0.5 + lane_offset_px
                    new_anchor = "CENTRE_DUAL"

                elif sr is not None:
                    # RIGHT line only: anchor to sr, stay half-lane-width left of it
                    new_target = _ev(sr) - lw_stable * 0.5 + lane_offset_px
                    new_anchor = "CENTRE_FROM_SR"

                else:
                    # Left only or both lost: trust perception's own result
                    new_target = perc.target_x + lane_offset_px
                    new_anchor = perc.anchor

                # Smooth with a responsive EMA (faster than perception's 0.8/0.2)
                if not hasattr(self, '_tgt_ema'):
                    self._tgt_ema = new_target
                self._tgt_ema = 0.35 * self._tgt_ema + 0.65 * new_target
                perc.target_x = self._tgt_ema
                perc.anchor   = new_anchor

                # Log lane state every 2 s
                if not hasattr(self, '_lane_log_t') or time.time() - self._lane_log_t > 2.0:
                    self._lane_log_t = time.time()
                    sr_s = f"sr={_ev(sr):.0f}px" if sr is not None else "sr=LOST"
                    sl_s = f"sl={_ev(sl):.0f}px" if sl is not None else "sl=LOST"
                    self.ui.log_event(
                        f"🛣 [{new_anchor}] {sl_s} {sr_s} "
                        f"lw={lw_stable:.0f}px → tgt={perc.target_x:.0f}px", "INFO")

                dbg_img = annotate_bev(perc, ctrl,
                                       self.is_calibrating, self.auto_start_time)
                cw_bev = self.ui.bev_label.winfo_width()
                ch_bev = self.ui.bev_label.winfo_height()
                img_bev = Image.fromarray(
                    cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB)
                ).resize((cw_bev if cw_bev > 20 else 440,
                          ch_bev if ch_bev > 20 else 330))
                imgtk_bev = ImageTk.PhotoImage(image=img_bev)
                self.ui.bev_label.imgtk = imgtk_bev
                self.ui.bev_label.configure(image=imgtk_bev, text="")

        # ══════════════════════════════════════════════════════════════
        # LOGIC COMMANDS
        # Priority order (highest → lowest):
        #   1. AUTO-PARK playback        – CSV controls everything
        #   2. ADAS rule engine          – computes speed override / flags
        #   3a. AUTONOMOUS driving       – controller.compute() for speed+steer
        #       └─ keyboard is BLOCKED in autonomous – cannot interfere
        #   3b. MANUAL driving           – keyboard only
        #   4. ADAS speed override       – applied as ABSOLUTE SET, not clamp
        # ══════════════════════════════════════════════════════════════

        if self.is_playing_back:
            # ── AUTO-PARK: CSV playback owns speed+steer completely ───────
            self.active_state_flags["park"] = True
            if self.playback_cmd is None:
                if self.playback_queue:
                    self.playback_cmd = self.playback_queue.pop(0)
                    self.playback_frames = 0
                else:
                    self.is_playing_back = False
                    self.is_parking_reverse_mode = False
                    # Immediately zero speed+steer so the car stops now
                    target_speed = 0
                    target_steer = 0
                    self.current_speed = 0.0
                    self.current_steer = 0.0
                    if self.is_connected:
                        self.handler.set_speed(0)
                        self.handler.set_steering(0)
                    self.ui.log_event("🅿️ Auto-Park sequence COMPLETE – stopped.", "SUCCESS")
            if self.is_playing_back and self.playback_cmd is not None:
                target_speed = self.playback_cmd["speed"]
                target_steer = self.playback_cmd["steering"]
                self.playback_frames += 1
                if self.playback_frames >= self.playback_cmd["duration"]:
                    self.playback_cmd = None

        elif self.is_auto_mode:
            # ══════════════════════════════════════════════════════════════
            # AUTONOMOUS MODE
            # Keyboard is COMPLETELY IGNORED here.
            # Speed and steering come ONLY from the lane controller.
            # ADAS overrides are applied as absolute commands afterwards.
            # ══════════════════════════════════════════════════════════════
            self.is_calibrating = False

            # Step 1: Run ADAS rule engine to determine any override
            if self.adas_enabled:
                self.apply_adas_rules()

            # Step 2: Get lane-following output from the controller
            if perc and self.controller:
                if time.time() - self.auto_start_time > 5.0:
                    if getattr(self, '_was_calibrating', False):
                        self.ui.log_event("✅ Calibration complete – AUTONOMOUS lane-following ACTIVE.", "SUCCESS")
                    self._was_calibrating = False

                    # traffic_mult from TrafficResult applies the module's
                    # stop/slow/approach speed multipliers to the controller output.
                    # perception.py already steers to the correct right-lane centre
                    # via its 3-tier targeting system — no extra correction needed.
                    ctrl = self.controller.compute(
                        perc_res=perc,
                        nav_state="NORMAL",
                        velocity_ms=max(self.current_speed / 1000.0, 0.0),
                        dt=dt,
                        base_speed=base_speed,
                        traffic_mult=self._last_traffic_mult,
                    )
                    target_speed = ctrl.speed_pwm
                    target_steer = ctrl.steer_angle_deg * steer_multiplier

                    # Log anchor / speed every 3 s
                    if not hasattr(self, '_lane_log_t') or time.time() - self._lane_log_t > 3.0:
                        self._lane_log_t = time.time()
                        self.ui.log_event(
                            f"🛣 Lane anchor={ctrl.anchor} "
                            f"tx={ctrl.target_x:.0f}px "
                            f"steer={ctrl.steer_angle_deg:.1f}° "
                            f"spd={ctrl.speed_pwm:.0f} "
                            f"t_mult={self._last_traffic_mult:.2f}", "INFO")
                else:
                    # Still in 5-second calibration warmup – hold still
                    self.is_calibrating = True
                    self._was_calibrating = True
                    target_speed, target_steer = 0, 0
                    if not getattr(self, '_was_calibrating', False):
                        self.ui.log_event("🤖 Calibrating … holding position for 5s.", "INFO")
            else:
                # No perception data this frame – hold last speed, hold steer
                # (don't slam to zero – that causes jerking)
                target_speed = self.current_speed
                target_steer = self.current_steer
                if not hasattr(self, '_no_perc_log_t') or time.time() - self._no_perc_log_t > 3.0:
                    self._no_perc_log_t = time.time()
                    self.ui.log_event("⚠️ No perception data – holding last speed/steer.", "WARN")

            # ── Step 3: Apply ADAS override (AUTONOMOUS – absolute command) ──
            # ai_override_speed is set by apply_adas_rules() above.
            #   == 0   → FULL STOP  (stop-sign timer, red light, pedestrian, no-entry)
            #   > 0    → SPEED CAP  (caution zone: crosswalk, roundabout, car ahead)
            #   None   → no override – controller runs at full base_speed
            if self.ai_override_speed is not None and self.adas_enabled:
                if self.ai_override_speed == 0:
                    # ABSOLUTE STOP: override both speed and steer completely.
                    # Car stops and waits. Steer zeroed so it goes straight on resume.
                    target_speed = 0
                    target_steer = 0
                    if not hasattr(self, '_adas_log_t') or time.time() - self._adas_log_t > 1.5:
                        self._adas_log_t = time.time()
                        self.ui.log_event("⚙️ ADAS [AUTO] FULL STOP – speed=0, steer=0.", "WARN")
                else:
                    # SPEED CAP: clamp speed to override limit, keep lane steer intact.
                    direction    = 1 if target_speed >= 0 else -1
                    target_speed = min(abs(target_speed), self.ai_override_speed) * direction
                    # Do NOT touch target_steer – lane tracking must continue
                    if not hasattr(self, '_adas_log_t') or time.time() - self._adas_log_t > 1.5:
                        self._adas_log_t = time.time()
                        self.ui.log_event(
                            f"⚙️ ADAS [AUTO] SLOW → speed capped at {self.ai_override_speed}, "
                            f"steer held for lane tracking.", "WARN")

        else:
            # ══════════════════════════════════════════════════════════════
            # MANUAL MODE  – keyboard controls speed and steer
            # ══════════════════════════════════════════════════════════════
            self.is_calibrating = False

            # Run ADAS in manual too
            if self.adas_enabled:
                self.apply_adas_rules()

            target_speed = (base_speed  if self.keys['Up']
                            else (-base_speed if self.keys['Down'] else 0))
            target_steer = (-25 * steer_multiplier if self.keys['Left']
                            else (25 * steer_multiplier if self.keys['Right'] else 0))

            # ── ADAS override in manual mode ─────────────────────────────
            if self.ai_override_speed is not None and self.adas_enabled:
                direction    = 1 if target_speed >= 0 else -1
                target_speed = min(abs(target_speed), self.ai_override_speed) * direction
                if self.ai_override_speed == 0:
                    target_speed = 0  # immediate hard stop in manual too
                    target_steer = 0
                if not hasattr(self, '_adas_log_t') or time.time() - self._adas_log_t > 1.5:
                    self._adas_log_t = time.time()
                    action = "FULL STOP" if self.ai_override_speed == 0 else f"SLOW → {self.ai_override_speed}"
                    self.ui.log_event(
                        f"⚙️ ADAS [MANUAL] override: {action}", "WARN")

        # ── SMOOTH APPLICATION ────────────────────────────────
        # Immediate snap to zero: no coasting when target is 0
        # (no key pressed, ADAS hard-stop, red light, etc.)
        # Ramp smoothing only applies when actually accelerating/steering.
        if target_speed == 0:
            self.current_speed = 0.0          # instant stop – no coast
        else:
            self.current_speed += (target_speed - self.current_speed) * 0.2

        if target_steer == 0:
            self.current_steer = 0.0          # instant steer centre
        else:
            self.current_steer += (target_steer - self.current_steer) * 0.15

        # ── HARDWARE OUTPUT ───────────────────────────────────
        if self.is_connected:
            self.handler.set_speed(int(self.current_speed))
            self.handler.set_steering(self.current_steer)
            hz = 1.0 / dt if dt > 0 else 0.0
            self.current_hz = 0.8 * self.current_hz + 0.2 * hz
            self.ui.lbl_hz.config(text=f"{self.current_hz:.1f} Hz", fg="cyan")
            if hasattr(self.handler, 'set_light_state'):
                if self.active_state_flags.get("park"):
                    self.handler.set_light_state("PARKING", True)
                elif self.active_state_flags.get("caution"):
                    self.handler.set_light_state("CAUTION", True)
                else:
                    self.handler.set_light_state("ALL", False)
        else:
            self.ui.lbl_hz.config(text="OFFLINE", fg="gray")
            # Offline simulation: only keyboard in MANUAL mode drives the map dot.
            # In AUTONOMOUS mode we still use whatever target_speed/steer the
            # controller computed above – keyboard is blocked even offline.
            if not self.is_auto_mode:
                target_speed = (base_speed if self.keys['Up']
                                else (-base_speed if self.keys['Down'] else 0))
                target_steer = (-25 * steer_multiplier if self.keys['Left']
                                else (25 * steer_multiplier if self.keys['Right'] else 0))
            self.current_speed = target_speed   # direct set offline for map sim
            self.current_steer = target_steer

        # ── KINEMATICS & MAGNETIC PATH PULL ──────────────────
        if abs(self.current_speed) < 1:  self.current_speed = 0
        if abs(self.current_steer) < 0.5: self.current_steer = 0

        v_ms = (self.current_speed / 1000.0) * map_sim_multiplier * 1.5
        steer_rad = math.radians(self.current_steer)
        self.car_yaw -= (v_ms / max(WHEELBASE_M, 0.01)) * math.tan(steer_rad) * dt
        self.car_yaw  = (self.car_yaw + math.pi) % (2 * math.pi) - math.pi
        self.car_x   += v_ms * math.cos(self.car_yaw) * dt
        self.car_y   += v_ms * math.sin(self.car_yaw) * dt

        # Lock car to path edges
        if not (self.is_playing_back or self.is_waiting_to_reverse):
            edges = ([(self.path[i], self.path[i + 1]) for i in range(len(self.path) - 1)]
                     if self.path else list(self.map_engine.G.edges()))
            if edges:
                min_dist = 999.0
                best_p, best_dx, best_dy = (self.car_x, self.car_y), 0, 0
                for u, v in edges:
                    ux = float(self.map_engine.G.nodes[u].get('x', 0))
                    uy = float(self.map_engine.G.nodes[u].get('y', 0))
                    vx = float(self.map_engine.G.nodes[v].get('x', 0))
                    vy = float(self.map_engine.G.nodes[v].get('y', 0))
                    dx, dy = vx - ux, vy - uy
                    l2 = dx * dx + dy * dy
                    if l2 == 0:
                        continue
                    t = max(0.0, min(1.0,
                        ((self.car_x - ux) * dx + (self.car_y - uy) * dy) / l2))
                    proj_x, proj_y = ux + t * dx, uy + t * dy
                    d = math.hypot(self.car_x - proj_x, self.car_y - proj_y)
                    if d < min_dist:
                        min_dist, best_p, best_dx, best_dy = d, (proj_x, proj_y), dx, dy

                if self.path or min_dist < 2.0:
                    self.car_x += (best_p[0] - self.car_x) * 0.95
                    self.car_y += (best_p[1] - self.car_y) * 0.95
                    if best_dx != 0 or best_dy != 0:
                        path_yaw = math.atan2(best_dy, best_dx)
                        diff = (path_yaw - self.car_yaw + math.pi) % (2 * math.pi) - math.pi
                        if abs(diff) > math.pi / 2:
                            path_yaw = math.atan2(-best_dy, -best_dx)
                            diff = (path_yaw - self.car_yaw + math.pi) % (2 * math.pi) - math.pi
                        self.car_yaw += diff * 0.5

        # ── TELEMETRY DISPLAY ─────────────────────────────────
        mode_str = ("AUTO-PARK" if self.is_playing_back
                    else ("AUTONOMOUS" if self.is_auto_mode else "MANUAL"))
        if self.is_calibrating:
            mode_str = "CALIBRATING..."
        tl_str = f" | TL:{self.current_light_status}" if self.current_light_status != "NONE" else ""
        self.ui.lbl_telemetry.config(
            text=f"SPD: {int(self.current_speed)} | STR: {self.current_steer:.1f}° "
                 f"| LMT: {base_speed} | [{mode_str}]{tl_str}"
        )
        self.ui.update_indicator_dots(self.active_state_flags, self.adas_enabled, self.is_playing_back)

        self.render_map()
        self.root.after(50, self.control_loop)

    # ─────────────────────────────────────────────────────────
    # CONFIGS & CLOSURE
    # ─────────────────────────────────────────────────────────
    def save_config(self): pass
    def load_config(self): pass

    def _on_key_press(self, e):
        # Block keyboard driving entirely in autonomous and playback modes
        if self.is_auto_mode or self.is_playing_back:
            return
        if e.keysym in self.keys:
            self.keys[e.keysym] = True

    def _on_key_release(self, e):
        if e.keysym in self.keys:
            self.keys[e.keysym] = False

    def on_close(self):
        if self.picam2:
            self.picam2.stop()
        if self.yolo:
            self.yolo.stop()
        if self.is_connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BFMC_App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

#Good code
