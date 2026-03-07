import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import os
import json
import math
import time
import networkx as nx
import csv

# --- AUTOMATION IMPORTS ---
try:
    from perception import VisionPipeline
    from control import Controller
    _AUTO_DRIVE_AVAILABLE = True
except ImportError:
    _AUTO_DRIVE_AVAILABLE = False
    print("WARNING: perception.py or control.py not found. Autonomous driving disabled.")

# --- CONFIGURATION & CONSTANTS ---
SVG_FILE = "Track.svg"
GRAPH_FILE = "Competition_track_graph.graphml"
SIGNS_DB_FILE = "signs_database.json"
CONFIG_FILE = "dashboard_config.json"
YOLO_MODEL_FILE = "Niranjan.pt"

FINAL_SCALE_X = 1.0640
FINAL_SCALE_Y = 1.0890
FINAL_OFF_X   = 0
FINAL_OFF_Y   = 0

DEFAULT_START_X  = 1.0  
DEFAULT_START_Y  = 1.0  

HIGHWAY_SPEED_PWM = 400   

CAMERA_FOCAL_LENGTH_PX = 450.0 
REAL_SIGN_HEIGHT_M = 0.08      
REAL_WIDTH_M  = 22.0
REAL_HEIGHT_M = 15.0

# --- LANE TRACKING PHYSICAL CONSTANTS ---
WHEELBASE_M          = 0.23    
LANE_WIDTH_M         = 0.35    
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])
RIGHT_LANE_OFFSET_PX = 70
DUAL_OFFSET_PX       = 0
SINGLE_DIV_OFFSET_PX = 40
SINGLE_EDGE_OFFSET_PX = -40
LOST_GRACE_FRAMES    = 8

THEME = {
    "bg": "#1e1e1e", "panel": "#252526", "canvas": "#111111", 
    "fg": "#cccccc", "accent": "#007acc", "danger": "#f44336", "success": "#4caf50",
    "warning": "#ff9800", "font_h": ("Segoe UI", 11, "bold"), "font_p": ("Segoe UI", 9),
    "sash": "#333333"
}

SIGN_MAP = {
    "stop-sign": {"name": "Stop", "emoji": "🛑"},
    "crosswalk-sign": {"name": "Crosswalk", "emoji": "🚶"},
    "priority-sign": {"name": "Priority", "emoji": "🔶"},
    "parking-sign": {"name": "Parking", "emoji": "🅿️"},
    "highway-entry-sign": {"name": "Hwy Entry", "emoji": "⬆️"},
    "highway-exit-sign": {"name": "Hwy Exit", "emoji": "↗️"},
    "pedestrian": {"name": "Pedestrian", "emoji": "🚸"},
    "traffic-light": {"name": "Light", "emoji": "🚦"},
    "roundabout-sign": {"name": "Roundabout", "emoji": "🔄"},
    "oneway-sign": {"name": "Oneway", "emoji": "⬆️"},
    "noentry-sign": {"name": "No Entry", "emoji": "⛔"},
    "car": {"name": "Car", "emoji": "🚙"}
}

# --- HARDWARE & AI IMPORTS ---
try:
    from serial_handler import STM32_SerialHandler
except ImportError:
    class STM32_SerialHandler:
        def connect(self): return True
        def disconnect(self): pass
        def set_speed(self, s): pass
        def set_steering(self, s): pass

try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    _CAM_AVAILABLE = False

try:
    from traffic_module import TrafficDecisionEngine, ThreadedYOLODetector
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False
    print("traffic_module.py not found - running without YOLO")


# ===========================================================================
# BUILT-IN AUTONOMOUS DRIVING MODULES
# ===========================================================================

class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.50
    STALE_FIT_FRAMES = 5

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

    def reset(self):
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None   
        self.sr         = None   
        self.left_stale  = 0
        self.right_stale = 0

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
            self.sl        = self._ema(self.sl, fl)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            self.sr         = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit  = None
                    self.sl        = None
                    self.left_stale = self.STALE_FIT_FRAMES  
                    has_l          = False
                else:
                    self.right_fit  = None
                    self.sr         = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r           = False

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, nav_state="NORMAL"):
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit): return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None: return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None: return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr is not None: return ev(sr) - hw + extra_offset_px, "JCT_EDGE"
            if sl is not None: return ev(sl) + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX + extra_offset_px, "DUAL"

        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + SINGLE_EDGE_OFFSET_PX + extra_offset_px, "GHOST_L"

        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + SINGLE_DIV_OFFSET_PX + extra_offset_px, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)

        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN
        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

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
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            xl0 = max(0, lx - self.SW_MARGIN); xl1 = min(self.w, lx + self.SW_MARGIN)
            xr0 = max(0, rx - self.SW_MARGIN); xr1 = min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]
            li.append(gl); ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(li); ri = np.concatenate(ri)
        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)

        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 80 < w < 560

    def _ema(self, prev, new):
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


class JunctionDetector:
    ENTRY_FRAMES       = 5
    EXIT_FRAMES        = 8
    CROSS_ENERGY_RATIO = 1.4
    WIDTH_RATIO_HIGH   = 1.6
    MIN_BOT_ENERGY     = 500

    def __init__(self):
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0

    def update(self, warped_binary, left_conf, right_conf, left_fit, right_fit, lane_width_px):
        h, w = warped_binary.shape
        both_lost = (left_conf < 200) and (right_conf < 200)
        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        hist_bot = float(np.sum(warped_binary[h // 2:, :]))

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

        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15:
                self.state       = "NORMAL"
                self.entry_count = 0

        return self.state


class RoundaboutNavigator:
    ENTRY_WIDTH_RATIO  = 0.60
    EXIT_WIDTH_RATIO   = 0.82
    MIN_CIRCLE_FRAMES  = 25
    MAX_CIRCLE_FRAMES  = 120

    def __init__(self):
        self.state  = "NORMAL"
        self.frames = 0

    def update(self, left_fit, right_fit, lane_width_px, img_h=480):
        y = img_h - 50
        if left_fit is not None and right_fit is not None:
            lx    = np.polyval(left_fit,  y)
            rx    = np.polyval(right_fit, y)
            ratio = (rx - lx) / max(float(lane_width_px), 1.0)

            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.state  = "ROUNDABOUT"
                    self.frames = 0
            elif self.state == "ROUNDABOUT":
                self.frames += 1
                normal_exit = (self.frames > self.MIN_CIRCLE_FRAMES and ratio > self.EXIT_WIDTH_RATIO)
                timeout_exit = self.frames > self.MAX_CIRCLE_FRAMES
                if normal_exit or timeout_exit:
                    self.state  = "NORMAL"
                    self.frames = 0
        elif self.state == "ROUNDABOUT":
            self.frames += 1
            if self.frames > self.MAX_CIRCLE_FRAMES:
                self.state  = "NORMAL"
                self.frames = 0
        return self.state


class DividerGuard:
    DIVIDER_SAFE_PX = 55    
    EDGE_SAFE_PX    = 50    
    GAIN            = 0.09  
    MAX_CORR        = 8.0   
    DEADBAND_PX     = 5

    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction  = 0.0
        speed_scale = 1.0
        triggered   = False

        div_corr = 0.0
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x          
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)   
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x        
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)  
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# =============================================================================
# MAIN APP CLASS
# =============================================================================
class BFMC_Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("BFMC Ultimate Command")
        self.root.geometry("1450x850")
        self.root.minsize(1350, 800) # Hard guardrail to prevent text vanishing from UI squish
        self.root.configure(bg=THEME["bg"])
        
        # SHARED STATE
        self.handler = STM32_SerialHandler()
        self.is_connected = False
        
        # Create a mock car_actions.csv for testing Auto-Parking if not present
        if not os.path.exists("car_actions.csv"):
            try:
                with open("car_actions.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["speed", "steering", "pwm", "direction", "duration_frames"])
                    writer.writerow([150, 20, 150, 1, 40])
                    writer.writerow([0, 0, 0, 1, 15])
                    writer.writerow([-150, -20, 150, -1, 40])
                    writer.writerow([0, 0, 0, 1, 10])
            except Exception:
                pass
        
        # AI Engine
        self.traffic_engine = None
        self.yolo = None
        if _AI_AVAILABLE:
            self.yolo = ThreadedYOLODetector(YOLO_MODEL_FILE) 
            self.traffic_engine = TrafficDecisionEngine(self.yolo)
            
        # Physics & State
        self.car_x = 0.5; self.car_y = 0.5; self.car_yaw = 0.0
        self.current_speed = 0.0; self.current_steer = 0.0
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.last_ctrl_time = time.time()
        self.current_hz = 0.0
        
        # --- LANE TRACKING ENGINES ---
        self.is_auto_mode = False
        self.is_calibrating = False
        self.auto_start_time = 0.0 
        
        self.tracker = HybridLaneTracker(img_shape=(480, 640))
        self.jct     = JunctionDetector()
        self.rbt     = RoundaboutNavigator()
        self.guard   = DividerGuard()
        self.M       = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self.clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0  
        self.last_target   = 320.0 + RIGHT_LANE_OFFSET_PX
        self.lost_frames   = 0
        
        # Planner Logic
        self.mode = "DRIVE"
        self.path_tool = "START"
        self.start_node = None; self.end_node = None; self.pass_node = None
        self.path = []
        self.visited_path_nodes = set() 
        self.signs = []
        
        # Tracking Logic
        self.path_signs = []   
        self.visible_signs = {} 

        # ADAS State Variables 
        self.speed_limit = 200 
        self.in_highway_mode = False      
        self.ai_override_speed = None 
        self.stop_until = 0.0         
        self.acted_signs = set()      
        self.last_logged_adas = ""    
        self.current_light_status = "NONE" 
        
        # Active State Flags for UI Glow Dots
        self.active_state_flags = {
            "stop_sign": False,
            "no_entry": False,
            "pedestrian": False,
            "red_light": False,
            "yellow_light": False,
            "green_light": False,
            "caution": False,
            "highway": False,
            "park": False,
            "overtake": False
        }
        
        # Overtake Logic State Machine
        self.overtake_state = "IDLE"
        self.overtake_timer = 0.0
        self.last_car_dist = 999.0
        self.last_car_time = time.time()
        self.current_overtake_offset = 0.0
        
        # Autonomous Playback State (Parking)
        self.is_playing_back = False
        self.is_parking_reverse_mode = False
        self.is_waiting_to_reverse = False
        self.parking_wait_until = 0.0
        self.playback_queue = []
        self.playback_cmd = None
        self.playback_frames = 0
        
        self.has_driven = False 
        
        # DATA LOADING
        self.load_signs()
        if not os.path.exists(GRAPH_FILE): 
            messagebox.showerror("Error", "Missing GraphML file")
            return
        self.G = nx.read_graphml(GRAPH_FILE)
        self.pil_bg, self.svg_w, self.svg_h = self._load_svg(SVG_FILE, 600)
        
        # SCALING
        self.ppm_x = self.svg_w / REAL_WIDTH_M
        self.ppm_y = self.svg_h / REAL_HEIGHT_M
        self.node_pixels = {n: self.to_pixel(float(d['x']), float(d['y'])) for n, d in self.G.nodes(data=True)}
        
        # CAMERA
        self.picam2 = None
        self.latest_frame = None
        self.latest_raw_bgr = None 
        
        if _CAM_AVAILABLE: self._init_camera()
        
        # BUILD UI
        self._setup_layout()
        self.load_config()
        
        # LOOPS
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.update_camera() 
        self.control_loop()
        self.render_map() 

    def _setup_layout(self):
        # 1. Top Status Bar
        status = tk.Frame(self.root, bg=THEME["panel"], height=35)
        status.pack(side=tk.TOP, fill=tk.X)
        self.lbl_conn = tk.Label(status, text="⚫ DISCONNECTED", bg=THEME["panel"], fg=THEME["danger"], font=THEME["font_h"])
        self.lbl_conn.pack(side=tk.LEFT, padx=15)
        self.lbl_telemetry = tk.Label(status, text="SPD: 0 | STR: 0° | LMT: 200 | [DRIVE]", bg=THEME["panel"], fg=THEME["accent"], font=("Consolas", 11, "bold"))
        self.lbl_telemetry.pack(side=tk.LEFT, padx=20)
        self.lbl_ai = tk.Label(status, text="AI: OFF", bg=THEME["panel"], fg="grey", font=("Consolas", 10))
        self.lbl_ai.pack(side=tk.RIGHT, padx=15)

        # 2. Main Splitter (2 Panes: Left and Right. The middle is now a solid Frame so it doesn't crush)
        main_panes = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=THEME["sash"], sashwidth=6, sashrelief=tk.RAISED)
        main_panes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === LEFT PANEL (Cameras) ===
        left_panes = tk.PanedWindow(main_panes, orient=tk.VERTICAL, bg=THEME["sash"], sashwidth=6, sashrelief=tk.RAISED)
        main_panes.add(left_panes, minsize=350, width=440, stretch="always")
        
        cam_frm = tk.LabelFrame(left_panes, text="Raw Camera (YOLO)", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        self.cam_label = tk.Label(cam_frm, bg="black")
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        placeholder = Image.new('RGB', (440, 330), color='black')
        self._ph_img = ImageTk.PhotoImage(placeholder)
        self.cam_label.configure(image=self._ph_img)
        left_panes.add(cam_frm, minsize=200, height=350, stretch="always")

        bev_frm = tk.LabelFrame(left_panes, text="BEV Lane Tracking", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        self.bev_label = tk.Label(bev_frm, bg="black")
        self.bev_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.bev_label.configure(image=self._ph_img)
        left_panes.add(bev_frm, minsize=200, height=350, stretch="always")
        
        # === MID PANEL (Solid Container Frame to prevent overlapping and squishing) ===
        mid_col_frame = tk.Frame(main_panes, bg=THEME["bg"])
        main_panes.add(mid_col_frame, minsize=380, stretch="never")

        def make_slider(parent, text, row, from_, to_, res, default):
            tk.Label(parent, text=text, bg=THEME["panel"], fg="#ccc", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            s = tk.Scale(parent, from_=from_, to=to_, resolution=res, orient=tk.HORIZONTAL, bg=THEME["panel"], fg="white", highlightthickness=0, bd=0, length=150, sliderlength=15, width=12)
            s.set(default)
            s.grid(row=row, column=1, sticky="e", padx=5, pady=2)
            return s

        # 1. System Controls (Grid Layout)
        sys_frm = tk.LabelFrame(mid_col_frame, text="System Controls", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        sys_frm.pack(fill=tk.X, padx=5, pady=3)
        sys_frm.columnconfigure(0, weight=1)
        sys_frm.columnconfigure(1, weight=1)
        
        tk.Button(sys_frm, text="💾 Save Config", bg="#2980b9", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), command=self.save_config).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        tk.Button(sys_frm, text="📂 Load Config", bg="#27ae60", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), command=self.load_config).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self.btn_connect = tk.Button(sys_frm, text="CONNECT CAR", bg=THEME["accent"], fg="white", relief="flat", font=("Segoe UI", 10, "bold"), command=self.toggle_connection)
        self.btn_connect.grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.btn_auto = tk.Button(sys_frm, text="MODE: MANUAL", bg="#444", fg="white", relief="flat", font=THEME["font_h"], command=self.toggle_auto_mode)
        self.btn_auto.grid(row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=3)

        # 2. Drive Dynamics (Grid Layout)
        dyn_frm = tk.LabelFrame(mid_col_frame, text="Drive Dynamics", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        dyn_frm.pack(fill=tk.X, padx=5, pady=3)
        dyn_frm.columnconfigure(1, weight=1)
        self.slider_base_speed = make_slider(dyn_frm, "Base Speed (PWM):", 0, 0, 500, 1, 200)
        self.slider_sim_speed = make_slider(dyn_frm, "Map Sim Mult:", 1, 0.1, 3.0, 0.1, 1.0)
        self.slider_steer_mult = make_slider(dyn_frm, "Steer Multiplier:", 2, 0.1, 3.0, 0.1, 1.0)
        self.slider_overtake_dist = make_slider(dyn_frm, "Overtake Dist (m):", 3, 0.5, 5.0, 0.1, 1.2)
        self.slider_overtake_time = make_slider(dyn_frm, "Overtake Time (s):", 4, 1.0, 5.0, 0.1, 2.0)

        # 3. Vision Tracking (Grid Layout)
        vis_frm = tk.LabelFrame(mid_col_frame, text="Vision Tracking", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        vis_frm.pack(fill=tk.X, padx=5, pady=3)
        vis_frm.columnconfigure(1, weight=1)
        self.slider_lookahead = make_slider(vis_frm, "Look Ahead (px):", 0, 60, 300, 1, 150)
        self.slider_lanewidth = make_slider(vis_frm, "Lane Width (px):", 1, 100, 400, 1, 280)
        self.slider_offset = make_slider(vis_frm, "Offset (L- / R+):", 2, 0, 100, 1, 50)

        # 4. Map Tools
        mode_frm = tk.LabelFrame(mid_col_frame, text="Map Tools", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        mode_frm.pack(fill=tk.X, padx=5, pady=3)
        btn_row2 = tk.Frame(mode_frm, bg=THEME["panel"])
        btn_row2.pack(fill=tk.X, padx=2, pady=2)
        tk.Button(btn_row2, text="📍 PLANNER", bg="#444", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), command=lambda: self.set_mode("NAV")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(btn_row2, text="🛑 SIGNS", bg="#444", fg="white", relief="flat", font=("Segoe UI", 9, "bold"), command=lambda: self.set_mode("SIGN")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.lbl_mode = tk.Label(mode_frm, text="MODE: DRIVE", bg=THEME["panel"], fg=THEME["warning"], font=("Segoe UI", 10, "bold"))
        self.lbl_mode.pack(pady=2)

        # 5. ADAS Active Responses
        adas_frm = tk.LabelFrame(mid_col_frame, text="Active Responses", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        adas_frm.pack(fill=tk.X, padx=5, pady=3)
        self.indicators = {}
        states = [
            ("🛑 STOP", "stop_sign"), 
            ("⛔ NO ENTRY", "no_entry"), 
            ("🚸 PEDESTRIAN", "pedestrian"), 
            ("🔴 RED LGT", "red_light"),
            ("🟡 YEL LGT", "yellow_light"),
            ("🟢 GRN LGT", "green_light"),
            ("⚠️ CAUTION", "caution"), 
            ("🛣️ HIGHWAY", "highway"), 
            ("🅿️ AUTO-PARK", "park"),
            ("🏎️ OVERTAKE", "overtake")
        ]
        grid_frm = tk.Frame(adas_frm, bg=THEME["panel"])
        grid_frm.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)
        grid_frm.columnconfigure(0, weight=1)
        grid_frm.columnconfigure(1, weight=1)
        
        for i, (text, key) in enumerate(states):
            row = i // 2
            col = i % 2
            f = tk.Frame(grid_frm, bg=THEME["panel"])
            f.grid(row=row, column=col, sticky="w", padx=4, pady=3)
            c = tk.Canvas(f, width=14, height=14, bg=THEME["panel"], highlightthickness=0)
            c.pack(side=tk.LEFT)
            dot = c.create_oval(2, 2, 12, 12, fill="#333333", outline="#555555")
            tk.Label(f, text=text, bg=THEME["panel"], fg="white", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
            self.indicators[key] = (c, dot)

        # 6. System Log (Takes remaining bottom space of the middle column)
        log_frm = tk.LabelFrame(mid_col_frame, text="System Log", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        log_frm.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        log_head = tk.Frame(log_frm, bg=THEME["panel"])
        log_head.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(log_head, text="Events:", bg=THEME["panel"], fg="#aaa", font=("Consolas", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_hz = tk.Label(log_head, text="OFFLINE", bg=THEME["panel"], fg="gray", font=("Consolas", 10, "bold"))
        self.lbl_hz.pack(side=tk.RIGHT)
        
        self.log_text = tk.Text(log_frm, height=4, bg="black", fg="#00ff00", font=("Consolas", 9), state="disabled", relief="flat")
        scroll_log = ttk.Scrollbar(log_frm, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll_log.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.tag_config("INFO", foreground="#cccccc")
        self.log_text.tag_config("WARN", foreground="orange")
        self.log_text.tag_config("CRITICAL", foreground="#ff3333")
        self.log_text.tag_config("AI", foreground="#00ffff")
        self.log_text.tag_config("SUCCESS", foreground="#00ff00")

        # === RIGHT PANEL (Map, Table, Settings) ===
        right_frame = tk.Frame(main_panes, bg=THEME["bg"])
        main_panes.add(right_frame, minsize=400, stretch="always")
        
        # Map Tools toolbar at top of right frame
        self.tool_frame = tk.Frame(right_frame, bg=THEME["panel"], height=40)
        self.tool_frame.pack(fill=tk.X, pady=(0, 5))
        self.build_nav_tools() 

        right_panes = tk.PanedWindow(right_frame, orient=tk.VERTICAL, bg=THEME["sash"], sashwidth=6, sashrelief=tk.RAISED)
        right_panes.pack(fill=tk.BOTH, expand=True)

        self.map_canvas = tk.Canvas(right_panes, bg="black", highlightthickness=0)
        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<B1-Motion>", self.on_map_click) 
        self.map_canvas.bind("<Button-3>", self.on_right_click)
        right_panes.add(self.map_canvas, minsize=200, height=350, stretch="always")

        # Status Container for Table & Settings
        status_container = tk.Frame(right_panes, bg=THEME["bg"])
        right_panes.add(status_container, minsize=200, stretch="never")

        table_frame = tk.LabelFrame(status_container, text="Route Manifest & Live Status", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        columns = ("type", "loc", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        self.tree.heading("type", text="Sign Type")
        self.tree.heading("loc", text="Location")
        self.tree.heading("status", text="Real-time Status")
        self.tree.column("type", width=120); self.tree.column("loc", width=80); self.tree.column("status", width=120)
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#333", foreground="white", fieldbackground="#333", borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#444", foreground="white", relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", THEME["accent"])])
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        options_frm = tk.LabelFrame(status_container, text="Global Settings", bg=THEME["panel"], fg="white", font=THEME["font_h"])
        options_frm.pack(fill=tk.X)
        self.chk_invert = tk.BooleanVar(value=True) 
        tk.Checkbutton(options_frm, text="Invert Track Colors (White Road/Black Lines)", variable=self.chk_invert, bg=THEME["panel"], fg="#ccc", selectcolor="#333", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=5, pady=2)
        self.chk_teleport = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frm, text="Teleport Digital Twin on Sign detection (<0.85m)", variable=self.chk_teleport, bg=THEME["panel"], fg="#ccc", selectcolor="#333", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=5, pady=2)
        self.chk_auto_reverse = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frm, text="Auto-Reverse Parking Exit (10s Wait)", variable=self.chk_auto_reverse, bg=THEME["panel"], fg="#ccc", selectcolor="#333", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=5, pady=2)

    # --- SAVE / LOAD CONFIG ---
    def save_config(self):
        config = {
            "base_speed": self.slider_base_speed.get(),
            "sim_speed": self.slider_sim_speed.get(),
            "steer_mult": self.slider_steer_mult.get(),
            "lookahead": self.slider_lookahead.get(),
            "lanewidth": self.slider_lanewidth.get(),
            "offset": self.slider_offset.get(),
            "overtake_dist": self.slider_overtake_dist.get(),
            "overtake_time": self.slider_overtake_time.get(),
            "invert_colors": self.chk_invert.get(),
            "teleport": self.chk_teleport.get(),
            "auto_reverse": self.chk_auto_reverse.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
            self.log_event("Dashboard tunings saved successfully.", "SUCCESS")
        except Exception as e:
            self.log_event(f"Failed to save config: {e}", "CRITICAL")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                self.slider_base_speed.set(config.get("base_speed", 200))
                self.slider_sim_speed.set(config.get("sim_speed", 1.0))
                self.slider_steer_mult.set(config.get("steer_mult", 1.0))
                self.slider_lookahead.set(config.get("lookahead", 150))
                self.slider_lanewidth.set(config.get("lanewidth", 280))
                self.slider_offset.set(config.get("offset", 50))
                self.slider_overtake_dist.set(config.get("overtake_dist", 1.2))
                self.slider_overtake_time.set(config.get("overtake_time", 2.0))
                self.chk_invert.set(config.get("invert_colors", True))
                self.chk_teleport.set(config.get("teleport", True))
                self.chk_auto_reverse.set(config.get("auto_reverse", True))
                self.log_event("Loaded saved dashboard tunings.", "INFO")
            except Exception as e:
                self.log_event(f"Failed to load config: {e}", "WARN")

    def toggle_auto_mode(self):
        self.is_auto_mode = not self.is_auto_mode
        if self.is_auto_mode:
            self.auto_start_time = time.time()
            self.is_calibrating = True
            self.btn_auto.config(text="MODE: AUTONOMOUS", bg="#9b59b6") 
            self.log_event("Switched to AUTONOMOUS. Calibrating lane tracker for 5 seconds...", "SUCCESS")
        else:
            self.is_calibrating = False
            self.btn_auto.config(text="MODE: MANUAL", bg="#444")
            self.log_event("Switched to MANUAL driving mode.", "WARN")

    # --- Logging Helpers ---
    def log_event(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}\n"
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, full_msg, level)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def adas_log(self, msg, level="INFO"):
        if self.last_logged_adas != msg:
            self.log_event(msg, level)
            self.last_logged_adas = msg

    # --- CSV ACTION PLAYBACK LOGIC ---
    def start_csv_playback(self, filename="car_actions.csv"):
        if self.is_playing_back or getattr(self, 'is_waiting_to_reverse', False):
            return
        try:
            with open(filename, mode='r') as file:
                reader = csv.DictReader(file)
                self.playback_queue = []
                for row in reader:
                    speed_val = float(row["speed"])
                    self.playback_queue.append({
                        "speed": speed_val,
                        "steering": float(row["steering"]),
                        "pwm": int(float(row.get("pwm", abs(speed_val)))),
                        "direction": int(float(row.get("direction", 1 if speed_val >= 0 else -1))),
                        "duration": int(float(row.get("duration_frames", 10)))
                    })
            self.is_playing_back = True
            self.is_parking_reverse_mode = False
            self.playback_cmd = None
            self.playback_frames = 0
            self.log_event(f"Loaded {len(self.playback_queue)} commands from {filename}. Commencing Auto-Park.", "SUCCESS")
        except FileNotFoundError:
            self.log_event(f"Auto-Park failed! Could not find '{filename}'. Normal driving resumed.", "CRITICAL")
            self.is_playing_back = False

    def start_reverse_parking_exit(self):
        filename = "car_actions.csv"
        try:
            with open(filename, mode='r') as file:
                reader = list(csv.DictReader(file))
                self.playback_queue = []
                for row in reversed(reader):
                    speed_val = float(row["speed"])
                    st = float(row["steering"])
                    rev_s = -speed_val
                    self.playback_queue.append({
                        "speed": rev_s,
                        "steering": st, 
                        "pwm": int(float(row.get("pwm", abs(speed_val)))),
                        "direction": int(1 if rev_s >= 0 else -1),
                        "duration": int(float(row.get("duration_frames", 10)))
                    })
            self.is_playing_back = True
            self.is_parking_reverse_mode = True
            self.playback_cmd = None
            self.playback_frames = 0
            self.log_event("Executing Reverse Auto-Park Exit sequence.", "WARN")
        except Exception as e:
            self.log_event("Reverse exit failed.", "CRITICAL")

    # --- BEV PIPELINE (V2 Updates) ---
    def _get_bev(self, frame, invert=False):
        if invert:
            frame = cv2.bitwise_not(frame)
            
        warped_colour = cv2.warpPerspective(frame, self.M, (640, 480))
        
        blurred = cv2.GaussianBlur(warped_colour, (5, 5), 0)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        L   = self.clahe.apply(hls[:, :, 1])
        
        binary = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -8)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    def detect_traffic_light_color(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if y2 - y1 < 5 or x2 - x1 < 5:
            return "NONE"

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 150]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 120, 150]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_yellow = cv2.inRange(hsv, np.array([15, 120, 150]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 120, 150]), np.array([90, 255, 255]))

        r_count = cv2.countNonZero(mask_red)
        y_count = cv2.countNonZero(mask_yellow)
        g_count = cv2.countNonZero(mask_green)

        max_count = max(r_count, y_count, g_count)
        
        box_area = (y2 - y1) * (x2 - x1)
        min_pixels = max(10, int(box_area * 0.05))

        if max_count < min_pixels:
            return "NONE"
        elif max_count == r_count:
            return "RED"
        elif max_count == y_count:
            return "YELLOW"
        else:
            return "GREEN"

    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / LANE_WIDTH_M
        dx    = target_x - 320.0
        dy    = max(float(look_ahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)

    def _draw_poly(self, img, fit, colour):
        if fit is None: return
        ploty = np.linspace(0, 479, 240).astype(np.float32)
        xs    = np.polyval(fit, ploty).astype(np.float32)
        pts = np.stack([xs, ploty], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(img, [pts], isClosed=False, color=colour, thickness=3)

    # --- LOGIC ---
    def update_camera(self):
        frame = None
        if self.picam2:
            try: frame = self.picam2.capture_array()
            except: pass
        else:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "NO CAM", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if frame is not None:
            if frame.ndim == 3 and frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else: 
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            self.latest_raw_bgr = bgr.copy()
            
            if _AI_AVAILABLE and self.traffic_engine:
                res = self.traffic_engine.process(bgr)
                self.visible_signs.clear()
                self.current_light_status = "NONE" 
                
                if hasattr(res, 'detections') and res.detections:
                    for det in res.detections:
                        lbl = getattr(det, 'label', '').lower()
                        if not lbl: continue
                        
                        if hasattr(det, 'bbox'):
                            x1, y1, x2, y2 = map(int, det.bbox)
                            if lbl == "traffic-light":
                                color = self.detect_traffic_light_color(bgr, x1, y1, x2, y2)
                                if color != "NONE":
                                    self.current_light_status = color
                                    
                        h = getattr(det, 'bbox_height', 0)
                        if h == 0 and hasattr(det, 'bbox'):
                            h = det.bbox[3] - det.bbox[1] 
                        if h == 0: h = 130  
                        
                        dist_m = (REAL_SIGN_HEIGHT_M * CAMERA_FOCAL_LENGTH_PX) / max(h, 1)

                        if lbl not in self.visible_signs or dist_m < self.visible_signs[lbl]['dist']:
                            self.visible_signs[lbl] = {'h': h, 'dist': dist_m}
                else:
                    for lbl in getattr(res, 'active_labels', []):
                        dist_m = 0.5 
                        self.visible_signs[lbl.lower()] = {'h': 130, 'dist': dist_m}

                final_img = res.yolo_debug_frame
                self.lbl_ai.config(text=f"AI: {res.state}", fg="cyan")
            else:
                final_img = frame
                self.lbl_ai.config(text="AI: N/A")

            if final_img is not None:
                final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                cw = self.cam_label.winfo_width()
                ch = self.cam_label.winfo_height()
                if cw > 20 and ch > 20:
                    img = Image.fromarray(final_img).resize((cw, ch))
                else:
                    img = Image.fromarray(final_img).resize((440, 330))
                    
                imgtk = ImageTk.PhotoImage(image=img)
                self.cam_label.imgtk = imgtk
                self.cam_label.configure(image=imgtk, text="")

        self.root.after(50, self.update_camera)

    def apply_adas_rules(self):
        self.ai_override_speed = None
        current_time = time.time()
        
        # 1. Obey active hard stops
        if current_time < self.stop_until:
            self.ai_override_speed = 0
            if self.stop_until > current_time + 1000:
                self.active_state_flags["no_entry"] = True
            else:
                self.active_state_flags["stop_sign"] = True
            return
            
        if not self.visible_signs:
            self.last_logged_adas = "" 
            
        active_labels = set(self.visible_signs.keys())
        for acted in list(self.acted_signs):
            if acted not in active_labels:
                self.acted_signs.remove(acted)
            
        # 2. --- CRITICAL DYNAMIC STOPS AND OVERTAKES ---
        if "car" not in self.visible_signs:
            self.last_car_dist = 999.0
            
        for lbl, info in self.visible_signs.items():
            dist_m = info['dist']
            
            if lbl == "car":
                approach_delta = 0.0
                if self.last_car_dist < 10.0:
                    approach_delta = self.last_car_dist - dist_m
                self.last_car_dist = dist_m
                self.last_car_time = current_time
                
                overtake_dist_thresh = self.slider_overtake_dist.get()
                
                if dist_m < 2.5:
                    if self.in_highway_mode and dist_m < overtake_dist_thresh and approach_delta > 0.01 and self.overtake_state == "IDLE":
                        self.overtake_state = "LANE_CHANGE_LEFT"
                        self.overtake_timer = current_time
                        self.log_event(f"ADAS: Slower car within {overtake_dist_thresh}m. Initiating HIGHWAY OVERTAKE.", "WARN")
                    elif self.overtake_state == "IDLE":
                        self.ai_override_speed = 100
                        self.active_state_flags["caution"] = True
                        self.adas_log("ADAS: Car ahead. Tailing at safe speed.", "INFO")
            
            if lbl == "pedestrian" and dist_m < 2.5:
                self.ai_override_speed = 0
                self.active_state_flags["pedestrian"] = True
                self.adas_log("ADAS: Pedestrian in path! Overwriting controls to STOP.", "CRITICAL")
                return
            if lbl == "traffic-light" and dist_m < 3.0:
                if self.current_light_status == "RED":
                    self.ai_override_speed = 0
                    self.active_state_flags["red_light"] = True
                    self.adas_log("ADAS: Traffic Light Red! Overwriting controls to STOP.", "CRITICAL")
                    return
                elif self.current_light_status == "YELLOW":
                    self.ai_override_speed = 100
                    self.active_state_flags["yellow_light"] = True
                    self.adas_log("ADAS: Traffic Light Yellow. Overwriting speed to CAUTION.", "WARN")
                elif self.current_light_status == "GREEN":
                    self.active_state_flags["green_light"] = True
                    self.adas_log("ADAS: Traffic Light Green! Proceeding.", "SUCCESS")

        # 3. --- MAPPED STATIC SIGNS (Robust Path-Order Execution) ---
        expected_signs = [s for s in self.path_signs if s.get('status') not in ["✅ CONFIRMED", "⚠️ MISSED"]][:2]
                
        for exp_sign in expected_signs:
            exp_type = exp_sign['type'].lower()
            
            if exp_type in self.visible_signs:
                dist_m = self.visible_signs[exp_type]['dist']
                
                if dist_m < 0.85:
                    exp_sign['status'] = "✅ CONFIRMED" 
                    
                    if exp_sign != expected_signs[0]:
                        expected_signs[0]['status'] = "⚠️ MISSED"
                        self.log_event(f"ADAS: Missed expected '{expected_signs[0]['type']}'. Auto-correcting sequence.", "WARN")
                    
                    if self.chk_teleport.get():
                        n_id = exp_sign['node']
                        if n_id in self.G.nodes:
                            self.car_x = float(self.G.nodes[n_id]['x'])
                            self.car_y = float(self.G.nodes[n_id]['y'])
                            self.log_event(f"ADAS: Snapped digital twin to {exp_type} node.", "INFO")
                    
                    if exp_type == "stop-sign":
                        self.stop_until = current_time + 3.0
                        self.ai_override_speed = 0
                        self.active_state_flags["stop_sign"] = True
                        self.log_event(f"ADAS: Reached Stop Sign (<0.85m). Halting for 3s.", "WARN")
                    elif exp_type == "parking-sign":
                        self.log_event(f"ADAS: Reached Parking Zone (<0.85m). Initiating Auto-Park!", "SUCCESS")
                        self.start_csv_playback("car_actions.csv")
                    elif exp_type == "highway-entry-sign":
                        self.in_highway_mode = True
                        self.log_event(f"ADAS: Highway Entry passed. Base Limit -> {HIGHWAY_SPEED_PWM}.", "INFO")
                    elif exp_type == "highway-exit-sign":
                        self.in_highway_mode = False
                        self.log_event(f"ADAS: Highway Exit passed. Base Limit restored.", "INFO")
                    elif exp_type == "noentry-sign":
                        self.stop_until = current_time + 9999.0 
                        self.ai_override_speed = 0
                        self.active_state_flags["no_entry"] = True
                        self.log_event(f"ADAS: Blocked by No Entry. STOP.", "CRITICAL")
                    elif exp_type == "priority-sign":
                        self.log_event(f"ADAS: Priority road entered.", "INFO")
                    elif exp_type == "roundabout-sign":
                        self.log_event(f"ADAS: Entering Roundabout.", "INFO")
                    elif exp_type == "oneway-sign":
                        self.log_event(f"ADAS: One-way confirmed.", "INFO")
                        
                    self.update_sign_table()

                elif dist_m < 1.5:
                    if exp_type == "crosswalk-sign":
                        self.ai_override_speed = 100
                        self.active_state_flags["caution"] = True
                        self.adas_log(f"ADAS: Approaching expected Crosswalk. Slowing down.", "WARN")
                    elif exp_type == "parking-sign":
                        self.ai_override_speed = 100
                        self.active_state_flags["caution"] = True
                        self.adas_log(f"ADAS: Approaching expected Parking zone. Slowing down.", "INFO")

    def check_sign_status(self):
        if not self.path_signs: return
        update_tree = False
        
        expected_signs = [s for s in self.path_signs if s.get('status') not in ["✅ CONFIRMED", "⚠️ MISSED"]][:2]
                
        for exp_sign in expected_signs:
            s_type_lower = exp_sign['type'].lower()
            if s_type_lower in self.visible_signs:
                dist_m = self.visible_signs[s_type_lower]['dist']
                current_status = exp_sign.get('status', '⏳ PENDING')
                
                if dist_m < 3.5 and current_status == "⏳ PENDING":
                    exp_sign['status'] = "🔴 LIVE DETECT"
                    self.log_event(f"AI CAMERA: Spotted '{exp_sign['type']}' ahead (~{dist_m:.1f}m).", "AI")
                    update_tree = True
                
        if update_tree:
            self.update_sign_table()

    def update_sign_table(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for s in self.path_signs:
            s_id = str(s['node'])
            s_type = s['type']
            emoji = SIGN_MAP.get(s_type, {"emoji": "?"})['emoji']
            name = SIGN_MAP.get(s_type, {"name": s_type})['name']
            status_text = s.get('status', "⏳ PENDING")
            row_id = self.tree.insert("", "end", values=(f"{emoji} {name}", f"Node {s_id}", status_text))
            if status_text == "🔴 LIVE DETECT":
                self.tree.item(row_id, tags=("live",))
        self.tree.tag_configure("live", background="#440000", foreground="#ff5555")

    def update_indicator_dots(self):
        for key, condition in self.active_state_flags.items():
            if key in self.indicators:
                c, dot = self.indicators[key]
                if condition:
                    c.itemconfig(dot, fill="#00ff00", outline="#ffffff")
                else:
                    c.itemconfig(dot, fill="#333333", outline="#555555")
                
    def update_path_progress(self):
        if not self.path: return
        min_dist = 999.0
        closest_idx = 0
        for i in range(len(self.path)-1):
            u = self.path[i]
            v = self.path[i+1]
            ux, uy = float(self.G.nodes[u]['x']), float(self.G.nodes[u]['y'])
            vx, vy = float(self.G.nodes[v]['x']), float(self.G.nodes[v]['y'])
            dx, dy = vx - ux, vy - uy
            l2 = dx*dx + dy*dy
            if l2 == 0: continue
            
            t = max(0, min(1, ((self.car_x - ux)*dx + (self.car_y - uy)*dy) / l2))
            proj_x, proj_y = ux + t*dx, uy + t*dy
            dist = math.hypot(self.car_x - proj_x, self.car_y - proj_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        for i in range(closest_idx):
            self.visited_path_nodes.add(self.path[i])
            self.visited_path_nodes.add(self.path[i+1])

    def control_loop(self):
        now = time.time()
        dt = max(now - self.last_ctrl_time, 0.001)
        self.last_ctrl_time = now
        
        for key in self.active_state_flags:
            self.active_state_flags[key] = False
        
        if self.in_highway_mode:
            self.speed_limit = HIGHWAY_SPEED_PWM
            self.active_state_flags["highway"] = True
        else:
            self.speed_limit = self.slider_base_speed.get()
            
        if self.overtake_state != "IDLE":
            self.active_state_flags["overtake"] = True
            
        map_sim_multiplier = self.slider_sim_speed.get()
        steer_multiplier = self.slider_steer_mult.get()
        overtake_time_thresh = self.slider_overtake_time.get()
        
        if self.is_connected:
            hz = 1.0 / dt if dt > 0 else 0.0
            self.current_hz = 0.8 * self.current_hz + 0.2 * hz
            self.lbl_hz.config(text=f"{self.current_hz:.1f} Hz", fg="cyan")
            
            if self.is_playing_back or getattr(self, 'is_waiting_to_reverse', False):
                self.active_state_flags["park"] = True
                mode_str = "AUTO-PARK"
                
                if self.is_playing_back:
                    if self.playback_cmd is None:
                        if len(self.playback_queue) > 0:
                            self.playback_cmd = self.playback_queue.pop(0)
                            self.playback_frames = 0
                        else:
                            if self.is_parking_reverse_mode:
                                self.is_playing_back = False
                                self.is_parking_reverse_mode = False
                                self.log_event("Reverse Park Exit Completed. Resuming drive.", "SUCCESS")
                                self.current_speed = 0; self.current_steer = 0
                            else:
                                self.is_playing_back = False
                                self.log_event("Auto-Park Sequence Completed.", "SUCCESS")
                                self.current_speed = 0; self.current_steer = 0
                                self.handler.set_speed(0); self.handler.set_steering(0)
                                
                                if self.chk_auto_reverse.get():
                                    self.log_event("Waiting 10s before auto-reversing...", "INFO")
                                    self.parking_wait_until = time.time() + 10.0
                                    self.is_waiting_to_reverse = True
                            
                    if self.is_playing_back and self.playback_cmd is not None:
                        p_speed = self.playback_cmd["speed"]
                        p_steer = self.playback_cmd["steering"]
                        p_pwm = self.playback_cmd.get("pwm", abs(p_speed))
                        p_dir = self.playback_cmd.get("direction", 1 if p_speed > 0 else -1)
                        
                        if hasattr(self.handler, 'set_pwm') and hasattr(self.handler, 'set_direction'):
                            self.handler.set_pwm(int(p_pwm))
                            self.handler.set_direction(int(p_dir))
                        else:
                            if abs(p_speed) > 1: self.handler.set_speed(int(p_speed))
                            else: self.handler.set_speed(0)
                                
                        self.handler.set_steering(p_steer)
                        self.current_speed = p_speed
                        self.current_steer = p_steer
                        
                        self.playback_frames += 1
                        if self.playback_frames >= self.playback_cmd["duration"]:
                            self.playback_cmd = None
                            
                elif getattr(self, 'is_waiting_to_reverse', False):
                    self.current_speed = 0
                    self.current_steer = 0
                    self.handler.set_speed(0)
                    self.handler.set_steering(0)
                    
                    remaining = self.parking_wait_until - time.time()
                    if remaining <= 0:
                        self.is_waiting_to_reverse = False
                        self.start_reverse_parking_exit()
                    else:
                        mode_str = f"PARK WAIT: {remaining:.1f}s"

            else:
                mode_str = "AUTONOMOUS" if self.is_auto_mode else "MANUAL"
                self.apply_adas_rules()
                
                look_ahead = self.slider_lookahead.get()
                lane_width_px = self.slider_lanewidth.get()
                fine_px = (self.slider_offset.get() - 50) * 2
                
                # --- OVERTAKE STATE MACHINE ---
                if self.overtake_state == "LANE_CHANGE_LEFT":
                    if now - self.overtake_timer > overtake_time_thresh:
                        self.overtake_state = "PASSING"
                        self.overtake_timer = now
                elif self.overtake_state == "PASSING":
                    if now - self.overtake_timer > overtake_time_thresh * 1.5: 
                        self.overtake_state = "LANE_CHANGE_RIGHT"
                        self.overtake_timer = now
                        self.log_event("ADAS: Overtake complete. Returning to lane.", "SUCCESS")
                elif self.overtake_state == "LANE_CHANGE_RIGHT":
                    if now - self.overtake_timer > overtake_time_thresh:
                        self.overtake_state = "IDLE"

                target_overtake_offset = 0.0
                if self.overtake_state in ["LANE_CHANGE_LEFT", "PASSING"]:
                    target_overtake_offset = -lane_width_px * 1.1 
                    
                self.current_overtake_offset += (target_overtake_offset - self.current_overtake_offset) * 0.1
                total_offset = RIGHT_LANE_OFFSET_PX + fine_px + self.current_overtake_offset
                
                # --- BEV TRACKING ---
                dbg = None
                nav_state = "NORMAL"
                anchor = "NONE"
                curvature = 0.0
                raw_steer = 0.0
                target_x = 320.0
                guard_on = False
                detect_mode = "SEARCH"
                
                if self.latest_raw_bgr is not None:
                    lab = cv2.cvtColor(self.latest_raw_bgr, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l_channel)
                    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
                    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    
                    if self.chk_invert.get():
                        eval_bgr = cv2.bitwise_not(enhanced_bgr)
                    else:
                        eval_bgr = enhanced_bgr
                    
                    warped = self._get_bev(eval_bgr, invert=False) 
                    sl, sr, dbg, detect_mode = self.tracker.update(warped)
                    
                    expected_sign = None
                    for s in self.path_signs:
                        if s.get('status') != "✅ CONFIRMED":
                            expected_sign = s
                            break
                            
                    jct_state = self.jct.update(warped, self.tracker.left_conf, self.tracker.right_conf, self.tracker.left_fit, self.tracker.right_fit, lane_width_px)
                    
                    if expected_sign and expected_sign['type'].lower() == "roundabout-sign":
                        rbt_state = self.rbt.update(self.tracker.left_fit, self.tracker.right_fit, lane_width_px)
                        nav_state = rbt_state if rbt_state == "ROUNDABOUT" else jct_state
                    else:
                        nav_state = jct_state
                    
                    eff_la = look_ahead
                    if nav_state == "ROUNDABOUT": eff_la = int(look_ahead * 0.55)
                    elif nav_state == "JUNCTION": eff_la = int(look_ahead * 0.75)
                    eff_la = max(60, eff_la)
                    y_eval = max(0, 480 - eff_la)
                    
                    t_x, anchor = self.tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state)
                    
                    if t_x is None:
                        self.lost_frames += 1
                        target_x = self.last_target
                    else:
                        self.lost_frames = 0
                        self.last_target = target_x = t_x
                        
                    raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)
                    curvature = self.tracker.get_curvature(y_eval)

                # --- CONTROL SELECTION ---
                spd = 0.0
                steer_target = 0.0
                
                if self.is_auto_mode:
                    elapsed_auto = time.time() - self.auto_start_time
                    self.is_calibrating = elapsed_auto < 5.0
                    
                    if self.is_calibrating:
                        spd = 0
                        steer_target = self.current_steer 
                        self.tracker.reset()
                        self.smooth_steer = 0.0
                        self.smooth_guard = 0.0
                        self.prev_steer = 0.0
                    else:
                        steer_delta = abs(raw_steer - self.smooth_steer)
                        alpha = 0.50 if steer_delta > 8.0 else 0.25
                        self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
                        steer_target = self.smooth_steer
                        
                        rate_delta = max(-5.0, min(5.0, steer_target - self.prev_steer))
                        steer_target = self.prev_steer + rate_delta
                        self.prev_steer = steer_target
                        
                        if self.overtake_state == "IDLE":
                            g_l = self.tracker.sl if self.tracker.left_stale == 0 else None
                            g_r = self.tracker.sr if self.tracker.right_stale == 0 else None
                            steer_guarded, guard_spd, guard_on = self.guard.apply(steer_target, g_l, g_r, y_eval=440)
                            
                            if self.lost_frames > 0: 
                                self.smooth_guard = 0.0
                                guard_on = False
                            else: 
                                guard_delta = steer_guarded - steer_target
                                self.smooth_guard = 0.55 * guard_delta + 0.45 * self.smooth_guard
                            steer_target = steer_target + self.smooth_guard
                            
                            if guard_on: spd *= guard_spd
                            
                        steer_target = max(-30.0, min(30.0, steer_target))
                        steer_target *= steer_multiplier 
                        
                        if self.lost_frames > LOST_GRACE_FRAMES: spd = 0.0
                        elif self.speed_limit == 0: spd = 0.0
                        elif nav_state == "ROUNDABOUT": spd = self.speed_limit * 0.50
                        elif nav_state == "JUNCTION": spd = self.speed_limit * 0.55
                        elif curvature > 0.003: spd = self.speed_limit * 0.60
                        elif curvature > 0.0015: spd = self.speed_limit * 0.80
                        elif anchor == "DUAL" and abs(steer_target) < 10: spd = self.speed_limit * 1.15
                        elif abs(steer_target) > 18: spd = self.speed_limit * 0.60
                        elif abs(steer_target) > 10: spd = self.speed_limit * 0.80
                        else: spd = float(self.speed_limit)
                        
                        if 0 < self.lost_frames <= LOST_GRACE_FRAMES:
                            spd *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)
                else:
                    spd = self.speed_limit if self.keys['Up'] else (-self.speed_limit if self.keys['Down'] else 0)
                    target_s = (-25.0 if self.keys['Left'] else (25.0 if self.keys['Right'] else 0.0)) * steer_multiplier
                    steer_target = target_s
                
                if self.ai_override_speed is not None:
                    if self.ai_override_speed == 0:
                        spd = 0
                    else:
                        if spd > self.ai_override_speed: spd = self.ai_override_speed
                        if spd < -self.ai_override_speed: spd = -self.ai_override_speed
                
                self.current_speed += (spd - self.current_speed) * 0.2
                
                if not self.is_auto_mode:
                    self.current_steer += (steer_target - self.current_steer) * 0.4
                elif getattr(self, 'is_calibrating', False):
                    pass 
                else: 
                    self.current_steer = steer_target 
                
                if abs(self.current_speed) < 1: self.current_speed = 0
                if abs(self.current_steer) < 0.5: self.current_steer = 0
                
                self.handler.set_speed(int(self.current_speed))
                self.handler.set_steering(self.current_steer)
                
                # --- V2 DEBUG DRAWING ---
                if dbg is not None:
                    self._draw_poly(dbg, sl, (255, 220, 0))  
                    self._draw_poly(dbg, sr, (0, 200, 255))
                    cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                    cv2.line(dbg, (int(target_x), y_eval), (320, 470), (0, 255, 0), 2)
                    cv2.line(dbg, (320, 450), (320, 480), (0, 0, 255), 3)

                    ref_x = 320 + RIGHT_LANE_OFFSET_PX
                    for y_tick in range(0, 480, 20):
                        cv2.line(dbg, (int(ref_x), y_tick), (int(ref_x), y_tick + 10), (100, 100, 100), 1)

                    if guard_on:
                        cv2.putText(dbg, "! GUARD !", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if self.lost_frames > 0:
                        grace_label = f"LOST ({self.lost_frames}/{LOST_GRACE_FRAMES})" if self.lost_frames <= LOST_GRACE_FRAMES else "LOST - STOPPED"
                        cv2.putText(dbg, grace_label, (130, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if getattr(self, 'is_calibrating', False):
                        cv2.putText(dbg, f"CALIBRATING: {5.0 - elapsed_auto:.1f}s", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                    else:
                        line1 = f"{detect_mode} | {anchor} | {nav_state} | {hz:.0f}fps"
                        line2 = f"Str:{steer_target:.1f} Spd:{spd:.0f} Crv:{curvature:.4f} Off:{int(total_offset)}"
                        cv2.putText(dbg, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                        cv2.putText(dbg, line2, (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                    cw_bev = self.bev_label.winfo_width()
                    ch_bev = self.bev_label.winfo_height()
                    dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                    img_bev = Image.fromarray(dbg_rgb).resize((cw_bev if cw_bev > 20 else 440, ch_bev if ch_bev > 20 else 330))
                    imgtk_bev = ImageTk.PhotoImage(image=img_bev)
                    self.bev_label.imgtk = imgtk_bev
                    self.bev_label.configure(image=imgtk_bev, text="")

            # Calculate Map Physics
            v_ms = (self.current_speed / 1000.0) * map_sim_multiplier
            
            if not getattr(self, 'is_waiting_to_reverse', False):
                self.update_path_progress()
                self.snap_to_graph(dt, v_ms)
                self.check_sign_status()
            
            self.lbl_telemetry.config(text=f"SPD: {int(self.current_speed)} | STR: {self.current_steer:.1f}° | LMT: {self.speed_limit} | [{mode_str}]")
            
        else:
            self.current_hz = 0.0
            self.lbl_hz.config(text="OFFLINE", fg="gray")
        
        self.update_indicator_dots()
        self.render_map()
        self.root.after(50, self.control_loop)

    def _init_camera(self):
        try:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "XRGB8888"})
            self.picam2.configure(cfg)
            self.picam2.start()
        except Exception as e:
            print(f"Cam Init Failed: {e}")
            self.picam2 = None

    def snap_to_graph(self, dt, v_ms):
        self.car_x += v_ms * math.cos(self.car_yaw) * dt
        self.car_y += v_ms * math.sin(self.car_yaw) * dt
        
        # Digital Map Twin steers correctly with the physical car
        if abs(self.current_steer) > 1.0:
            self.car_yaw -= (v_ms / 0.26) * math.sin(math.radians(self.current_steer)) * dt
            
        min_dist = 999.0; best = (self.car_x, self.car_y)
        best_dx, best_dy = 0, 0
        
        edges = [(self.path[i], self.path[i+1]) for i in range(len(self.path)-1)] if self.path else self.G.edges()
        
        for u, v in edges:
            ux = float(self.G.nodes[u]['x']); uy = float(self.G.nodes[u]['y'])
            vx = float(self.G.nodes[v]['x']); vy = float(self.G.nodes[v]['y'])
            px, py = self.car_x, self.car_y
            dx, dy = vx-ux, vy-uy; 
            if dx==0 and dy==0: continue
            t = max(0, min(1, ((px-ux)*dx + (py-uy)*dy) / (dx*dx+dy*dy)))
            nx, ny = ux+t*dx, uy+t*dy
            d = math.hypot(px-nx, py-ny)
            
            if self.path and d < 1.5:
                if math.hypot(px - ux, py - uy) < 1.5: self.visited_path_nodes.add(u)
                if math.hypot(px - vx, py - vy) < 1.5: self.visited_path_nodes.add(v)
                
            if d < min_dist: 
                min_dist = d
                best = (nx, ny)
                best_dx, best_dy = dx, dy
                
        if min_dist < 1.0: 
            self.car_x += (best[0]-self.car_x)*0.3
            self.car_y += (best[1]-self.car_y)*0.3
            
            if abs(self.current_steer) < 1.0 and best_dx != 0:
                target_yaw = math.atan2(best_dy, best_dx)
                diff = (target_yaw - self.car_yaw + math.pi) % (2 * math.pi) - math.pi
                self.car_yaw += diff * 0.8 

    def render_map(self):
        pil = self.pil_bg.copy(); draw = ImageDraw.Draw(pil)
        if self.path:
            for i in range(len(self.path)-1):
                n1, n2 = self.path[i], self.path[i+1]
                color = THEME["danger"] if (n1 in self.visited_path_nodes and n2 in self.visited_path_nodes) else THEME["accent"]
                
                p1 = self.node_pixels.get(n1)
                p2 = self.node_pixels.get(n2)
                if p1 and p2:
                    draw.line([p1, p2], fill=color, width=4)
        
        try: font = ImageFont.truetype("seguiemj.ttf", 20) 
        except: font = ImageFont.load_default()
        
        path_nodes = set(self.path)
        for s in self.signs:
            p = self.node_pixels[s['node']]
            s_type = s['type']
            emoji = SIGN_MAP.get(s_type, {"emoji": "?"})['emoji']
            outline = None
            
            if s['node'] in path_nodes:
                status = s.get('status', '⏳ PENDING')
                if status == "✅ CONFIRMED": outline = THEME["danger"]      
                elif status == "🔴 LIVE DETECT": outline = "#00ffff"        
                else: outline = THEME["success"]                             
            
            if outline: draw.ellipse([p[0]-14, p[1]-14, p[0]+14, p[1]+14], outline=outline, width=3)
            draw.text((p[0]-10, p[1]-10), emoji, font=font, fill="white", embedded_color=True)
            
        def mark(n, c): 
            if n: draw.ellipse([self.node_pixels[n][0]-6, self.node_pixels[n][1]-6, self.node_pixels[n][0]+6, self.node_pixels[n][1]+6], fill=c)
        mark(self.start_node, THEME["success"]); mark(self.pass_node, "cyan"); mark(self.end_node, THEME["danger"])
        
        if self.is_connected:
            cx, cy = self.to_pixel(self.car_x, self.car_y)
            hx = cx + math.cos(-self.car_yaw)*20; hy = cy + math.sin(-self.car_yaw)*20
            draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill="cyan", outline="white", width=2)
            draw.line([cx, cy, hx, hy], fill="white", width=2)
        
        self.tk_map = ImageTk.PhotoImage(pil)
        self.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map)

    def set_mode(self, m):
        self.mode = m
        self.lbl_mode.config(text=f"MODE: {m}")
        for w in self.tool_frame.winfo_children(): w.destroy()
        if m == "NAV": self.build_nav_tools()
        elif m == "SIGN": self.build_sign_tools()

    def build_nav_tools(self):
        self.var_path = tk.StringVar(value="START")
        tk.Radiobutton(self.tool_frame, text="🟢 Start", variable=self.var_path, value="START", bg=THEME["panel"], fg="white", selectcolor="#333", indicatoron=0).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.tool_frame, text="🔵 Pass", variable=self.var_path, value="PASS", bg=THEME["panel"], fg="white", selectcolor="#333", indicatoron=0).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.tool_frame, text="🔴 End", variable=self.var_path, value="END", bg=THEME["panel"], fg="white", selectcolor="#333", indicatoron=0).pack(side=tk.LEFT, padx=5)
        tk.Button(self.tool_frame, text="Clear Route", bg=THEME["danger"], fg="white", relief="flat", command=self.clear_route).pack(side=tk.RIGHT, padx=10)

    def build_sign_tools(self):
        self.var_sign = tk.StringVar(value="stop-sign")
        sign_choices = list(SIGN_MAP.keys())
        opt = tk.OptionMenu(self.tool_frame, self.var_sign, *sign_choices)
        opt.config(bg="#444", fg="white", highlightthickness=0); opt.pack(side=tk.LEFT, padx=5)
        self.chk_del = tk.BooleanVar(value=False)
        tk.Checkbutton(self.tool_frame, text="Delete Mode", variable=self.chk_del, bg=THEME["panel"], fg=THEME["danger"], selectcolor="#333").pack(side=tk.LEFT, padx=10)
        tk.Button(self.tool_frame, text="Save DB", bg=THEME["success"], fg="white", relief="flat", command=self.save_signs).pack(side=tk.RIGHT, padx=10)

    def on_map_click(self, event):
        mx, my = self.to_meter(event.x, event.y)
        click_point = np.array([event.x, event.y])
        
        if self.mode == "DRIVE":
            self.car_x = mx
            self.car_y = my
            self.current_speed = 0 
            self.render_map()
            return 

        closest = None
        min_d = 60.0
        for n, p in self.node_pixels.items():
            d = np.linalg.norm(np.array(p) - click_point)
            if d < min_d:
                min_d = d
                closest = n
            
        if self.mode == "NAV" and closest:
            t = self.var_path.get()
            if t == "START": self.start_node = closest
            elif t == "END": self.end_node = closest; self.calc_path()
            elif t == "PASS": self.pass_node = closest
            self.render_map()
            
        elif self.mode == "SIGN":
            if self.chk_del.get():
                original_count = len(self.signs)
                self.signs = [s for s in self.signs if np.linalg.norm(np.array(self.node_pixels[s['node']]) - click_point) > 60]
                if len(self.signs) < original_count:
                    self.log_event("Deleted sign(s) at click location.", "WARN")
            elif closest:
                exists = any(s['node'] == closest for s in self.signs)
                if not exists:
                    self.signs.append({
                        "node": closest, 
                        "type": self.var_sign.get(), 
                        "x": self.G.nodes[closest]['x'], 
                        "y": self.G.nodes[closest]['y'],
                        "status": "⏳ PENDING"
                    })
                    self.log_event(f"Placed '{self.var_sign.get()}' at node {closest}.", "SUCCESS")
                else:
                    self.log_event("A sign already exists at this node.", "WARN")
            self.calc_path()
            self.render_map()
            
    def calc_path(self):
        if not self.start_node or not self.end_node: return
        self.path_signs = []
        self.visited_path_nodes.clear()
        try:
            if self.pass_node:
                p1 = nx.shortest_path(self.G, self.start_node, self.pass_node)
                p2 = nx.shortest_path(self.G, self.pass_node, self.end_node)
                self.path = p1 + p2[1:]
            else: self.path = nx.shortest_path(self.G, self.start_node, self.end_node)
            
            if len(self.path) >= 2:
                n1, n2 = self.path[0], self.path[1]
                x1, y1 = float(self.G.nodes[n1]['x']), float(self.G.nodes[n1]['y'])
                x2, y2 = float(self.G.nodes[n2]['x']), float(self.G.nodes[n2]['y'])
                self.car_yaw = math.atan2(y2 - y1, x2 - x1)
                self.car_x = x1
                self.car_y = y1
                
            for node in self.path:
                for s in self.signs:
                    if s['node'] == node: 
                        s['status'] = "⏳ PENDING" 
                        self.path_signs.append(s)
            self.update_sign_table()
        except: self.path = []

    def align_car_to_path(self):
        if not self.path or len(self.path) < 2:
            return
        
        min_dist = 999.0
        best_dx, best_dy = 0, 0
        
        for i in range(len(self.path)-1):
            u = self.path[i]
            v = self.path[i+1]
            ux, uy = float(self.G.nodes[u]['x']), float(self.G.nodes[u]['y'])
            vx, vy = float(self.G.nodes[v]['x']), float(self.G.nodes[v]['y'])
            dx, dy = vx - ux, vy - uy
            l2 = dx*dx + dy*dy
            if l2 == 0: continue
            
            t = max(0, min(1, ((self.car_x - ux)*dx + (self.car_y - uy)*dy) / l2))
            proj_x, proj_y = ux + t*dx, uy + t*dy
            d = math.hypot(self.car_x - proj_x, self.car_y - proj_y)
            
            if d < min_dist:
                min_dist = d
                best_dx = dx
                best_dy = dy
                
        if best_dx != 0 or best_dy != 0:
            self.car_yaw = math.atan2(best_dy, best_dx)

    def toggle_connection(self):
        if not self.is_connected:
            if self.handler.connect():
                self.is_connected = True
                self.lbl_conn.config(text="🟢 CONNECTED", fg=THEME["success"])
                self.btn_connect.config(text="DISCONNECT", bg=THEME["danger"])
                self.log_event("Connected to STM32 Hardware.", "INFO")
                
                is_valid_resume = False
                if self.has_driven:
                    resume = messagebox.askyesno("Resume Run", "Do you want to resume from the car's last map position and state?")
                    if resume:
                        is_valid_resume = True
                
                if is_valid_resume:
                    self.log_event("Resuming from previous position.", "INFO")
                    self.align_car_to_path() 
                else:
                    self.in_highway_mode = False
                    self.ai_override_speed = None
                    self.stop_until = 0.0
                    self.acted_signs.clear()
                    self.is_playing_back = False
                    self.is_parking_reverse_mode = False
                    self.is_waiting_to_reverse = False
                    self.playback_queue = []
                    self.overtake_state = "IDLE"
                    self.current_overtake_offset = 0.0
                    self.last_car_dist = 999.0
                    self.visited_path_nodes.clear()
                    self.has_driven = True
                    
                    for s in self.path_signs:
                        if s.get('status') in ["✅ CONFIRMED", "⚠️ MISSED"]:
                             s['status'] = "⏳ PENDING"
                    self.update_sign_table()
                    
                    if self.start_node:
                        self.log_event(f"Car placed at defined Start Node ({self.start_node}).", "INFO")
                        self.car_x = float(self.G.nodes[self.start_node]['x'])
                        self.car_y = float(self.G.nodes[self.start_node]['y'])
                        if self.path and len(self.path) >= 2:
                            x1, y1 = float(self.G.nodes[self.path[0]]['x']), float(self.G.nodes[self.path[0]]['y'])
                            x2, y2 = float(self.G.nodes[self.path[1]]['x']), float(self.G.nodes[self.path[1]]['y'])
                            self.car_yaw = math.atan2(y2 - y1, x2 - x1)
                        else:
                            self.car_yaw = 0.0
                    else:
                        self.car_x = DEFAULT_START_X
                        self.car_y = DEFAULT_START_Y
                        self.car_yaw = 0.0
                        self.log_event("Car placed at Default Map Coordinates.", "INFO")
                
                self.current_speed = 0.0
                self.current_steer = 0.0
                self.render_map()
                
        else:
            self.handler.disconnect(); self.is_connected = False
            self.lbl_conn.config(text="⚫ DISCONNECTED", fg=THEME["danger"])
            self.btn_connect.config(text="CONNECT CAR", bg=THEME["accent"])
            self.log_event("Disconnected from Hardware.", "WARN")

    def to_pixel(self, x, y): return int((float(x)*self.ppm_x*FINAL_SCALE_X)+FINAL_OFF_X), int(self.svg_h-((float(y)*self.ppm_y*FINAL_SCALE_Y)+FINAL_OFF_Y))
    def to_meter(self, x, y): return (x-FINAL_OFF_X)/(self.ppm_x*FINAL_SCALE_X), (self.svg_h-y-FINAL_OFF_Y)/(self.ppm_y*FINAL_SCALE_Y)
    def _load_svg(self, path, w):
        from svglib.svglib import svg2rlg; from reportlab.graphics import renderPM
        d=svg2rlg(path); s=w/d.width; d.width*=s; d.height*=s; d.scale(s,s)
        return renderPM.drawToPIL(d, bg=0x111111), int(d.width), int(d.height)
    
    def save_signs(self):
        with open(SIGNS_DB_FILE, 'w') as f: json.dump(self.signs, f)
        self.log_event("Sign Database Saved.", "INFO")
        
    def load_signs(self):
        if os.path.exists(SIGNS_DB_FILE):
            with open(SIGNS_DB_FILE, 'r') as f: 
                self.signs = json.load(f)
            
            migration_map = {
                "Stop": "stop-sign", "Crosswalk": "crosswalk-sign", "Priority": "priority-sign",
                "Parking": "parking-sign", "Highway Entry": "highway-entry-sign",
                "Highway Exit": "highway-exit-sign", "Pedestrian": "pedestrian",
                "Traffic Light": "traffic-light", "Roundabout": "roundabout-sign",
                "Oneway": "oneway-sign", "No Entry": "noentry-sign"
            }
            for s in self.signs:
                if s['type'] in migration_map:
                    s['type'] = migration_map[s['type']]
                    
    def clear_route(self): 
        self.start_node = None; self.end_node = None; self.pass_node = None; self.path = []
        self.visited_path_nodes.clear()
        for item in self.tree.get_children(): self.tree.delete(item)
        if hasattr(self, 'acted_signs'): self.acted_signs.clear()
        self.render_map()
        self.log_event("Route cleared.", "WARN")
    def _on_key_press(self, e): 
        if e.keysym in self.keys: self.keys[e.keysym] = True
    def _on_key_release(self, e): 
        if e.keysym in self.keys: self.keys[e.keysym] = False
    def on_right_click(self, event): self.set_mode("DRIVE")
    def on_close(self):
        if self.picam2: self.picam2.stop()
        if self.yolo: self.yolo.stop()
        if self.is_connected: self.handler.set_speed(0); self.handler.set_steering(0); self.handler.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BFMC_Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
