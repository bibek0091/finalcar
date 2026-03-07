# config.py
import math

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

HIGHWAY_SPEED_PWM = 300   

CAMERA_FOCAL_LENGTH_PX = 450.0 
REAL_SIGN_HEIGHT_M = 0.08      
REAL_WIDTH_M  = 22.0
REAL_HEIGHT_M = 15.0

# --- PHYSICAL CONSTANTS ---
WHEELBASE_M = 0.23    

THEME = {
    "bg": "#1e1e1e", "panel": "#252526", "canvas": "#111111", 
    "fg": "#cccccc", "accent": "#007acc", "danger": "#f44336", "success": "#4caf50",
    "warning": "#ff9800", "font_h": ("Helvetica", 11, "bold"), "font_p": ("Helvetica", 10),
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