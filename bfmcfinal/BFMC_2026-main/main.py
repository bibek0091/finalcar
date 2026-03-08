# main.py
import tkinter as tk
from tkinter import messagebox
import time
import math
import numpy as np
import cv2
import csv
import json
import os
import sys
import subprocess
import argparse
from PIL import Image, ImageTk

# ── IMPORTS FROM PACKAGES ──────────────────────────────────
from config import *
from dashboard.dashboard_ui import DashboardUI
from dashboard.map_engine import MapEngine
from dashboard.adas_vision_utils import annotate_bev, JunctionDetector, RoundaboutNavigator

try:
    from hardware.serial_handler import STM32_SerialHandler
except ImportError:
    class STM32_SerialHandler:
        def __init__(self): self.running = False
        def connect(self): return True
        def disconnect(self): pass
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def set_light_state(self, state, on): pass

try:
    from hardware.imu_sensor import IMUSensor
except ImportError:
    class IMUSensor:
        def __init__(self): self.is_calibrated = True
        def start(self): pass
        def stop(self): pass
        def get_yaw(self): return 0.0

try:
    from v2x.v2x_client import V2XClient
except ImportError:
    class V2XClient:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def update_state(self, *args, **kwargs): pass

# ── AUTONOMOUS STACK IMPORTS (from tempfile) ───────────────
try:
    from perception.camera import Camera
    from perception.lane_detector import LaneDetector
    from control.controller import Controller
    _AUTO_DRIVE_AVAILABLE = True
except ImportError:
    _AUTO_DRIVE_AVAILABLE = False

try:
    from traffic.traffic_module import TrafficDecisionEngine, ThreadedYOLODetector
    from traffic.behavior_controller import BehaviorController
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  V2X LAUNCHER
# ─────────────────────────────────────────────────────────────
def launch_v2x_servers():
    """Launch V2X infrastructure as background subprocesses."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(base_dir, "servers", "trafficCommunicationServer")
    sim_dir = os.path.join(base_dir, "servers", "carsAndSemaphoreStreamSIM")

    procs = []
    print("\n--- Starting V2X Servers ---")
    try:
        # 1. Traffic Communication Server (TCP:5000 + UDP:9000 + LocSys:4691)
        p1 = subprocess.Popen([sys.executable, "TrafficCommunication.py"], cwd=server_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p1)
        print("[V2X] TrafficCommunication Server started.")

        # 2. Semaphore + Car Simulator (UDP:5007) 
        p2 = subprocess.Popen([sys.executable, "udpStreamSIM.py"], cwd=sim_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p2)
        print("[V2X] Semaphore + Car Simulator started.")
    except Exception as e:
        print(f"[V2X] Warning: Could not start servers: {e}")
    return procs


# ─────────────────────────────────────────────────────────────
class MockCtrl: pass

# ─────────────────────────────────────────────────────────────
class BFMC_App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.headless = args.headless
        
        if not self.headless:
            self.root.title("TEAM OPTINX BFMC 2026")
            self.root.geometry("1400x850")
            self.root.minsize(1200, 700)
            self.root.configure(bg=THEME["bg"])
            self.ui = DashboardUI(self.root, self)
        
        self.map_engine = MapEngine()

        # Hardware setup
        self.handler = STM32_SerialHandler()
        self.is_connected = False
        
        self.imu = IMUSensor()
        self.imu.start()

        # V2X Client (Daemon thread)
        self.v2x_client = V2XClient(host=V2X_SERVER_HOST, port=V2X_SERVER_PORT)
        if not args.no_v2x:
            self.v2x_client.start()

        # Physics State
        self.car_x, self.car_y, self.car_yaw = 0.5, 0.5, 0.0
        self.current_speed, self.current_steer = 0.0, 0.0
        self.keys = {'Up': False, 'Down': False, 'Left': False, 'Right': False}
        self.last_ctrl_time = time.time()
        self.current_hz = 0.0
        self.is_calibrating = False

        # ADAS State
        self.adas_enabled = True
        self.in_highway_mode = False

        # Routing State
        self.mode = "DRIVE"
        self.start_node = None; self.end_node = None; self.pass_node = None
        self.path = []
        self.visited_path_nodes = set()
        self.path_signs = []
        self.visible_signs = {}

        # Parking/Playback State
        self.is_playing_back = False
        self.is_parking_reverse_mode = False
        self.playback_queue = []; self.playback_cmd = None; self.playback_frames = 0

        # Autonomous Pipelines (Lane Detection)
        self.is_auto_mode    = False
        self.auto_start_time = 0.0
        
        self.camera = Camera(sim_video=None)
        self.detector = LaneDetector() if _AUTO_DRIVE_AVAILABLE else None
        self.controller = Controller() if _AUTO_DRIVE_AVAILABLE else None

        self.traffic_engine, self.behavior, self.yolo = None, None, None
        if _AI_AVAILABLE:
            try:
                self.yolo = ThreadedYOLODetector(YOLO_MODEL_FILE)
                self.traffic_engine = TrafficDecisionEngine(self.yolo)
                self.behavior = BehaviorController()
            except Exception as e:
                print(f"[SYS] Warning: Failed to load AI models: {e}")

        # Bindings & Loops
        if not self.headless:
            self.root.bind("<KeyPress>", self._on_key_press)
            self.root.bind("<KeyRelease>", self._on_key_release)
        
        self.set_mode("DRIVE")
        self.control_loop()
        
        if not self.headless:
            self.render_map()

    def set_mode(self, m):
        self.mode = m
        if self.headless: return
        self.ui.var_main_mode.set(m)
        for w in self.ui.tool_frame.winfo_children():
            w.destroy()
        if m == "NAV":
            self.ui.build_nav_tools(self)
        elif m == "SIGN":
            self.ui.build_sign_tools(self)
        else:
            tk.Label(self.ui.tool_frame,
                     text="Drive Mode Active - Click Map to Teleport Digital Twin",
                     bg=THEME["panel"], fg=THEME["success"],
                     font=THEME["font_h"]).pack(side=tk.LEFT, padx=10, pady=5)

    def on_map_click(self, event):
        if self.headless: return
        canvas_x = self.ui.map_canvas.canvasx(event.x)
        canvas_y = self.ui.map_canvas.canvasy(event.y)
        mx, my = self.map_engine.to_meter(canvas_x, canvas_y)
        
        if self.mode == "DRIVE":
            self.car_x, self.car_y = mx, my
            self.render_map()

    def render_map(self):
        if self.headless: return
        pil = self.map_engine.render_map(
            self.car_x, self.car_y, self.car_yaw,
            self.path, self.visited_path_nodes, self.path_signs,
            True, self.start_node, self.pass_node, self.end_node
        )
        self.tk_map = ImageTk.PhotoImage(pil)
        self.ui.map_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_map)
        self.ui.map_canvas.config(scrollregion=self.ui.map_canvas.bbox(tk.ALL))

    def toggle_connection(self):
        if not self.is_connected:
            if self.handler.connect():
                self.is_connected = True
                if not self.headless:
                    self.ui.lbl_conn.config(text="🟢 CONNECTED", fg=THEME["success"])
                    self.ui.btn_connect.config(text="DISCONNECT", bg=THEME["danger"])
                    self.ui.log_event("🔗 Connected to STM32 Hardware successfully.", "SUCCESS")
        else:
            self.handler.disconnect(); self.is_connected = False
            if not self.headless:
                self.ui.log_event("🔌 Disconnected from STM32.", "WARN")
                self.ui.lbl_conn.config(text="⚫ DISCONNECTED", fg=THEME["danger"])
                self.ui.btn_connect.config(text="CONNECT CAR", bg=THEME["accent"])

    def toggle_auto_mode(self):
        self.is_auto_mode = not self.is_auto_mode
        if self.is_auto_mode:
            self.auto_start_time = time.time()
            self.is_calibrating  = True
            for k in self.keys: self.keys[k] = False
            if not self.headless:
                self.ui.btn_auto.config(text="MODE: AUTONOMOUS", bg="#9b59b6")
                self.ui.log_event("🤖 Switched to AUTONOMOUS. Calibrating 5s …", "SUCCESS")
        else:
            self.is_calibrating  = False
            for k in self.keys: self.keys[k] = False
            if not self.headless:
                self.ui.btn_auto.config(text="MODE: MANUAL", bg="#444")
                self.ui.log_event("🖐 Switched to MANUAL mode.", "WARN")

    # ─────────────────────────────────────────────────────────
    # CONTROL LOOP  (20 Hz)
    # ─────────────────────────────────────────────────────────
    def control_loop(self):
        now = time.time()
        dt = max(now - self.last_ctrl_time, 0.001)
        self.last_ctrl_time = now

        base_speed = float(self.ui.slider_base_speed.get() if not self.headless else 50.0)
        steer_mult = float(self.ui.slider_steer_mult.get() if not self.headless else 1.0)

        target_speed, target_steer = 0.0, 0.0
        
        # 1. Grab Frame
        frame = self.camera.read_frame()
        lane_result = None
        t_res = None
        behav_out = None

        if frame is not None and self.detector and self.controller:
            # 2. Process Lane Detection
            # Inject IMU yaw for Dead Reckoning
            yaw_deg = self.imu.get_yaw()
            lane_result = self.detector.process(
                frame, dt=dt, velocity_ms=max(self.current_speed / 1000.0, 0.0), 
                last_steering=self.current_steer, current_yaw=yaw_deg
            )
            
            # 3. Calculate Steering & Speed Control
            if self.is_auto_mode:
                if time.time() - self.auto_start_time > 5.0 and self.imu.is_calibrated:
                    self.is_calibrating = False
                    ctrl_out = self.controller.compute(lane_result, velocity_ms=max(self.current_speed / 1000.0, 0.0), base_speed=base_speed, dt=dt)
                    target_speed = ctrl_out.speed_pwm
                    target_steer = ctrl_out.steer_angle_deg * steer_mult
                    
                    # 4. Semantic Traffic Overrides
                    if self.traffic_engine and self.behavior:
                        line_type = getattr(lane_result, 'lane_type', 'UNKNOWN')
                        t_res = self.traffic_engine.process(frame, line_type)
                        behav_out = self.behavior.compute(
                            lane_result, t_res, dt, base_steer=ctrl_out.steer_angle_deg
                        )
                        # Semantic overrides win over raw physics steering
                        target_speed = behav_out.speed_pwm
                        target_steer = behav_out.steer_deg
                else:
                    self.is_calibrating = True
                    target_speed, target_steer = 0.0, 0.0
                    
            # 5. Dashboard CAM + BEV Render
            if not self.headless:
                final_cam = t_res.yolo_debug_frame if (t_res and getattr(t_res, 'yolo_debug_frame', None) is not None) else frame
                final_cam = cv2.cvtColor(final_cam, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(final_cam).resize((440, 330))
                self.ui.cam_label.imgtk = ImageTk.PhotoImage(image=img)
                self.ui.cam_label.configure(image=self.ui.cam_label.imgtk)

                if hasattr(lane_result, 'lane_dbg'):
                    dbg = lane_result.lane_dbg.copy()

                    # Add text info (matching tempfile behavior)
                    cv2.putText(dbg, lane_result.anchor, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(dbg, f"Target X: {lane_result.target_x:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(dbg, f"Lat Error: {lane_result.lateral_error_px:+.1f}px", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    
                    steer_color = (100,255,100) if abs(self.current_steer)<15 else (100,100,255)
                    cv2.putText(dbg, f"STEER: {self.current_steer:+.1f} deg", (420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, steer_color, 2)
                    cv2.putText(dbg, f"SPEED: {self.current_speed:.0f} PWM", (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
                    
                    if t_res is not None and behav_out is not None:
                        cv2.putText(dbg, f"STATE: {behav_out.state}", (420, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(dbg, f"ZONE: {behav_out.zone_mode}", (420, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
                        y_offset = 120
                        if t_res.active_labels:
                            cv2.putText(dbg, "YOLO Detections:", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            for label in t_res.active_labels:
                                cv2.putText(dbg, f"- {label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                                y_offset += 20

                    bev = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
                    img_bev = Image.fromarray(bev).resize((440, 330))
                    self.ui.bev_label.imgtk = ImageTk.PhotoImage(image=img_bev)
                    self.ui.bev_label.configure(image=self.ui.bev_label.imgtk)

        # ── MANUAL OVERRIDES ──────────────────────────────────
        if not self.is_auto_mode:
            self.is_calibrating = False
            target_speed = (base_speed  if self.keys['Up'] else (-base_speed if self.keys['Down'] else 0))
            target_steer = (-25 * steer_mult if self.keys['Left'] else (25 * steer_mult if self.keys['Right'] else 0))

        # ── SMOOTH APPLICATION ────────────────────────────────
        if target_speed == 0:
            self.current_speed = 0.0 
        else:
            self.current_speed += (target_speed - self.current_speed) * 0.2

        if target_steer == 0:
            self.current_steer = 0.0 
        else:
            self.current_steer += (target_steer - self.current_steer) * 0.2

        # ── HARDWARE OUTPUT ───────────────────────────────────
        if self.is_connected:
            if not self.imu.is_calibrated and (self.is_auto_mode):
                self.handler.set_speed(0)
                self.handler.set_steering(0)
            else:
                self.handler.set_speed(int(self.current_speed))
                self.handler.set_steering(self.current_steer)

        # ── V2X TELEMETRY PUSH ────────────────────────────────
        # Update client thread with current values
        yaw_rad = math.radians(self.imu.get_yaw())
        self.v2x_client.update_state(
            x=self.car_x,
            y=self.car_y,
            yaw=self.imu.get_yaw(),
            speed=self.current_speed
        )

        # ── KINEMATICS SIMULATION (Map update) ────────────────
        if abs(self.current_speed) < 1:  self.current_speed = 0
        if abs(self.current_steer) < 0.5: self.current_steer = 0

        sim_mult = float(self.ui.slider_sim_speed.get() if not self.headless else 1.0)
        v_ms = (self.current_speed / 1000.0) * sim_mult * 1.5
        steer_rad = math.radians(self.current_steer)
        self.car_yaw -= (v_ms / max(WHEELBASE_M, 0.01)) * math.tan(steer_rad) * dt
        self.car_yaw  = (self.car_yaw + math.pi) % (2 * math.pi) - math.pi
        self.car_x   += v_ms * math.cos(self.car_yaw) * dt
        self.car_y   += v_ms * math.sin(self.car_yaw) * dt

        # ── UI UPDATES ────────────────────────────────────────
        if not self.headless:
            hz = 1.0 / dt if dt > 0 else 0.0
            self.current_hz = 0.8 * self.current_hz + 0.2 * hz
            self.ui.lbl_hz.config(text=f"{self.current_hz:.1f} Hz", fg="cyan")
            
            mode_str = "AUTONOMOUS" if self.is_auto_mode else "MANUAL"
            if self.is_calibrating: mode_str = "CALIBRATING..."
            self.ui.lbl_telemetry.config(
                text=f"SPD: {int(self.current_speed)} | STR: {self.current_steer:.1f}° | LMT: {base_speed} | [{mode_str}]"
            )
            self.render_map()

        if self.headless:
            print(f"[CTRL] Spd:{int(self.current_speed):4d} | Str:{self.current_steer:5.1f}° | Yaw:{self.imu.get_yaw():5.1f}° | Pos:({self.car_x:.1f},{self.car_y:.1f}) | {1/dt:.0f}Hz", end="\r")

        if self.headless:
            # headless equivalent of root.after
            time.sleep(0.05)
            self.control_loop()
        else:
            self.root.after(50, self.control_loop)


    def save_config(self):
        pass

    def load_config(self):
        pass

    def toggle_adas_mode(self):
        self.adas_enabled = not self.adas_enabled
        if not self.headless:
            if self.adas_enabled:
                self.ui.btn_adas.config(text="ADAS ASSIST: ON", bg="#9b59b6")
                self.ui.log_event("✅ ADAS ASSIST enabled.", "SUCCESS")
            else:
                self.ui.btn_adas.config(text="ADAS ASSIST: OFF", bg="#444")
                self.ui.log_event("⚠️ ADAS ASSIST DISABLED.", "WARN")

    def clear_route(self):
        self.start_node = None; self.end_node = None; self.pass_node = None; self.path = []
        self.visited_path_nodes.clear()
        if not self.headless:
            for item in self.ui.tree.get_children():
                self.ui.tree.delete(item)
            self.render_map()
            self.ui.log_event("🗑 Route & sign history cleared. Ready for new run.", "WARN")

    def _on_key_press(self, e):
        if self.is_auto_mode or self.is_playing_back: return
        if e.keysym in self.keys: self.keys[e.keysym] = True

    def _on_key_release(self, e):
        if e.keysym in self.keys: self.keys[e.keysym] = False

    def on_close(self):
        self.camera.stop()
        if self.yolo: self.yolo.stop()
        if self.is_connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        self.imu.stop()
        self.v2x_client.stop()
        if not self.headless:
            self.root.destroy()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFMC 2026 Unified Autonomous Stack")
    parser.add_argument("--headless", action="store_true", help="Run in terminal only, no Tkinter GUI")
    parser.add_argument("--no-v2x", action="store_true", help="Do not start the background V2X servers")
    args = parser.parse_args()

    v2x_procs = []
    if not args.no_v2x:
        pass # User manually runs the servers in separate terminals

    try:
        if args.headless:
            # Fake root object
            class FakeRoot: pass
            app = BFMC_App(FakeRoot(), args)
            # control_loop calls itself via sleep in headless mode, so this will block
        else:
            root = tk.Tk()
            app = BFMC_App(root, args)
            root.protocol("WM_DELETE_WINDOW", app.on_close)
            root.mainloop()

    except KeyboardInterrupt:
        print("\n[SYS] Interrupted by user.")
    except Exception as e:
        import traceback
        print("\n[SYS] FATAL ERROR:")
        traceback.print_exc()
    finally:
        if not args.headless:
            try: app.on_close()
            except: pass
        print("\n[SYS] Cleaning up V2X servers...")
        for p in v2x_procs:
            p.terminate()
