"""
processAutonomous.py
====================
BFMC Autonomous Module – Main Semantic Engine
Fuses Lane Tracking, YOLO detection, and Behavior FSMs.
"""

import time
import os
import queue
import logging
import base64
import cv2
import numpy as np
from src.templates.workerprocess import WorkerProcess
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import SpeedMotor, SteerMotor, CurrentSpeed

# Import our custom stack tools
from src.hardware.imu.imu_sensor import IMUSensor
from src.autonomous.utils.lane_detector import LaneDetector
from src.autonomous.utils.controller import Controller
from src.dashboard.traffic_module import ThreadedYOLODetector, TrafficDecisionEngine
from src.autonomous.utils.behavior_controller import BehaviorController

def annotate_bev(lane_result, control_output, t_res=None, behav_out=None):
    raw = lane_result.lane_dbg
    if raw is None:
        raw = lane_result.warped_binary
    if raw is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    dbg = raw.copy()
    if len(dbg.shape) == 2:  # grayscale -> BGR
        dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

    def draw_poly(fit, color):
        if fit is None: return
        ys  = np.linspace(40,479,240).astype(np.float32)
        xs  = np.clip(np.polyval(fit,ys),0,639).astype(np.float32)
        pts = np.stack([xs,ys],axis=1).reshape(-1,1,2).astype(np.int32)
        cv2.polylines(dbg,[pts],False,color,3,cv2.LINE_AA)

    # Draw lane polynomials
    draw_poly(lane_result.sl, (255, 80, 80))  # Left line
    draw_poly(lane_result.sr, (80, 80, 255))  # Right line

    # Target crosshair
    yrow = int(lane_result.y_eval)
    tx = max(4, min(636, int(lane_result.target_x)))
    cv2.line(dbg, (tx-12, yrow), (tx+12, yrow), (0, 255, 255), 2, cv2.LINE_AA)
    
    # Motor annotations
    steer_color = (100,255,100) if abs(control_output.steer_angle_deg)<15 else (100,100,255)
    cv2.putText(dbg, f"STEER: {control_output.steer_angle_deg:+.1f} deg", (420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, steer_color, 2)
    cv2.putText(dbg, f"SPEED: {control_output.speed_pwm:.0f} PWM", (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
    
    if t_res is not None and behav_out is not None:
        cv2.putText(dbg, f"STATE: {behav_out.state}", (420, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(dbg, f"ZONE: {behav_out.zone_mode}", (420, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
        
        y_offset = 100
        cv2.putText(dbg, "YOLO Detections:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for label in getattr(t_res, 'active_labels', []) or []:
            y_offset += 20
            cv2.putText(dbg, f"- {label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    return dbg

class processAutonomous(WorkerProcess):
    def __init__(self, queueList, logging_obj, ready_event=None, debugging=False):
        self.queuesList = queueList
        self.logger = logging_obj
        self.debugging = debugging
        
        # Init publishers/subscribers
        self.speedSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.currentSpeedSubscriber = messageHandlerSubscriber(self.queuesList, CurrentSpeed, "lastOnly", True)

        # Init the core Lane and Semantic Models
        self.detector = LaneDetector()
        self.controller = Controller()
        try:
            self.imu = IMUSensor()
            self.imu.start()
            self.logger.info("[Autonomous] IMU hardware listener started.")
        except Exception as e:
            self.logger.warning(f"[Autonomous] Failed to load IMU: {e}")
            self.imu = None
        
        # Determine model path unconditionally
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "models", "best.pt"))
        
        try:
            self.yolo_detector = ThreadedYOLODetector(model_path)
            self.traffic_engine = TrafficDecisionEngine(self.yolo_detector)
            self.behavior = BehaviorController()
            self.logger.info("[Autonomous] YOLOv11 & Behavior FSM loaded.")
        except Exception as e:
            self.logger.warning(f"[Autonomous] Failed to load YOLO/FSM: {e}")
            self.yolo_detector = None
            self.traffic_engine = None
            self.behavior = None
            
        self.last_time = time.time()
        self.last_steer = 0.0
        self._dash_counter = 0  # throttle dashboard updates to ~5 fps

        super(processAutonomous, self).__init__(self.queuesList, ready_event)

    def _init_threads(self):
        pass

    def process_work(self):
        # 1. Grab raw numpy frame from the fast-path Vision queue
        if "Vision" not in self.queuesList:
            time.sleep(0.01)
            return
            
        try:
            frame = self.queuesList["Vision"].get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            return

        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now

        # Read IMU for dead reckoning (if available)
        current_yaw = self.imu.get_yaw() if self.imu else 0.0
        
        # Fake velocity until encoders mapped
        velocity_ms = 0.5 

        # 2. Process Lane Detection
        lane_result = self.detector.process(frame, dt=dt, velocity_ms=velocity_ms, last_steering=self.last_steer, current_yaw=current_yaw)
        
        # 3. Process Base Steering
        control_output = self.controller.compute(lane_result, velocity_ms=velocity_ms, base_speed=50.0, dt=dt)
        base_steer = control_output.steer_angle_deg
        self.last_steer = base_steer
        
        final_speed = float(control_output.speed_pwm)
        final_steer = float(base_steer)
        
        # 4. Semantic Traffic Overrides (Stop signs, Pedestrians, Traffic Lights)
        t_res = None
        behav_out = None
        if self.traffic_engine and self.behavior:
            line_type = getattr(lane_result, 'lane_type', 'UNKNOWN')
            t_res = self.traffic_engine.process(frame, line_type)
            
            behav_out = self.behavior.compute(
                perc_res=lane_result,
                t_res=t_res,
                dt=dt,
                base_steer=base_steer
            )
            
            if behav_out:
                final_speed = behav_out.speed_pwm
                final_steer = behav_out.steer_deg

        # 5. Dispatch Motor Commands
        self.speedSender.send(str(int(final_speed)))
        self.steerSender.send(str(int(final_steer)))
        
        # 6. Stream to Dashboard (throttled to ~5 fps)
        self._dash_counter += 1
        if self._dash_counter % 6 == 0:  # every ~6 frames ≈ 5fps at 30fps input
            try:
                # BEV debug frame
                bev_q = self.queuesList.get("DashBEV")
                if bev_q is not None:
                    bev_dbg = annotate_bev(
                        lane_result, control_output, t_res, behav_out,
                    )
                    # Convert BGR→RGB before JPEG encoding (browsers render JPEGs as RGB)
                    _, buf = cv2.imencode('.jpg', cv2.cvtColor(bev_dbg, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_JPEG_QUALITY, 60])
                    b64 = base64.b64encode(buf).decode('ascii')
                    if bev_q.full():
                        try: bev_q.get_nowait()
                        except: pass
                    bev_q.put_nowait(b64)

                # YOLO annotated frame — use the debug frame already built by TrafficDecisionEngine
                # (it is built directly on the BGR camera frame with bboxes drawn)
                yolo_q = self.queuesList.get("DashYOLO")
                if yolo_q is not None:
                    if t_res is not None and getattr(t_res, 'yolo_debug_frame', None) is not None:
                        yolo_frame = t_res.yolo_debug_frame
                    else:
                        yolo_frame = frame.copy()
                    # Convert BGR→RGB before JPEG encoding (browsers render JPEGs as RGB)
                    _, buf = cv2.imencode('.jpg', cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_JPEG_QUALITY, 65])
                    b64 = base64.b64encode(buf).decode('ascii')
                    if yolo_q.full():
                        try: yolo_q.get_nowait()
                        except: pass
                    yolo_q.put_nowait(b64)

                # Decision state JSON
                dec_q = self.queuesList.get("DashDecision")
                if dec_q is not None:
                    yolo_labels = getattr(t_res, 'active_labels', []) if t_res else []
                    decision = {
                        'state': behav_out.state if behav_out else 'LANE_FOLLOW',
                        'reason': behav_out.reason if behav_out else 'lane tracking',
                        'priority': behav_out.priority if behav_out else 10,
                        'zone': behav_out.zone_mode if behav_out else 'CITY',
                        'speed': final_speed,
                        'steer': final_steer,
                        'yolo_labels': yolo_labels,
                    }
                    if dec_q.full():
                        try: dec_q.get_nowait()
                        except: pass
                    dec_q.put_nowait(decision)
            except Exception:
                pass  # Dashboard streaming must never crash the autonomous loop
