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
    dbg = lane_result.lane_dbg.copy()

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
        for label in t_res.active_labels:
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
        
        # 6. Optional: Render visualization window (Only if not headless)
        try:
            dbg_frame = annotate_bev(lane_result, control_output, t_res if self.traffic_engine else None, behav_out if self.behavior else None)
            
            # Prevent Fatal X11 C++ aborts on Headless Raspberry Pi
            if os.environ.get('DISPLAY'):
                cv2.imshow("BFMC Semantic Brain", dbg_frame)
                cv2.waitKey(1)
            else:
                # Optionally write to disk for debugging or just pass
                pass
        except Exception as e:
            pass
