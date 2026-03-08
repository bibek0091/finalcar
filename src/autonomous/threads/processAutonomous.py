import logging
import time
import base64
import os
import cv2
import numpy as np

from src.templates.workerprocess import WorkerProcess
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import (
    CurrentSpeed,
    Semaphores,
    Location,
    mainCamera,
)
from src.statemachine.stateMachine import StateMachine

import sys
# Ensure root directory is in sys.path for hardware and lane_detection imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Resolve YOLO model path from project root (not CWD-dependent)
_DEFAULT_MODEL_PATH = os.path.join(root_dir, "yolov8n.pt")
# Prefer Niranjan.pt if available next to this file
_NIRANJAN_MODEL_PATH = os.path.join(root_dir, "cleandash", "Niranjan.pt")
YOLO_MODEL_PATH = _NIRANJAN_MODEL_PATH if os.path.exists(_NIRANJAN_MODEL_PATH) else _DEFAULT_MODEL_PATH

# Local imports (absolute-path safe)
from lane_detection.lane_detector import LaneDetector
from lane_detection.controller import Controller
from lane_detection.traffic_module import ThreadedYOLODetector, TrafficDecisionEngine
from lane_detection.behavior_controller import BehaviorController
from hardware.serial_handler import STM32_SerialHandler
from hardware.imu_sensor import IMUSensor


def encode_to_b64(img):
    """Encode a BGR numpy frame as a base64 JPEG string for queue transport."""
    if img is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode('utf-8')


def annotate_bev(lane_result, control_output, t_res=None, behav_out=None):
    dbg = lane_result.lane_dbg.copy()

    def draw_poly(fit, color):
        if fit is None:
            return
        ys  = np.linspace(40, 479, 240).astype(np.float32)
        xs  = np.clip(np.polyval(fit, ys), 0, 639).astype(np.float32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(dbg, [pts], False, color, 3, cv2.LINE_AA)

    draw_poly(lane_result.sl, (255, 80, 80))
    draw_poly(lane_result.sr, (80, 80, 255))

    if lane_result.sl is not None and lane_result.sr is not None:
        lx = int(np.clip(np.polyval(lane_result.sl, 400), 0, 639))
        rx = int(np.clip(np.polyval(lane_result.sr, 400), 0, 639))
        cv2.line(dbg, (lx, 400), (rx, 400), (70, 170, 70), 1, cv2.LINE_AA)

    yrow = int(lane_result.y_eval)
    for xi in range(0, 640, 18):
        cv2.line(dbg, (xi, yrow), (xi + 9, yrow), (50, 200, 50), 1, cv2.LINE_AA)

    tx = max(4, min(636, int(lane_result.target_x)))
    for yi in range(360, 440, 12):
        cv2.line(dbg, (tx, yi), (tx, yi + 6), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dbg, (tx - 12, yrow), (tx + 12, yrow), (0, 255, 255), 2, cv2.LINE_AA)

    steer_color = (100, 255, 100) if abs(control_output.steer_angle_deg) < 15 else (100, 100, 255)
    cv2.putText(dbg, f"STEER: {control_output.steer_angle_deg:+.1f} deg", (420, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, steer_color, 2)
    cv2.putText(dbg, f"SPEED: {control_output.speed_pwm:.0f} PWM", (420, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

    if t_res is not None and behav_out is not None:
        cv2.putText(dbg, f"STATE: {behav_out.state}", (420, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(dbg, f"ZONE: {behav_out.zone_mode}", (420, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
    return dbg


class processAutonomous(WorkerProcess):
    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.logger = logging
        self.debugging = debugging
        self.qList = queueList

        # V2X state mirrors
        self._current_speed_mm_s = 0.0
        self._semaphores_dict    = {}
        self._location           = {}

        super(processAutonomous, self).__init__(queueList, ready_event)

    def _init_threads(self):
        # Main loop runs directly inside run() — no sub-threads needed
        pass

    def run(self):
        self._init_subscribers()

        # ── Hardware ──────────────────────────────────────────────────────────
        serial_handler = STM32_SerialHandler()
        if serial_handler.connect():
            self.logger.info("[HW] Connected to motor controller!")
        else:
            self.logger.warning("[HW] Could not connect to motor controller.")

        detector  = LaneDetector()
        controller = Controller()
        imu = IMUSensor()
        imu.start()

        # ── YOLO / Traffic ────────────────────────────────────────────────────
        try:
            yolo_detector  = ThreadedYOLODetector(YOLO_MODEL_PATH)
            traffic_engine = TrafficDecisionEngine(yolo_detector)
            behavior       = BehaviorController()
            self.logger.info(f"[SYS] YOLO model loaded: {YOLO_MODEL_PATH}")
            self.logger.info("[SYS] Behavioral Rule Engine ready.")
        except Exception as e:
            self.logger.warning(f"[SYS] Failed to load YOLO/Rule Engine: {e}")
            traffic_engine, behavior = None, None

        # ── Telemetry queue (Tkinter only) ────────────────────────────────────
        tk_telemetry_queue = self.qList.get("TkinterTelemetry")

        last_time   = time.time()
        last_steer  = 0.0
        target_fps  = 30
        frame_period = 1.0 / target_fps

        try:
            while True:
                # 1. Poll V2X queues (non-blocking) ──────────────────────────
                speed_recv = self.speedSubscriber.receive_nowait()
                if speed_recv is not None:
                    self._current_speed_mm_s = float(speed_recv)
                    try:
                        StateMachine.shared_memory["deviceSpeed"] = self._current_speed_mm_s / 10.0
                    except Exception:
                        pass

                sem_recv = self.semaphoreSubscriber.receive_nowait()
                if sem_recv is not None:
                    self._semaphores_dict = dict(sem_recv)

                # 2. Get camera frame ─────────────────────────────────────────
                frame_data = self.cameraSubscriber.receive_nowait()
                if frame_data is None or not isinstance(frame_data, str):
                    time.sleep(0.01)
                    continue

                try:
                    img_bytes = base64.b64decode(frame_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    self.logger.warning(f"[Camera] Base64 decode error: {e}")
                    continue

                if frame is None:
                    continue

                now    = time.time()
                dt     = max(now - last_time, 0.001)
                last_time = now

                # 3. Lane Detection & Control ──────────────────────────────────
                current_yaw  = imu.get_yaw()
                lane_result  = detector.process(
                    frame, dt=dt, velocity_ms=0.5,
                    last_steering=last_steer, current_yaw=current_yaw
                )
                control_output = controller.compute(
                    lane_result, velocity_ms=0.5, base_speed=50.0, dt=dt
                )
                last_steer = control_output.steer_angle_deg

                # 4. Traffic Semantics ─────────────────────────────────────────
                t_res, behav_out = None, None
                if traffic_engine and behavior:
                    t_res    = traffic_engine.process(frame, lane_result.lane_type)
                    behav_out = behavior.compute(
                        lane_result, t_res, dt,
                        base_steer=control_output.steer_angle_deg
                    )
                    control_output.speed_pwm      = behav_out.speed_pwm
                    control_output.steer_angle_deg = behav_out.steer_deg

                # 5. Send Commands to Hardware ─────────────────────────────────
                if serial_handler.running:
                    is_parking = behav_out and behav_out.maneuver == "PARKING"
                    if not imu.is_calibrated:
                        serial_handler.set_speed(0)
                        serial_handler.set_steering(0)
                    elif (
                        "DEAD_RECKONING" in lane_result.anchor
                        and lane_result.lateral_error_px == 0.0
                        and not is_parking
                    ):
                        # Dead reckoning with zero error — hold still
                        serial_handler.set_speed(0)
                        serial_handler.set_steering(0)
                    else:
                        serial_handler.set_speed(control_output.speed_pwm)
                        serial_handler.set_steering(control_output.steer_angle_deg)

                # 6. Push Telemetry to Tkinter Dashboard ──────────────────────
                # Skip expensive JPEG encoding if queue is already full
                if tk_telemetry_queue is not None and not tk_telemetry_queue.full():
                    bev_img  = annotate_bev(lane_result, control_output, t_res, behav_out)
                    yolo_img = t_res.yolo_debug_frame if t_res else frame

                    payload = {
                        "steer_angle":   control_output.steer_angle_deg,
                        "speed_pwm":     control_output.speed_pwm,
                        "lane_error":    lane_result.lateral_error_px,
                        "imu_calibrated": imu.is_calibrated,
                        "bev_b64":       encode_to_b64(bev_img),
                        "yolo_b64":      encode_to_b64(yolo_img),
                        "state":         behav_out.state if behav_out else "MANUAL",
                        "zone":          behav_out.zone_mode if behav_out else "UNKNOWN",
                        "active_labels": list(t_res.active_labels) if t_res else [],
                        "light_status":  t_res.light_status if t_res else "NONE",
                    }
                    try:
                        tk_telemetry_queue.put_nowait(payload)
                    except Exception:
                        pass  # Queue went full between check and put — acceptable drop

                # 7. Frame pacing ──────────────────────────────────────────────
                elapsed    = time.time() - now
                sleep_time = max(0.001, frame_period - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Autonomous process interrupted.")
        finally:
            imu.stop()
            serial_handler.disconnect()

    def _init_subscribers(self):
        self.speedSubscriber = messageHandlerSubscriber(self.qList, CurrentSpeed, "lastOnly", True)
        self.semaphoreSubscriber = messageHandlerSubscriber(self.qList, Semaphores, "lastOnly", True)
        self.locationSubscriber  = messageHandlerSubscriber(self.qList, Location, "lastOnly", True)
        self.cameraSubscriber    = messageHandlerSubscriber(self.qList, mainCamera, "lastOnly", True)

    def _init_senders(self):
        pass  # Hardware writes handled directly via STM32_SerialHandler
