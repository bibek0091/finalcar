import cv2
import time
import numpy as np
import argparse
from lane_detection.camera import Camera
from lane_detection.lane_detector import LaneDetector
from lane_detection.controller import Controller
from hardware.serial_handler import STM32_SerialHandler
from hardware.imu_sensor import IMUSensor
from lane_detection.traffic_module import ThreadedYOLODetector, TrafficDecisionEngine
from lane_detection.behavior_controller import BehaviorController

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

    # Annotate Width
    if lane_result.sl is not None and lane_result.sr is not None:
        lx = int(np.clip(np.polyval(lane_result.sl, 400), 0, 639))
        rx = int(np.clip(np.polyval(lane_result.sr, 400), 0, 639))
        cv2.line(dbg, (lx, 400), (rx, 400), (70, 170, 70), 1, cv2.LINE_AA)
        cv2.putText(dbg, f"w={lane_result.lane_width_px:.0f}px", ((lx+rx)//2-20, 396),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 170, 70), 1)

    # Draw target crosshair
    yrow = int(lane_result.y_eval)
    for xi in range(0, 640, 18):
        cv2.line(dbg, (xi, yrow), (xi+9, yrow), (50, 200, 50), 1, cv2.LINE_AA)

    tx = max(4, min(636, int(lane_result.target_x)))
    for yi in range(360, 440, 12):
        cv2.line(dbg, (tx, yi), (tx, yi+6), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(dbg, (tx-12, yrow), (tx+12, yrow), (0, 255, 255), 2, cv2.LINE_AA)

    # Add text info
    cv2.putText(dbg, lane_result.anchor, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(dbg, f"Target X: {lane_result.target_x:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(dbg, f"Lat Error: {lane_result.lateral_error_px:+.1f}px", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    # Motor annotations
    steer_color = (100,255,100) if abs(control_output.steer_angle_deg)<15 else (100,100,255)
    cv2.putText(dbg, f"STEER: {control_output.steer_angle_deg:+.1f} deg", (420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, steer_color, 2)
    cv2.putText(dbg, f"SPEED: {control_output.speed_pwm:.0f} PWM", (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
    
    if t_res is not None and behav_out is not None:
        cv2.putText(dbg, f"STATE: {behav_out.state}", (420, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(dbg, f"ZONE: {behav_out.zone_mode}", (420, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 2)
        
        # Draw YOLO detections dynamically onto the BEV window upper-left
        y_offset = 100
        cv2.putText(dbg, "YOLO Detections:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for label in t_res.active_labels:
            y_offset += 20
            cv2.putText(dbg, f"- {label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    return dbg


def main():
    parser = argparse.ArgumentParser(description="BFMC Tempfile Stack for Raspberry Pi")
    parser.add_argument("--headless", action="store_true", help="Run without UI displays (avoids X11 crashes on headless Pi)")
    parser.add_argument("--model", type=str, default="temp/tempfile/lane_detection/lane_detection/best.pt", help="Path to YOLO model (e.g. best.onnx for Pi speedup)")
    args = parser.parse_args()

    print("\n--- Initializing Autonomous Lane Follower ---")
    if args.headless:
        print("[SYS] Running in HEADLESS mode (No GUI windows will be shown).")
    
    # Hardware/Serial (STM32 connection)
    serial_handler = STM32_SerialHandler()
    if serial_handler.connect():
        print("[HW] Connected to motor controller!")
    else:
        print("[HW] Warning: Could not connect to motor controller. Running in visually-only mode.")

    # Camera Pipeline
    camera = Camera(sim_video=None) 
    detector = LaneDetector()
    
    # Steering Model
    controller = Controller()
    
    # IMU Hardware Integration
    imu = IMUSensor()
    imu.start()
    
    # Traffic Semantics & Rule Engine
    try:
        yolo_detector = ThreadedYOLODetector(args.model)
        traffic_engine = TrafficDecisionEngine(yolo_detector)
        behavior = BehaviorController()
        print("[SYS] YOLOv11 and BFMC Semantic Rule Engine loaded.")
    except Exception as e:
        print(f"[SYS] Warning: Failed to load Rule Engine: {e}")
        yolo_detector = None
        traffic_engine = None
        behavior = None

    print("[SYS] Pipeline ready. Press 'q' in the view window to exit.")
    
    last_time = time.time()
    last_steer = 0.0
    
    target_fps = 30
    frame_period = 1.0 / target_fps
    
    try:
        while True:
            # 1. Grab Frame
            frame = camera.read_frame()
            if frame is None:
                # Wait for frames to buffer
                time.sleep(0.01)
                continue
            
            # Timestamp calculation
            now = time.time()
            dt = max(now - last_time, 0.001)
            last_time = now

            # 2. Process frame for Lane Detection
            current_yaw = imu.get_yaw()
            lane_result = detector.process(frame, dt=dt, velocity_ms=0.5, last_steering=last_steer, current_yaw=current_yaw)
            
            # 3. Calculate Steering & Speed Control
            # Assuming ~0.0 velocity_ms for now as encoder feedback is detached without IMU
            control_output = controller.compute(lane_result, velocity_ms=0.5, base_speed=50.0, dt=dt)
            last_steer = control_output.steer_angle_deg
            
            # 3.5 Semantic Traffic Overrides
            t_res = None
            behav_out = None
            if traffic_engine and behavior:
                line_type = getattr(lane_result, 'lane_type', 'UNKNOWN')
                t_res = traffic_engine.process(frame, line_type)
                behav_out = behavior.compute(
                    lane_result, 
                    t_res, 
                    dt, 
                    base_steer=control_output.steer_angle_deg
                )
                
                # Semantic overrides win over raw physics steering
                control_output.speed_pwm = behav_out.speed_pwm
                control_output.steer_angle_deg = behav_out.steer_deg
            
            # 4. Dispatch Hardware Commands
            if serial_handler.running:
                # Hardware failsafe logic: stop motors if dead reckoning 
                if not imu.is_calibrated:
                    serial_handler.set_speed(0)
                    serial_handler.set_steering(0)
                elif "DEAD_RECKONING" in lane_result.anchor and "0.00" in lane_result.anchor:
                    serial_handler.set_speed(0)
                    serial_handler.set_steering(0)
                    print("LOST LANES: Motors paused.")
                else:    
                    serial_handler.set_speed(control_output.speed_pwm)
                    serial_handler.set_steering(control_output.steer_angle_deg)
            
            # 5. Create Bird's Eye View overlay
            bev_image = annotate_bev(lane_result, control_output, t_res, behav_out)
            if not imu.is_calibrated:
                cv2.putText(bev_image, "IMU CALIBRATING - DO NOT DRIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                control_output.speed_pwm = 0.0

            # 6. Display the results
            if not args.headless:
                cv2.imshow("Lane Detection (BEV)", bev_image)
                if t_res and t_res.yolo_debug_frame is not None:
                    cv2.imshow("Raw Camera", cv2.resize(t_res.yolo_debug_frame, (640, 480)))
                else:
                    cv2.imshow("Raw Camera", cv2.resize(frame, (640, 480)))
            
            # Pacing
            elapsed = time.time() - now
            sleep_time = max(0.001, frame_period - elapsed)
            time.sleep(sleep_time)
            
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up hardware connections...")
        imu.stop()
        if yolo_detector:
            yolo_detector.stop()
        serial_handler.disconnect()
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
