"""
processAutonomous.py
====================
BFMC Autonomous Module – Main Autonomous Driving WorkerProcess

This is the brain of the autonomous car. It fuses:
  1. Camera-based lane keeping (HybridLaneTracker)
  2. Intersection detection (JunctionDetector)
  3. Topological dead-reckoning navigation (TopologicalNavigator)
  4. V2X semaphore state (UDP port 5007 via processSemaphores)
  5. YOLO object detection results (from processVision)

And publishes to:
  - SpeedMotor queue  → threadWrite → STM32 (speed PWM)
  - SteerMotor queue  → threadWrite → STM32 (steer degrees)
  - StateMachine.shared_memory → LiveTraffic telemetry (deviceSpeed, historyData)

Message Flow
------------
  processSerialHandler --[CurrentSpeed]--> processAutonomous
  processSemaphores    --[Semaphores]  --> processAutonomous
  processTrafficCom    --[Location]    --> processAutonomous
  processVision        --[YoloDetection]->processAutonomous
  processCamera        --[mainCamera]  --> processAutonomous
  processAutonomous    --[SpeedMotor]  --> threadWrite --> STM32
  processAutonomous    --[SteerMotor]  --> threadWrite --> STM32

Metric Conversions (heavily commented per spec)
------------------------------------------------
  CurrentSpeed is in mm/s (millimeters per second).

  For odometry (distance integration):
      speed_m_s = speed_mm_s / 1000.0     (mm/s ÷ 1000 = m/s)

  For LiveTraffic telemetry deviceSpeed field:
      speed_cm_s = speed_mm_s / 10.0      (mm/s ÷ 10 = cm/s)
"""

import logging
import time
import cv2
import numpy as np

from src.templates.workerprocess import WorkerProcess
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import (
    CurrentSpeed,
    Semaphores,
    Location,
    YoloDetection,
    mainCamera,
    SpeedMotor,
    SteerMotor,
)
from src.statemachine.stateMachine import StateMachine
from src.autonomous.utils.lane_tracker import (
    HybridLaneTracker,
    JunctionDetector,
    get_bev,
    pure_pursuit,
    SRC_PTS,
    DST_PTS,
    LANE_WIDTH_M,
)
from src.autonomous.utils.topological_nav import (
    TopologicalNavigator,
    SPEED_LANE_FOLLOW,
    SPEED_STOP,
    LOOK_AHEAD_PX,
    LANE_WIDTH_PX,
)
from src.autonomous.utils.yolo_handler import get_bfmc_id

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# LANE FOLLOWING CONSTANTS
# -------------------------------------------------------------------------
MAX_STEER_DEG    = 25.0    # hard clamp on steering output  (±degrees)
LOST_GRACE_FRAMES = 8      # hold last steering for this many frames before stopping

# -------------------------------------------------------------------------
# TELEMETRY CONSTANTS
# -------------------------------------------------------------------------
TELEMETRY_HISTORY_MAX = 20   # max entries in historyData ring buffer


class processAutonomous(WorkerProcess):
    """
    Main autonomous driving WorkerProcess.

    Constructor Parameters
    ----------------------
    queueList   : dict   – shared multiprocessing Queues
    logging     : Logger – standard Python logger
    ready_event : Event  – set when process is ready (optional)
    debugging   : bool   – enable verbose logging
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.logger    = logging
        self.debugging = debugging

        # ------------------------------------------------------------------
        # Object detection state (filled by YoloDetection messages)
        # ------------------------------------------------------------------
        self._latest_yolo       = []    # list of detection dicts this cycle
        self._history_data_ring = []    # ring buffer for SharedMemory historyData

        # ------------------------------------------------------------------
        # Camera preprocessing (perspective transform + CLAHE)
        # ------------------------------------------------------------------
        self._M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # ------------------------------------------------------------------
        # Core perception + navigation objects
        # ------------------------------------------------------------------
        self._tracker   = HybridLaneTracker(img_shape=(480, 640))
        self._jct       = JunctionDetector()
        self._topnav    = TopologicalNavigator()

        # ------------------------------------------------------------------
        # Lane-following state (smoothed steering, lost frame counter)
        # ------------------------------------------------------------------
        self._smooth_steer  = 0.0
        self._lost_frames   = 0
        self._last_target_x = 320.0    # fallback target (image centre)

        # ------------------------------------------------------------------
        # Speed / semaphore state
        # ------------------------------------------------------------------
        self._current_speed_mm_s = 0.0   # raw value from STM32 (mm/s)
        self._semaphores_dict    = {}    # {sem_id: state_int}
        self._location           = {}    # GPS/V2X location dict

        # ------------------------------------------------------------------
        # Timing
        # ------------------------------------------------------------------
        self._last_work_time = time.monotonic()

        self._init_subscribers(queueList)
        self._init_senders(queueList)

        super(processAutonomous, self).__init__(queueList, ready_event)

    # ======================================================================
    # SUBSCRIPTIONS / SENDERS
    # ======================================================================
    def _init_subscribers(self, queueList):
        # CurrentSpeed: mm/s from STM32 serial handler
        self.speedSubscriber = messageHandlerSubscriber(
            queueList, CurrentSpeed, "lastOnly", True
        )
        # Semaphores: dict of {int: int} from processSemaphores (UDP 5007)
        self.semaphoreSubscriber = messageHandlerSubscriber(
            queueList, Semaphores, "lastOnly", True
        )
        # Location: GPS dict from processTrafficCommunication (TCP 9000)
        self.locationSubscriber = messageHandlerSubscriber(
            queueList, Location, "lastOnly", True
        )
        # YoloDetection: dict from processVision
        self.yoloSubscriber = messageHandlerSubscriber(
            queueList, YoloDetection, "fifo", True
        )
        # mainCamera: raw BGR frames from processCamera
        self.cameraSubscriber = messageHandlerSubscriber(
            queueList, mainCamera, "lastOnly", True
        )

    def _init_senders(self, queueList):
        # SpeedMotor → threadWrite → STM32 speed command
        self.speedSender = messageHandlerSender(queueList, SpeedMotor)
        # SteerMotor → threadWrite → STM32 steer command
        self.steerSender = messageHandlerSender(queueList, SteerMotor)

    # ======================================================================
    # WorkerProcess INTERFACE
    # ======================================================================
    def _init_threads(self):
        """No sub-threads; all work runs in process_work() on a tight loop."""
        pass

    def process_work(self):
        """
        Main control loop — called repeatedly by WorkerProcess.run().

        Execution order:
          1. Read CurrentSpeed  → update odometry + telemetry deviceSpeed
          2. Read Semaphores    → update V2X semaphore dict
          3. Read Location      → update location (passive, used by telemetry)
          4. Drain YoloDetection queue → collect detections + update historyData
          5. Read camera frame  → run lane tracker + junction detector
          6. Compute lane-following steering command
          7. Ask TopologicalNavigator for override
          8. Send final SpeedMotor + SteerMotor commands
        """
        now = time.monotonic()
        dt  = now - self._last_work_time
        self._last_work_time = now

        # ------------------------------------------------------------------
        # 1. READ CURRENT SPEED (mm/s from STM32)
        # ------------------------------------------------------------------
        speed_recv = self.speedSubscriber.receive()
        if speed_recv is not None:
            self._current_speed_mm_s = float(speed_recv)

        # Update dead-reckoning odometry in navigator
        # METRIC: mm/s is passed; topological_nav converts internally to m/s
        self._topnav.update_distance(self._current_speed_mm_s, dt=dt)

        # ------------------------------------------------------------------
        # 2. WRITE deviceSpeed TO shared_memory  (mm/s → cm/s)
        #    LiveTraffic expects cm/s.
        #    METRIC CONVERSION: speed_cm_s = speed_mm_s / 10.0
        # ------------------------------------------------------------------
        speed_cm_s = self._current_speed_mm_s / 10.0   # mm/s ÷ 10 = cm/s
        try:
            StateMachine.shared_memory["deviceSpeed"] = speed_cm_s
        except Exception:
            pass  # shared_memory may not be initialised on very first tick

        # ------------------------------------------------------------------
        # 3. READ SEMAPHORES (V2X — UDP 5007)
        # ------------------------------------------------------------------
        sem_recv = self.semaphoreSubscriber.receive()
        if sem_recv is not None:
            self._semaphores_dict = dict(sem_recv)

        # ------------------------------------------------------------------
        # 4. READ LOCATION (TCP 9000 — passive store)
        # ------------------------------------------------------------------
        loc_recv = self.locationSubscriber.receive()
        if loc_recv is not None:
            self._location = dict(loc_recv)

        # ------------------------------------------------------------------
        # 5. DRAIN YoloDetection queue → collect detections this cycle
        # ------------------------------------------------------------------
        self._latest_yolo = []
        while True:
            det = self.yoloSubscriber.receive()
            if det is None:
                break
            self._latest_yolo.append(det)

        # Update historyData in shared_memory for LiveTraffic telemetry
        self._update_history_data()

        # ------------------------------------------------------------------
        # 6. READ CAMERA FRAME → lane tracking + junction detection
        # ------------------------------------------------------------------
        frame = self.cameraSubscriber.receive()
        is_junction = False
        lane_steer  = 0.0   # steering from lane follower (degrees)
        lane_speed  = SPEED_LANE_FOLLOW

        if frame is not None:
            try:
                # Bird's-eye view preprocessing
                bev = get_bev(frame, M=self._M, clahe=self._clahe)

                # Run lane tracker (sliding window / polynomial)
                sl, sr, mode_label = self._tracker.update(bev)

                # Junction detection
                jct_state = self._jct.update(
                    bev,
                    self._tracker.left_conf,
                    self._tracker.right_conf,
                    self._tracker.left_fit,
                    self._tracker.right_fit,
                    lane_width_px=LANE_WIDTH_PX,
                )
                is_junction = (jct_state == "JUNCTION")

                # Compute lane-following target x
                nav_state  = "JUNCTION" if is_junction else "NORMAL"
                target_x, anchor = self._tracker.get_target_x(
                    y_eval=480 - LOOK_AHEAD_PX,
                    lane_width_px=LANE_WIDTH_PX,
                    nav_state=nav_state,
                )

                if target_x is None:
                    # Lane lost — use grace-period fallback
                    self._lost_frames += 1
                    target_x = self._last_target_x
                else:
                    self._lost_frames = 0
                    self._last_target_x = target_x

                # Pure pursuit → steering angle (degrees)
                raw_steer = pure_pursuit(target_x, LOOK_AHEAD_PX, LANE_WIDTH_PX)

                # EMA smoothing (α=0.4 fast)
                self._smooth_steer = (0.4 * raw_steer
                                      + 0.6 * self._smooth_steer)
                lane_steer = float(np.clip(self._smooth_steer,
                                           -MAX_STEER_DEG, MAX_STEER_DEG))

                # Lost-lane speed penalty
                if self._lost_frames > LOST_GRACE_FRAMES:
                    lane_speed = SPEED_STOP
                    if self._lost_frames % 30 == 0:
                        log.warning("[Autonomous] Lane lost — stopping.")
                elif self._lost_frames > 0:
                    scale = max(0.3, 1.0 - self._lost_frames / LOST_GRACE_FRAMES)
                    lane_speed = int(SPEED_LANE_FOLLOW * scale)

                if self.debugging:
                    log.info(
                        "[Autonomous] anchor=%s steer=%.1f° lost=%d jct=%s nav=%s",
                        anchor, lane_steer, self._lost_frames, jct_state,
                        self._topnav.nav_state
                    )

            except Exception as e:
                log.error("[Autonomous] Vision error: %s", e)

        # ------------------------------------------------------------------
        # 7. TOPOLOGICAL NAVIGATOR OVERRIDE
        #    Checks V2X semaphores and executes intersection maneuvers.
        # ------------------------------------------------------------------
        yolo_labels = [d["label"] for d in self._latest_yolo]

        override_speed, override_steer, nav_state_name = self._topnav.process_logic(
            is_junction    = is_junction,
            yolo_labels    = yolo_labels,
            semaphores_dict = self._semaphores_dict,
        )

        # Decide final commands
        # Navigator returns SPEED_LANE_FOLLOW when not overriding
        if nav_state_name in ("LANE_FOLLOW",):
            # Use lane-following commands from camera
            final_speed = lane_speed
            final_steer = lane_steer
        else:
            # Navigator is actively controlling (maneuver, halt, stop)
            final_speed = override_speed
            final_steer = override_steer

        # ------------------------------------------------------------------
        # 8. SEND FINAL COMMANDS TO STM32 VIA QUEUE
        # ------------------------------------------------------------------
        self.speedSender.send(str(int(final_speed)))
        self.steerSender.send(str(int(round(final_steer))))

    # ======================================================================
    # TELEMETRY HELPERS
    # ======================================================================
    def _update_history_data(self):
        """
        Push YOLO detections into the LiveTraffic historyData.

        Format expected by LiveTraffic server:
            [obstacle_id, x_normalised, y_normalised]

        Only BFMC-mapped detections (bfmc_id != -1) are forwarded.
        """
        for det in self._latest_yolo:
            bfmc_id = det.get("bfmc_id", -1)
            if bfmc_id == -1:
                continue   # not a BFMC-tracked class

            entry = [bfmc_id, det["x"], det["y"]]
            self._history_data_ring.append(entry)

        # Trim ring buffer
        if len(self._history_data_ring) > TELEMETRY_HISTORY_MAX:
            self._history_data_ring = self._history_data_ring[-TELEMETRY_HISTORY_MAX:]

        # Write to shared_memory (consumed by processTrafficCommunication)
        try:
            StateMachine.shared_memory["historyData"] = list(self._history_data_ring)
        except Exception:
            pass
