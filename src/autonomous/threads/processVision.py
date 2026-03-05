"""
processVision.py
================
BFMC Autonomous Module – Vision WorkerProcess

Runs YOLOv8 object detection on camera frames in a dedicated multiprocessing
process. Sends YoloDetection messages to the General queue so that
processAutonomous can consume them.

Message Flow
------------
  processCamera   --[mainCamera]--> processVision
  processVision   --[YoloDetection]--> processAutonomous
"""

import logging
import time
import numpy as np

from src.templates.workerprocess import WorkerProcess
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.allMessages import mainCamera, YoloDetection
from src.autonomous.utils.yolo_handler import filter_bfmc_detections

log = logging.getLogger(__name__)

# Minimum confidence threshold to forward a detection
CONF_THRESHOLD = 0.45

# Path to the YOLOv8 model weights
YOLO_MODEL_PATH = "yolov8n.pt"


class processVision(WorkerProcess):
    """
    WorkerProcess that loads a YOLOv8 model and runs inference on every
    camera frame it receives.

    Subscriptions
    -------------
    mainCamera (Owner=threadCamera) – raw BGR camera frames as numpy arrays.

    Publications
    ------------
    YoloDetection (Owner=processVision) – dict with keys:
        label   : str    – YOLO class name
        bfmc_id : int    – BFMC obstacle ID (-1 if unknown)
        conf    : float  – detection confidence
        x       : float  – bbox centre x (normalised 0..1)
        y       : float  – bbox centre y (normalised 0..1)
    """

    def __init__(self, queueList, logging, ready_event=None, debugging=False):
        self.logger    = logging
        self.debugging = debugging
        self.model     = None  # loaded in run() to avoid forking issues

        self._init_subscribers(queueList)
        self._init_senders(queueList)

        super(processVision, self).__init__(queueList, ready_event)

    # ------------------------------------------------------------------
    # SUBSCRIBERS / SENDERS
    # ------------------------------------------------------------------
    def _init_subscribers(self, queueList):
        self.cameraSubscriber = messageHandlerSubscriber(
            queueList, mainCamera, "lastOnly", True
        )

    def _init_senders(self, queueList):
        self.yoloSender = messageHandlerSender(queueList, YoloDetection)

    # ------------------------------------------------------------------
    # WorkerProcess INTERFACE
    # ------------------------------------------------------------------
    def _init_threads(self):
        """No sub-threads; all work happens in process_work()."""
        pass

    def run(self):
        """Override run() to load the YOLO model AFTER fork, then call super()."""
        self._load_model()
        super(processVision, self).run()

    def process_work(self):
        """Poll for a camera frame and run YOLO inference."""
        frame = self.cameraSubscriber.receive()
        if frame is None:
            return

        if self.model is None:
            return  # model failed to load

        try:
            results     = self.model(frame, verbose=False)
            detections  = filter_bfmc_detections(results)

            for det in detections:
                if det["conf"] >= CONF_THRESHOLD:
                    # Forward only BFMC-relevant detections (bfmc_id != -1)
                    # Also forward bfmc_id == -1 so processAutonomous can log unknowns
                    self.yoloSender.send(det)

                    if self.debugging:
                        log.info(
                            "[Vision] Detected: %s (BFMC_ID=%d, conf=%.2f)",
                            det["label"], det["bfmc_id"], det["conf"]
                        )
        except Exception as e:
            log.error("[Vision] Inference error: %s", e)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load YOLOv8 model. Logs a warning if ultralytics is not installed."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL_PATH)
            log.info("[Vision] YOLOv8 model loaded: %s", YOLO_MODEL_PATH)
        except ImportError:
            log.warning(
                "[Vision] ultralytics not installed — YOLO disabled. "
                "Run: pip install ultralytics"
            )
            self.model = None
        except Exception as e:
            log.error("[Vision] Failed to load model '%s': %s", YOLO_MODEL_PATH, e)
            self.model = None
