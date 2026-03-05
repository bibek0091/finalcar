"""
yolo_handler.py
===============
BFMC Autonomous Module – YOLO-to-BFMC ID Mapping

Maps YOLOv8 class label strings to BFMC official obstacle/sign IDs for
inserting into the LiveTraffic shared_memory historyData telemetry.

BFMC ID Reference
-----------------
  1  = Stop Sign
  4  = Crosswalk
  11 = Pedestrian Crosswalk
  12 = Pedestrian Road
  14 = Traffic Light

Usage
-----
from src.autonomous.utils.yolo_handler import get_bfmc_id, BFMC_ID_MAP

bfmc_id = get_bfmc_id("stop sign")   # returns 1
bfmc_id = get_bfmc_id("unknown")     # returns -1 (not a BFMC-tracked class)
"""

# ---------------------------------------------------------------------------
# BFMC OFFICIAL ID MAPPING
# Keys are normalised (lowercase, stripped) YOLO label strings.
# Values are the integer IDs expected by the LiveTraffic server.
# ---------------------------------------------------------------------------
BFMC_ID_MAP = {
    # Stop sign
    "stop sign":            1,
    "stop_sign":            1,
    "stop":                 1,

    # Crosswalk (painted road marking)
    "crosswalk":            4,

    # Pedestrian crosswalk (sign indicating a pedestrian crossing ahead)
    "pedestrian crosswalk": 11,
    "pedestrian_crosswalk": 11,

    # Pedestrian road (sign indicating pedestrians may be on the road)
    "pedestrian road":      12,
    "pedestrian_road":      12,
    "pedestrian":           12,

    # Traffic light (the physical signal head)
    "traffic light":        14,
    "traffic_light":        14,
    "trafficlight":         14,
}


def get_bfmc_id(yolo_label: str) -> int:
    """
    Convert a raw YOLO class label string to its BFMC obstacle ID.

    Parameters
    ----------
    yolo_label : str
        Class label returned by ultralytics YOLO (e.g. "stop sign", "person").

    Returns
    -------
    int
        BFMC obstacle ID if recognised, -1 if the class is not in the map.
    """
    key = str(yolo_label).strip().lower()
    return BFMC_ID_MAP.get(key, -1)


def filter_bfmc_detections(results):
    """
    Extract BFMC-relevant detections from a list of ultralytics Results objects.

    Parameters
    ----------
    results : list[ultralytics.engine.results.Results]
        Raw YOLO inference results (from model(frame)).

    Returns
    -------
    list[dict]
        Each dict has:
          "label"   : str   – original YOLO class name
          "bfmc_id" : int   – BFMC obstacle ID  (-1 if not in map)
          "conf"    : float – detection confidence [0.0, 1.0]
          "x"       : float – bounding box centre x (normalised 0–1)
          "y"       : float – bounding box centre y (normalised 0–1)
    """
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            label    = result.names[int(box.cls)]
            bfmc_id  = get_bfmc_id(label)
            conf     = float(box.conf)
            # xywhn: (x_centre, y_centre, width, height) normalised
            xywhn    = box.xywhn[0].tolist()
            detections.append({
                "label":   label,
                "bfmc_id": bfmc_id,
                "conf":    conf,
                "x":       xywhn[0],
                "y":       xywhn[1],
            })
    return detections
