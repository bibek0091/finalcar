import os
import sys
import time
import numpy as np
import cv2

# Add temp/tempfile to path to import modules
sys.path.insert(0, os.path.abspath("temp/tempfile"))

try:
    from lane_detection.traffic_module import ThreadedYOLODetector
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

print("--- YOLOv11 Model Verification ---")
model_path = "lane_detection/lane_detection/best.pt"
print(f"Target model path relative to traffic_module: {model_path}")

try:
    detector = ThreadedYOLODetector(model_path)
except Exception as e:
    print(f"Exception creating ThreadedYOLODetector: {e}")
    sys.exit(1)

# Wait a moment for thread to initialize and load model
print("Waiting for YOLO to load onto GPU/CPU in background thread...")
for i in range(10):
    time.sleep(1)
    if detector.yolo_ok:
        break

if not detector.yolo_ok:
    print("ERROR: YOLO failed to load.")
    detector.stop()
    sys.exit(1)

print(f"SUCCESS: YOLO loaded correctly.")
print(f"Resolved path: {detector.model_path_used}")
try:
    names = detector.model.names
    print(f"Model recognizes {len(names)} classes:")
    # Print first few classes
    print(", ".join(list(names.values())[:10]) + " ...")
except Exception as e:
    print("Could not fetch class names:", e)

# Test detection on a dummy frame (zeros)
print("Feeding a dummy frame and checking output...")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Fake a bounding box like a stop sign for the model to "see"? 
# Actually we can't easily fool YOLO with random pixels, but we can verify it doesn't crash on standard input.
detector.update_frame(dummy_frame)

time.sleep(2) # allow thread to run prediction

dets = detector.get_detections()
print(f"Inference complete. Detections found: {len(dets)}")

if len(dets) == 0:
    print("Normal: dummy frame resulted in 0 detections.")
else:
    print(f"Unexpected: found {len(dets)} detections on black frame?!")
    print(dets)

print("Stopping detector...")
detector.stop()
print("Verification complete.")
