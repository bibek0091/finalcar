import os
import sys

model_path = "temp/tempfile/lane_detection/lane_detection/best.pt"
print(f"Loading model at: {os.path.abspath(model_path)}")

try:
    from ultralytics import YOLO
    print("Ultralytics imported successfully.")
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
    sys.exit(1)

import torch
_orig = torch.load
def _patched(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig(*args, **kwargs)
torch.load = _patched

try:
    model = YOLO(os.path.abspath(model_path))
    print("Model loaded successfully!")
except Exception as e:
    import traceback
    with open('exception.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("Failed to load model. Traceback saved to exception.txt")

torch.load = _orig
