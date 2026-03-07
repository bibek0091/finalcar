"""Quick validation script for the BFMC autonomous stack."""
import sys
import os

# Set path to the new stack
TARGET = "C:/Users/p23mi/Documents/final_bfmc_stack"
sys.path.insert(0, TARGET)
os.chdir(TARGET)

errors = []

# Test 1: import utils
try:
    from src.autonomous.utils.lane_tracker import HybridLaneTracker, JunctionDetector, get_bev, pure_pursuit
    from src.autonomous.utils.topological_nav import TopologicalNavigator
    from src.autonomous.utils.yolo_handler import get_bfmc_id, BFMC_ID_MAP
    print("PASS: All autonomous utils imported")
except Exception as e:
    errors.append(f"FAIL utils import: {e}")

# Test 2: odometry conversion  500 mm/s * 0.1 s = 0.05 m
try:
    nav = TopologicalNavigator()
    nav.update_distance(500.0, dt=0.1)
    assert abs(nav.distance_accumulated - 0.05) < 1e-9, f"Got {nav.distance_accumulated}"
    print(f"PASS: odometry 500 mm/s * 0.1 s = {nav.distance_accumulated:.4f} m")
except Exception as e:
    errors.append(f"FAIL odometry: {e}")

# Test 3: YOLO mapping
try:
    assert get_bfmc_id("stop sign") == 1
    assert get_bfmc_id("traffic light") == 14
    assert get_bfmc_id("pedestrian crosswalk") == 11
    assert get_bfmc_id("dog") == -1
    print("PASS: YOLO ID mapping")
except Exception as e:
    errors.append(f"FAIL yolo mapping: {e}")

# Test 4: RED semaphore halts
try:
    nav2 = TopologicalNavigator()
    speed, steer, state = nav2.process_logic(
        is_junction=True, yolo_labels=[], semaphores_dict={2: 0}
    )
    assert speed == 0, f"speed={speed}"
    assert state == "WAITING_RED", f"state={state}"
    print(f"PASS: RED semaphore → HALT (speed={speed}, state={state})")
except Exception as e:
    errors.append(f"FAIL semaphore RED: {e}")

# Test 5: message definitions
try:
    from src.utils.messages.allMessages import (
        YoloDetection, CurrentSpeed, SpeedMotor, SteerMotor, Semaphores, Location
    )
    assert YoloDetection.Owner.value == "processVision"
    assert YoloDetection.msgType.value == "dict"
    assert CurrentSpeed.Owner.value == "threadRead"
    print("PASS: allMessages YoloDetection + existing messages OK")
except Exception as e:
    errors.append(f"FAIL allMessages: {e}")

# Test 6: process imports (no hardware, just class loading)
try:
    from src.autonomous.threads.processVision import processVision
    from src.autonomous.threads.processAutonomous import processAutonomous
    print("PASS: Worker process classes imported")
except Exception as e:
    errors.append(f"FAIL process import: {e}")

print()
if errors:
    print("=== FAILURES ===")
    for err in errors:
        print(" ", err)
    sys.exit(1)
else:
    print("=== ALL TESTS PASSED ===")
