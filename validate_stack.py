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
    errors.append("FAIL utils import: " + str(e))

# Test 2: odometry conversion  500 mm/s * 0.1 s = 0.05 m
try:
    nav = TopologicalNavigator()
    nav.update_distance(500.0, dt=0.1)
    assert abs(nav.distance_accumulated - 0.05) < 1e-9, "Got " + str(nav.distance_accumulated)
    print("PASS: odometry 500 mm/s * 0.1 s = " + str(round(nav.distance_accumulated, 4)) + " m")
except Exception as e:
    errors.append("FAIL odometry: " + str(e))

# Test 3: YOLO mapping
try:
    assert get_bfmc_id("stop sign") == 1
    assert get_bfmc_id("traffic light") == 14
    assert get_bfmc_id("pedestrian crosswalk") == 11
    assert get_bfmc_id("dog") == -1
    print("PASS: YOLO ID mapping")
except Exception as e:
    errors.append("FAIL yolo mapping: " + str(e))

# Test 4: RED semaphore halts — must test on leg 1 (LEFT, sem_id=2)
# NOTE: Leg 0 is STRAIGHT with sem_id=None — no semaphore check (correct by design).
# The semaphore is only checked at leg 1 (LEFT turn at junction 2).
try:
    nav2 = TopologicalNavigator()
    nav2.route_idx = 1  # jump to LEFT waypoint which has sem_id=2
    speed, steer, state = nav2.process_logic(
        is_junction=True, yolo_labels=[], semaphores_dict={2: 0}
    )
    assert speed == 0, "Expected speed=0, got " + str(speed)
    assert state == "WAITING_RED", "Expected WAITING_RED, got " + str(state)
    print("PASS: RED semaphore on leg 1 -> HALT (speed=" + str(speed) + ", state=" + state + ")")
except Exception as e:
    errors.append("FAIL semaphore RED: " + str(e))

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
    errors.append("FAIL allMessages: " + str(e))

# Test 6: process imports (no hardware, just class loading)
try:
    from src.autonomous.threads.processVision import processVision
    from src.autonomous.threads.processAutonomous import processAutonomous
    print("PASS: Worker process classes imported")
except Exception as e:
    errors.append("FAIL process import: " + str(e))

print()
if errors:
    print("=== FAILURES ===")
    for err in errors:
        print("  " + err)
    sys.exit(1)
else:
    print("=== ALL TESTS PASSED ===")
