"""
topological_nav.py
==================
BFMC Autonomous Module – Topological Dead-Reckoning Navigator

Strategy
--------
At intersections we abandon the camera and execute hardcoded turning
maneuvers using integrative dead-reckoning from the STM32 odometer.

Odometry Source
---------------
CurrentSpeed comes from STM32_SerialHandler in millimeters per second (mm/s).
We convert: distance_m += (speed_mm_s / 1000.0) * dt_seconds

Route Format
------------
Each waypoint is a tuple: (action, distance_m, semaphore_id)

  action       : "STRAIGHT" | "LEFT" | "RIGHT" | "STOP"
  distance_m   : metres to travel before this action is considered complete
  semaphore_id : int ID to check in semaphores_dict (or None to skip)

V2X Semaphore States (UDP port 5007)
-------------------------------------
  0 = RED    -> HALT; do not proceed through the intersection
  1 = YELLOW -> proceed with caution (treated as GREEN here)
  2 = GREEN  -> proceed

Steering Angles for Dead-Reckoning Maneuvers
---------------------------------------------
  STRAIGHT : 0 degrees
  LEFT     : -23 degrees  (negative = left turn in BFMC convention)
  RIGHT    : +23 degrees  (positive = right turn in BFMC convention)
  STOP     : 0 degrees, speed = 0 PWM
"""

import time
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MANEUVER PARAMETERS
# ---------------------------------------------------------------------------
STEER_STRAIGHT_DEG = 0.0
STEER_LEFT_DEG     = -23.0   # negative = steer left
STEER_RIGHT_DEG    = +23.0   # positive = steer right

# Speed PWM during lane following vs intersection maneuvers
SPEED_LANE_FOLLOW  = 50      # PWM units sent to SpeedMotor during normal driving
SPEED_MANEUVER     = 35      # slower during dead-reckoning intersection execution
SPEED_STOP         = 0       # full stop

# How many metres to travel straight AFTER the turn to re-acquire the lane
LANE_REACQUIRE_M   = 0.30

# Default lane look-ahead and width for the lane follower
LOOK_AHEAD_PX      = 150
LANE_WIDTH_PX      = 280


# ---------------------------------------------------------------------------
# NAVIGATOR CLASS
# ---------------------------------------------------------------------------
class TopologicalNavigator:
    """
    Manages a pre-programmed topological route and overrides lane-following
    commands at intersections using dead-reckoning.

    Usage
    -----
    nav = TopologicalNavigator()

    # In your control loop:
    nav.update_distance(current_speed_mm_s)                   # call every cycle
    speed, steer, state = nav.process_logic(
        is_junction=True/False,
        yolo_labels=["stop_sign"],
        semaphores_dict={2: 2}                                # {id: state}
    )
    """

    # -----------------------------------------------------------------------
    # ROUTE DEFINITION
    # Each entry: (action, target_distance_m, semaphore_id_or_None)
    # -----------------------------------------------------------------------
    ROUTE = [
        ("STRAIGHT", 0.70, None),   # leg 0: straight 0.70 m to first junction
        ("LEFT",     0.95,    2),   # leg 1: left turn at junction 2
                                    #        (check semaphore ID 2 before proceeding)
        ("RIGHT",    0.65, None),   # leg 2: right turn 0.65 m to next junction
        ("STOP",     0.0,  None),   # leg 3: end of route, full stop
    ]

    def __init__(self):
        self.route_idx           = 0          # current waypoint index
        self.distance_accumulated = 0.0       # metres driven since last waypoint
        self._last_time          = time.monotonic()

        # Internal state machine
        self._state              = "LANE_FOLLOW"  # LANE_FOLLOW | WAITING_GREEN | MANEUVER | REACQUIRE | DONE
        self._maneuver_dist      = 0.0            # distance counter for active maneuver
        self._halted_for_red     = False

        log.info("[TopologicalNavigator] Initialized. Route has %d waypoints.", len(self.ROUTE))

    # -----------------------------------------------------------------------
    # PUBLIC: ODOMETRY UPDATE (call every process_work cycle)
    # -----------------------------------------------------------------------
    def update_distance(self, current_speed_mm_s, dt=None):
        """
        Integrate speed into travelled distance.

        Parameters
        ----------
        current_speed_mm_s : float
            Speed received from STM32 in millimeters per second (mm/s).

        dt : float | None
            Time delta in seconds. If None, computed automatically.

        Conversion
        ----------
        speed_m_s = speed_mm_s / 1000.0       (mm/s  →  m/s)
        delta_m   = speed_m_s  * dt            (m/s   →  metres)
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self._last_time
        self._last_time = now

        # METRIC CONVERSION: mm/s -> m/s -> metres
        speed_m_s = current_speed_mm_s / 1000.0   # mm/s  ÷ 1000  =  m/s
        delta_m   = speed_m_s * dt                # m/s   ×  s     =  m

        self.distance_accumulated += delta_m
        self._maneuver_dist       += delta_m

    # -----------------------------------------------------------------------
    # PUBLIC: CONTROL LOGIC (call every process_work cycle)
    # -----------------------------------------------------------------------
    def process_logic(self, is_junction, yolo_labels, semaphores_dict):
        """
        Determine whether to override lane-following commands.

        Parameters
        ----------
        is_junction     : bool   – True while JunctionDetector reports JUNCTION
        yolo_labels     : list   – YOLO class labels detected this frame
        semaphores_dict : dict   – {semaphore_id (int): state (int)}
                                   state: 0=RED, 1=YELLOW, 2=GREEN

        Returns
        -------
        speed_pwm   : int    – PWM value to send to SpeedMotor
        steer_deg   : float  – Steering angle in degrees for SteerMotor
        state_name  : str    – Human-readable state for telemetry/debugging
        """
        # ---- Route exhausted -------------------------------------------
        if self._state == "DONE" or self.route_idx >= len(self.ROUTE):
            return SPEED_STOP, STEER_STRAIGHT_DEG, "DONE"

        action, target_dist_m, sem_id = self.ROUTE[self.route_idx]

        # ---- STOP waypoint -----------------------------------------------
        if action == "STOP":
            log.info("[TopologicalNavigator] STOP waypoint reached.")
            self._state = "DONE"
            return SPEED_STOP, STEER_STRAIGHT_DEG, "STOP"

        # ---- Normal lane-following until a junction is reached -----------
        if self._state == "LANE_FOLLOW":
            if is_junction:
                # Check semaphore BEFORE entering intersection
                if sem_id is not None:
                    sem_state = semaphores_dict.get(sem_id, 2)  # default GREEN if unknown
                    if sem_state == 0:   # RED
                        self._halted_for_red = True
                        log.warning("[TopologicalNavigator] RED semaphore ID=%d — HALTING.", sem_id)
                        return SPEED_STOP, STEER_STRAIGHT_DEG, "WAITING_RED"
                    else:
                        self._halted_for_red = False

                # Green / no semaphore → begin maneuver
                log.info("[TopologicalNavigator] Entering MANEUVER: %s (leg %d)", action, self.route_idx)
                self._state = "MANEUVER"
                self._maneuver_dist = 0.0  # reset maneuver distance counter

        # ---- Waiting for green (re-evaluate every cycle) ------------------
        if self._state == "WAITING_RED":
            if sem_id is not None:
                sem_state = semaphores_dict.get(sem_id, 2)
                if sem_state != 0:   # YELLOW or GREEN — go
                    log.info("[TopologicalNavigator] Semaphore ID=%d now GREEN — proceeding.", sem_id)
                    self._state = "MANEUVER"
                    self._maneuver_dist = 0.0
                    # Fall through to MANEUVER block below
                else:
                    return SPEED_STOP, STEER_STRAIGHT_DEG, "WAITING_RED"

        # ---- Execute intersection maneuver (dead-reckoning) ---------------
        if self._state == "MANEUVER":
            steer = self._action_to_steer(action)

            if self._maneuver_dist >= target_dist_m:
                # Maneuver distance reached → advance to next waypoint
                log.info("[TopologicalNavigator] Maneuver '%s' complete (%.3f m). Advancing.",
                         action, self._maneuver_dist)
                self.route_idx           += 1
                self.distance_accumulated = 0.0
                self._maneuver_dist       = 0.0
                self._state               = "REACQUIRE"
                return SPEED_MANEUVER, STEER_STRAIGHT_DEG, "REACQUIRE"

            return SPEED_MANEUVER, steer, f"MANEUVER_{action}"

        # ---- Brief straight segment to let the tracker reacquire the lane -
        if self._state == "REACQUIRE":
            if self._maneuver_dist >= LANE_REACQUIRE_M or not is_junction:
                self._state = "LANE_FOLLOW"
                log.info("[TopologicalNavigator] Lane reacquired. Switching to LANE_FOLLOW.")
            return SPEED_MANEUVER, STEER_STRAIGHT_DEG, "REACQUIRE"

        # ---- Default: normal lane-following --------------------------------
        return SPEED_LANE_FOLLOW, STEER_STRAIGHT_DEG, "LANE_FOLLOW"

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------
    def _action_to_steer(self, action):
        """Map route action string to steering angle in degrees."""
        mapping = {
            "STRAIGHT": STEER_STRAIGHT_DEG,
            "LEFT":     STEER_LEFT_DEG,
            "RIGHT":    STEER_RIGHT_DEG,
        }
        return mapping.get(action, STEER_STRAIGHT_DEG)

    @property
    def current_waypoint(self):
        """Return the current (action, target_dist_m, sem_id) tuple."""
        if self.route_idx < len(self.ROUTE):
            return self.ROUTE[self.route_idx]
        return ("STOP", 0.0, None)

    @property
    def nav_state(self):
        """Return the current internal state label."""
        return self._state
