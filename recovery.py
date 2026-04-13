import math
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass
class RecoveryConfig:
    stuck_vel_threshold: float = 0.05      # DWA commanded speed < this means we are stuck
    stuck_time_limit: float = 2.0          # Seconds before triggering recovery
    history_resolution: float = 0.1        # Drop a breadcrumb every 10cm
    history_length: int = 50               # Keep last 50 points (5 meters of history)
    backtrack_distance: float = 0.4        # How far back along the path to target
    reverse_speed: float = -0.3            # m/s to drive backward
    max_recovery_time: float = 4.0         # Abort recovery if taking too long

def _normalize_angle(angle: float) -> float:
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class RecoveryManager:
    def __init__(self, cfg: RecoveryConfig = None):
        self.cfg = cfg or RecoveryConfig()
        self.history = deque(maxlen=self.cfg.history_length)
        
        # State tracking
        self.in_recovery = False
        self.stuck_elapsed = 0.0
        self.recovery_elapsed = 0.0
        self.target_pt = None

    def reset(self):
        self.history.clear()
        self.in_recovery = False
        self.stuck_elapsed = 0.0
        self.recovery_elapsed = 0.0
        self.target_pt = None

    def step(
        self,
        dwa_v: float,
        dwa_omega: float,
        lidar: np.ndarray,
        state,
        dt: float = 0.1,
    ) -> tuple:
        """
        Takes the DWA's intended velocity and the robot's current state.
        Returns (final_v, final_omega, is_recovering).
        """
        # 1. Log Breadcrumbs (Only if moving forward and not in recovery)
        if not self.in_recovery and dwa_v > self.cfg.stuck_vel_threshold:
            if not self.history:
                self.history.append((state.x, state.y))
            else:
                last_x, last_y = self.history[-1]
                dist = math.hypot(state.x - last_x, state.y - last_y)
                if dist >= self.cfg.history_resolution:
                    self.history.append((state.x, state.y))

        # 2. Check if Stuck
        is_stuck = abs(dwa_v) <= self.cfg.stuck_vel_threshold and abs(dwa_omega) <= 0.1
        
        if not self.in_recovery:
            if is_stuck:
                self.stuck_elapsed += dt
                if self.stuck_elapsed > self.cfg.stuck_time_limit:
                    self._trigger_recovery(state)
            else:
                self.stuck_elapsed = 0.0

        # 3. Execute Recovery Math
        if self.in_recovery:
            # Timeout check: If we've been backing up too long, abort and reset
            self.recovery_elapsed += dt
            if self.recovery_elapsed > self.cfg.max_recovery_time:
                self.in_recovery = False
                self.stuck_elapsed = 0.0
                self.recovery_elapsed = 0.0
                return 0.0, 0.0, False

            # If we have no history, just spin in place to try and clear the LiDAR
            if not self.target_pt:
                return 0.0, 0.5, True 

            # Reverse Pure Pursuit Math
            tx, ty = self.target_pt
            dist_to_target = math.hypot(state.x - tx, state.y - ty)
            
            # Goal reached?
            if dist_to_target < 0.15:
                self.in_recovery = False
                self.stuck_elapsed = 0.0
                self.recovery_elapsed = 0.0
                return 0.0, 0.0, False

            # Calculate steering to back up toward the target
            # The *rear* of the robot is at (yaw + pi)
            angle_to_target = math.atan2(ty - state.y, tx - state.x)
            rear_heading = _normalize_angle(state.yaw + math.pi)
            heading_error = _normalize_angle(angle_to_target - rear_heading)

            # Proportional controller for steering while reversing
            cmd_v = self.cfg.reverse_speed
            cmd_omega = 1.5 * heading_error  # Kp = 1.5
            
            # Limit angular velocity
            cmd_omega = max(min(cmd_omega, 1.0), -1.0)
            
            return cmd_v, cmd_omega, True

        # 4. Normal Operation
        return dwa_v, dwa_omega, False

    def _trigger_recovery(self, state):
        self.in_recovery = True
        self.recovery_elapsed = 0.0
        self.target_pt = None
        
        # Search backward through history for a point ~0.4m away
        for i in range(len(self.history)-1, -1, -1):
            px, py = self.history[i]
            if math.hypot(state.x - px, state.y - py) >= self.cfg.backtrack_distance:
                self.target_pt = (px, py)
                # Trim history so we don't try to backtrack here again
                while len(self.history) > i:
                    self.history.pop()
                break
