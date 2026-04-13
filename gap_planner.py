"""
gap_planner.py  —  Follow-the-Gap local planner for BARN Challenge 2026
========================================================================
Replaces DWA as the local velocity controller.

Algorithm (based on the winning approach from several BARN teams):
  1. Convert the LiDAR scan into a polar obstacle map.
  2. Threshold beams: free if range > threshold, blocked otherwise.
  3. Find all contiguous free gaps.
  4. Score each gap by how well its centre angle points toward the
     current local waypoint (A* output).
  5. Steer toward the best gap centre; set speed proportional to the
     nearest obstacle distance.

This works in dense BARN environments because it reasons directly about
free space in the scan rather than sampling kinematic trajectories —
so obstacle density doesn't cause combinatorial explosion.

Public API (mirrors the DWA interface used in navigator.py / simulate.py):

    planner = FollowGapPlanner(cfg)
    v, omega, debug = planner.compute_velocity(state, local_goal, lidar_ranges)

The returned `debug` dict contains gap information for the animator.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GapConfig:
    # LiDAR thresholding
    gap_threshold:      float = 0.8   # [m] range below this = obstacle beam
    min_gap_width:      int   = 5     # minimum consecutive free beams for a valid gap
    safety_bubble:      float = 0.35  # [m] radius of safety bubble around closest obstacle

    # Speed control
    max_linear_speed:   float = 0.8   # [m/s]
    min_linear_speed:   float = 0.15  # [m/s]  never completely stop in normal mode
    max_angular_speed:  float = 1.2   # [rad/s]
    
    # Speed scaling: slow down when obstacles are close
    speed_lookahead:    float = 1.5   # [m]  distance at which to start slowing
    speed_floor:        float = 0.2   # [m/s] minimum speed in cluttered space

    # Goal-steering blend
    # 0.0 = always steer toward gap centre
    # 1.0 = always steer toward goal (ignores gaps, degenerates to pure pursuit)
    goal_blend:         float = 0.7   # blend toward goal direction once in a gap

    # Angular controller
    k_angular:          float = 2.0   # proportional gain for heading error → omega

    # Heading error threshold beyond which we rotate in place before moving
    rotate_threshold:   float = 0.7   # [rad] ~45 degrees


# ──────────────────────────────────────────────────────────────────────────────
#  Gap data structure
# ──────────────────────────────────────────────────────────────────────────────

class Gap:
    """One contiguous free sector in the LiDAR scan."""

    def __init__(self, start_idx: int, end_idx: int,
                 angles: np.ndarray, ranges: np.ndarray):
        self.start_idx = start_idx
        self.end_idx   = end_idx
        self.width     = end_idx - start_idx + 1   # number of free beams

        # Deepest point in the gap (longest range within the sector)
        mid_idx   = (start_idx + end_idx) // 2
        deep_idx  = start_idx + int(np.argmax(ranges[start_idx:end_idx + 1]))

        self.center_angle = float(angles[mid_idx])
        self.deep_angle   = float(angles[deep_idx])
        self.deep_range   = float(ranges[deep_idx])

    @property
    def center_idx(self) -> int:
        return (self.start_idx + self.end_idx) // 2


# ──────────────────────────────────────────────────────────────────────────────
#  Main planner
# ──────────────────────────────────────────────────────────────────────────────

class FollowGapPlanner:

    def __init__(self, cfg: GapConfig = None):
        self.cfg = cfg or GapConfig()

    # ── public interface ───────────────────────────────────────────────────────

    def compute_velocity(
        self,
        state,                        # RobotState (has .x .y .yaw .v .omega)
        local_goal: Tuple[float, float],
        lidar_ranges: np.ndarray,     # raw LiDAR range array (robot frame)
        dt: float = 0.1,
    ) -> Tuple[float, float, dict]:
        """
        Returns (linear_velocity, angular_velocity, debug_dict).
        debug_dict contains gap info for the animator.
        """
        cfg    = self.cfg
        # --- GOAL PROXIMITY OVERRIDE ---
        # If very close to the local goal, ignore gaps and drive straight there
        dist_to_goal = math.hypot(local_goal[0] - state.x, local_goal[1] - state.y)
        if dist_to_goal < 1.2:
            goal_angle_world = math.atan2(local_goal[1] - state.y, local_goal[0] - state.x)
            heading_err = _normalize_angle(goal_angle_world - state.yaw)
            omega = float(np.clip(cfg.k_angular * heading_err, 
                                -cfg.max_angular_speed, cfg.max_angular_speed))
            speed = cfg.max_linear_speed * max(0.3, 1.0 - abs(heading_err) / math.pi)
            return float(np.clip(speed, 0.0, cfg.max_linear_speed)), omega, \
                {"gaps": [], "chosen_angle": heading_err, "mode": "goal_proximity"}
        # --- END OVERRIDE ---
        ranges = lidar_ranges.copy().astype(float)
        n      = len(ranges)

        # Replace inf/nan with the max threshold so they look like free space
        ranges = np.where(np.isfinite(ranges), ranges, cfg.gap_threshold * 2)
        ranges = np.clip(ranges, 0.01, cfg.gap_threshold * 2)

        # Beam angles in the robot frame (centred at 0 = forward)
        angles = np.linspace(-math.pi, math.pi, n)

        # 1. Safety bubble: zero out beams near the closest obstacle
        ranges = self._apply_safety_bubble(ranges, angles)

        # 2. Find all free gaps
        gaps = self._find_gaps(ranges, angles)

        # 3. Direction to local waypoint (in robot frame)
        goal_angle_world = math.atan2(
            local_goal[1] - state.y,
            local_goal[0] - state.x,
        )
        goal_angle_robot = _normalize_angle(goal_angle_world - state.yaw)

        # 4. Select the best gap
        if not gaps:
            # No gap found — rotate toward goal
            omega = cfg.k_angular * goal_angle_robot
            omega = float(np.clip(omega, -cfg.max_angular_speed, cfg.max_angular_speed))
            return 0.0, omega, {"gaps": [], "chosen_angle": goal_angle_robot, "mode": "no_gap"}

        best_gap = self._select_gap(gaps, goal_angle_robot)

        # 5. Compute steer angle: blend gap centre with goal direction
        gap_angle  = best_gap.deep_angle   # steer toward deepest part of gap
        steer_angle = _normalize_angle(
            (1.0 - cfg.goal_blend) * gap_angle
            + cfg.goal_blend * goal_angle_robot
        )

        # 6. Angular velocity
        omega = cfg.k_angular * steer_angle
        omega = float(np.clip(omega, -cfg.max_angular_speed, cfg.max_angular_speed))

        # 7. Linear velocity — scale with proximity to nearest obstacle
        min_range     = float(np.min(ranges))
        speed         = self._compute_speed(min_range, abs(steer_angle))

        # If heading error is large, rotate in place first
        if abs(steer_angle) > cfg.rotate_threshold:
            speed = 0.0

        debug = {
            "gaps":         gaps,
            "chosen_gap":   best_gap,
            "chosen_angle": steer_angle,
            "min_range":    min_range,
            "mode":         "normal",
        }

        return speed, omega, debug

    # ── internal helpers ───────────────────────────────────────────────────────

    def _apply_safety_bubble(
        self, ranges: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """
        Zero out (set to 0) beams within safety_bubble radius of the
        closest obstacle point.  This prevents the robot from trying to
        squeeze through gaps that are too narrow for its body.
        """
        cfg = self.cfg
        min_idx = int(np.argmin(ranges))
        min_r   = ranges[min_idx]

        if min_r >= cfg.gap_threshold:
            return ranges   # nothing close enough to worry about

        # angular spread corresponding to safety_bubble at distance min_r
        if min_r > 0:
            half_angle = math.asin(min(1.0, cfg.safety_bubble / min_r))
        else:
            half_angle = math.pi

        n         = len(ranges)
        angle_res = 2.0 * math.pi / n
        spread    = int(half_angle / angle_res) + 1

        result = ranges.copy()
        for i in range(-spread, spread + 1):
            idx = (min_idx + i) % n
            result[idx] = 0.0

        return result

    def _find_gaps(self, ranges: np.ndarray, angles: np.ndarray) -> List[Gap]:
        """Find all contiguous free sectors above the gap threshold."""
        cfg  = self.cfg
        free = ranges > cfg.gap_threshold
        gaps = []

        i = 0
        n = len(free)
        while i < n:
            if free[i]:
                j = i
                while j < n and free[j]:
                    j += 1
                width = j - i
                if width >= cfg.min_gap_width:
                    gaps.append(Gap(i, j - 1, angles, ranges))
                i = j
            else:
                i += 1

        return gaps

    def _select_gap(self, gaps: List[Gap], goal_angle: float) -> Gap:
        """
        Score each gap and return the best one.

        Score = gap_width_weight * normalised_width
              + goal_align_weight * (1 - |angle_diff| / pi)
              + depth_weight * normalised_depth

        Larger is better.
        """
        max_width = max(g.width for g in gaps)
        max_depth = max(g.deep_range for g in gaps)

        best_score = -float("inf")
        best_gap   = gaps[0]

        for gap in gaps:
            width_score = gap.width / max(max_width, 1)
            align_score = 1.0 - abs(_normalize_angle(gap.deep_angle - goal_angle)) / math.pi
            depth_score = gap.deep_range / max(max_depth, 0.01)

            score = 0.15 * width_score + 0.70 * align_score + 0.15 * depth_score

            if score > best_score:
                best_score = score
                best_gap   = gap

        return best_gap

    def _compute_speed(self, min_range: float, heading_error: float) -> float:
        """Scale speed with distance to nearest obstacle and heading error."""
        cfg = self.cfg

        # Distance-based scaling
        if min_range >= cfg.speed_lookahead:
            dist_factor = 1.0
        else:
            dist_factor = max(
                cfg.speed_floor / cfg.max_linear_speed,
                min_range / cfg.speed_lookahead,
            )

        # Heading-based scaling (slow down when turning hard)
        heading_factor = max(0.3, 1.0 - abs(heading_error) / math.pi)

        speed = cfg.max_linear_speed * dist_factor * heading_factor
        return float(np.clip(speed, cfg.min_linear_speed, cfg.max_linear_speed))


# ──────────────────────────────────────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a
