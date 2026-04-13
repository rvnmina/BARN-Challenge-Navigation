import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DWAConfig:
    # Velocity limits
    max_speed: float = 0.8          # Increased from 0.5
    min_speed: float = -0.1         
    max_yaw_rate: float = 1.2       # Slightly faster rotation
    max_accel: float = 1.0          # Snappier acceleration
    max_delta_yaw_rate: float = 1.5 

    # Trajectory sampling
    v_resolution: float = 0.02      
    yaw_rate_resolution: float = 0.05  
    dt: float = 0.1                 
    predict_time: float = 3.0       # AGGRESSIVE: Was 3.0 (Looks only 1.5s ahead)

    # Cost function weights
    to_goal_cost_gain: float = 0.35
    goal_dist_cost_gain: float = 1.25
    speed_cost_gain: float = 0.35
    obstacle_cost_gain: float = 1.4
    path_align_cost_gain: float = 0.45

    # Safety
    robot_radius: float = 0.25      # AGGRESSIVE: Was 0.3 (True physical edge of Jackal)
    obstacle_clearance: float = 0.08

    # Stuck detection
    stuck_vel_threshold: float = 0.05  
    stuck_time_limit: float = 3.0

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    omega: float = 0.0


class DWA:

    def __init__(self, config: DWAConfig):
        self.cfg = config


    def compute_velocity(
        self,
        state:          RobotState,
        goal:           Tuple[float, float],
        obstacles:      np.ndarray,             # (N, 2)  [x, y]
        obs_velocities: Optional[np.ndarray] = None,  # (N, 2) [vx, vy], optional
        ref_path:       Optional[np.ndarray] = None,
    ) -> Tuple[float, float, List[np.ndarray]]:

        dw = self._dynamic_window(state)
        best_u, best_traj = self._evaluate_trajectories(
            state, goal, obstacles, dw, obs_velocities, ref_path
        )
        return best_u[0], best_u[1], best_traj


    def _dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """Compute [v_min, v_max, omega_min, omega_max] reachable in dt."""
        cfg = self.cfg
        vs = (cfg.min_speed, cfg.max_speed, -cfg.max_yaw_rate, cfg.max_yaw_rate)
        vd = (
            state.v   - cfg.max_accel          * cfg.dt,
            state.v   + cfg.max_accel          * cfg.dt,
            state.omega - cfg.max_delta_yaw_rate * cfg.dt,
            state.omega + cfg.max_delta_yaw_rate * cfg.dt,
        )
        return (
            max(vs[0], vd[0]), min(vs[1], vd[1]),
            max(vs[2], vd[2]), min(vs[3], vd[3]),
        )


    def _evaluate_trajectories(
        self,
        state:          RobotState,
        goal:           Tuple[float, float],
        obstacles:      np.ndarray,
        dw:             Tuple[float, float, float, float],
        obs_velocities: Optional[np.ndarray],
        ref_path:       Optional[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:

        cfg = self.cfg
        min_cost = float("inf")
        best_u    = np.array([0.0, 0.0])
        best_traj = [np.array([state.x, state.y, state.yaw, state.v, state.omega])]

        for v in np.arange(dw[0], dw[1] + cfg.v_resolution * 0.5, cfg.v_resolution):
            for omega in np.arange(dw[2], dw[3] + cfg.yaw_rate_resolution * 0.5, cfg.yaw_rate_resolution):
                traj     = self._predict_trajectory(state, v, omega)
                ob_cost  = self._obstacle_cost(traj, obstacles, obs_velocities)
                if ob_cost == float("inf"):
                    continue
                heading_cost = cfg.to_goal_cost_gain * self._goal_heading_cost(traj, goal)
                dist_cost = cfg.goal_dist_cost_gain * self._goal_distance_cost(traj, goal)
                speed_cost = cfg.speed_cost_gain * self._speed_cost(traj[-1, 3])
                path_cost = cfg.path_align_cost_gain * self._path_align_cost(traj, ref_path)
                total_cost = heading_cost + dist_cost + speed_cost + path_cost + cfg.obstacle_cost_gain * ob_cost

                if total_cost < min_cost:
                    min_cost  = total_cost
                    best_u    = np.array([v, omega])
                    best_traj = traj

        # Fallback: if every trajectory was blocked, rotate toward goal
        if min_cost == float("inf"):
            import math
            dx         = goal[0] - state.x
            dy         = goal[1] - state.y
            angle_diff = math.atan2(dy, dx) - state.yaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            rot_dir    = 1.0 if angle_diff > 0 else -1.0
            best_u     = np.array([0.0, rot_dir * cfg.max_yaw_rate * 0.8])

        return best_u, best_traj

    def _predict_trajectory(
        self, init_state: RobotState, v: float, omega: float
    ) -> np.ndarray:
        """Simulate the robot forward for predict_time seconds."""
        cfg   = self.cfg
        state = np.array([init_state.x, init_state.y, init_state.yaw, v, omega])
        traj  = [state.copy()]
        t     = 0.0
        while t <= cfg.predict_time:
            state = self._motion(state, [v, omega], cfg.dt)
            traj.append(state.copy())
            t += cfg.dt
        return np.array(traj)

    @staticmethod
    def _motion(state: np.ndarray, u: List[float], dt: float) -> np.ndarray:
        """Unicycle kinematic model: [x, y, yaw, v, omega]."""
        s     = state.copy()
        s[2] += u[1] * dt
        s[0] += u[0] * np.cos(s[2]) * dt
        s[1] += u[0] * np.sin(s[2]) * dt
        s[3]  = u[0]
        s[4]  = u[1]
        return s


    def _goal_heading_cost(self, traj: np.ndarray, goal: Tuple[float, float]) -> float:
        dx           = goal[0] - traj[-1, 0]
        dy           = goal[1] - traj[-1, 1]
        angle_to_goal = np.arctan2(dy, dx)
        angle_err = angle_to_goal - traj[-1, 2]
        angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
        return abs(angle_err)

    @staticmethod
    def _goal_distance_cost(traj: np.ndarray, goal: Tuple[float, float]) -> float:
        return float(np.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1]))

    def _speed_cost(self, final_v: float) -> float:
        return float(max(0.0, self.cfg.max_speed - final_v))

    @staticmethod
    def _path_align_cost(
        traj: np.ndarray,
        ref_path: Optional[np.ndarray],
    ) -> float:
        if ref_path is None or len(ref_path) == 0:
            return 0.0
        end_xy = traj[-1, :2]
        dists = np.linalg.norm(ref_path - end_xy, axis=1)
        return float(np.min(dists))

    def _obstacle_cost(
        self,
        traj:           np.ndarray,
        obstacles:      np.ndarray,
        obs_velocities: Optional[np.ndarray] = None,
    ) -> float:

        cfg = self.cfg
        if obstacles.size == 0:
            return 0.0

        use_dynamic = (
            obs_velocities is not None
            and obs_velocities.shape == obstacles.shape
        )

        min_dist = float("inf")
        for step_idx, point in enumerate(traj):
            t = step_idx * cfg.dt  # elapsed time at this trajectory step

            if use_dynamic:
                # Project each obstacle to its position at time t.
                # Clamp projected displacement to avoid extrapolating noise
                # too far: anything beyond 3 m from its current position
                # is almost certainly a tracking error, so we cap it.
                displacement = obs_velocities * t
                displacement_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
                max_proj = 3.0
                scale = np.where(
                    displacement_norm > max_proj,
                    max_proj / np.maximum(displacement_norm, 1e-9),
                    1.0,
                )
                obs_at_t = obstacles + displacement * scale
            else:
                obs_at_t = obstacles

            dists = np.linalg.norm(obs_at_t - point[:2], axis=1)
            d     = np.min(dists) - cfg.robot_radius
            if d <= cfg.obstacle_clearance:
                return float("inf")
            min_dist = min(min_dist, d)

        return 1.0 / min_dist if min_dist != float("inf") else 0.0
