#!/usr/bin/env python3

import math
import heapq
import numpy as np
from tqdm import tqdm
try:
    import rospy
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf.transformations import euler_from_quaternion
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARN] ROS not found -- running in simulation-only mode.")

from dwa import DWA, DWAConfig, RobotState
from gap_planner import FollowGapPlanner, GapConfig
from adaptive_tuner import AdaptiveTuner, AdaptiveConfig
from recovery import RecoveryManager, RecoveryConfig



class AStarPlanner:

    _MOVES = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    _COSTS = [1.0,   1.0,  1.0,   1.0,  1.414,   1.414,  1.414,  1.414]

    def __init__(
        self,
        grid_res:  float = 0.1,
        x_range:   tuple = (-8.0, 12.0),
        y_range:   tuple = (-10.0, 16.0),
        inflate_r: float = 0.31,
    ):
        self.res       = grid_res
        self.x_range   = x_range
        self.y_range   = y_range
        self.inflate_r = inflate_r
        self.nx = int((x_range[1] - x_range[0]) / grid_res)
        self.ny = int((y_range[1] - y_range[0]) / grid_res)

    def plan(self, start: RobotState, goal: tuple, obstacles: np.ndarray):
        """Returns list of (x, y) waypoints, or None if no path exists."""
        grid = self._build_grid(obstacles)
        sx, sy = self._clamp(*self._w2g(start.x, start.y))
        gx, gy = self._clamp(*self._w2g(goal[0],  goal[1]))
        if grid[gx, gy]:
            gx, gy = self._nearest_free(grid, gx, gy)
        path_cells = self._astar(grid, sx, sy, gx, gy)
        if path_cells is None:
            return None
        waypoints = [self._g2w(cx, cy) for cx, cy in path_cells]
        waypoints.append(goal)
        return self._decimate(waypoints, min_dist=0.3)

    def _w2g(self, x, y):
        return (
            int((x - self.x_range[0]) / self.res),
            int((y - self.y_range[0]) / self.res),
        )

    def _g2w(self, gx, gy):
        return (
            gx * self.res + self.x_range[0],
            gy * self.res + self.y_range[0],
        )

    def _clamp(self, gx, gy):
        return (
            max(0, min(gx, self.nx - 1)),
            max(0, min(gy, self.ny - 1)),
        )

    def _build_grid(self, obstacles: np.ndarray) -> np.ndarray:
        grid    = np.zeros((self.nx, self.ny), dtype=bool)
        if obstacles.size == 0:
            return grid
        r_cells = int(self.inflate_r / self.res) + 1
        for ox, oy in obstacles:
            cgx, cgy = self._w2g(ox, oy)
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    gx, gy = cgx + dx, cgy + dy
                    if not (0 <= gx < self.nx and 0 <= gy < self.ny):
                        continue
                    wx, wy = self._g2w(gx, gy)
                    if math.hypot(wx - ox, wy - oy) <= self.inflate_r:
                        grid[gx, gy] = True
        return grid

    def _nearest_free(self, grid, gx, gy):
        for r in range(1, 30):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx_, ny_ = gx + dx, gy + dy
                    if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny and not grid[nx_, ny_]:
                        return nx_, ny_
        return gx, gy

    def _astar(self, grid, sx, sy, gx, gy):
        open_set  = [(0.0, sx, sy)]
        came_from = {}
        g_score   = {(sx, sy): 0.0}
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if cx == gx and cy == gy:
                path = []
                node = (cx, cy)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path
            for (dx, dy), cost in zip(self._MOVES, self._COSTS):
                nx_, ny_ = cx + dx, cy + dy
                if not (0 <= nx_ < self.nx and 0 <= ny_ < self.ny):
                    continue
                if grid[nx_, ny_]:
                    continue
                ng = g_score.get((cx, cy), float("inf")) + cost
                if ng < g_score.get((nx_, ny_), float("inf")):
                    g_score[(nx_, ny_)] = ng
                    h = math.hypot(nx_ - gx, ny_ - gy)
                    heapq.heappush(open_set, (ng + h, nx_, ny_))
                    came_from[(nx_, ny_)] = (cx, cy)
        return None

    @staticmethod
    def _decimate(path, min_dist=0.3):
        if not path:
            return path
        out = [path[0]]
        for pt in path[1:]:
            if math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) >= min_dist:
                out.append(pt)
        if path[-1] != out[-1]:
            out.append(path[-1])
        return out



class DynamicObstacleTracker:
    """
    Estimates per-point obstacle velocities from consecutive LiDAR scans
    using nearest-neighbour data association.

    Each call to update() matches current obstacle points to the previous
    frame's points. Matched pairs produce a finite-difference velocity
    estimate; unmatched points get zero velocity (treated as static).

    Noise filter: velocities above max_speed_m_s are clamped to zero
    because they almost certainly come from false associations between
    two unrelated scan points rather than from a physically moving obstacle.
    In the BARN 2026 setting the only moving objects are iRobot Creates
    (~0.3 m/s max), so the 1.5 m/s default is already conservative.
    """

    def __init__(
        self,
        dt:             float = 0.1,
        max_match_dist: float = 0.6,
        max_speed_m_s:  float = 1.5,
        history:        int   = 3,
    ):
        self.dt             = dt
        self.max_match_dist = max_match_dist
        self.max_speed      = max_speed_m_s
        self.history        = history
        self._prev_obs:    np.ndarray = np.empty((0, 2))
        self._vel_history: list       = []

    def update(self, obstacles: np.ndarray) -> np.ndarray:
        """
        Call once per control cycle with the current obstacle point cloud.
        Returns velocity array of shape (N, 2) aligned with obstacles.
        """
        velocities = np.zeros_like(obstacles)

        if self._prev_obs.size > 0 and obstacles.size > 0:
            velocities = self._associate(obstacles)

        self._vel_history.append(velocities)
        if len(self._vel_history) > self.history:
            self._vel_history.pop(0)

        self._prev_obs = obstacles.copy()

        # Average over history frames that have the same obstacle count.
        # When the count changes (obstacle appears/disappears) use the
        # latest frame only to avoid mixing misaligned arrays.
        n = len(obstacles)
        matching = [v for v in self._vel_history if len(v) == n]
        return np.mean(matching, axis=0) if matching else velocities

    def reset(self):
        self._prev_obs    = np.empty((0, 2))
        self._vel_history = []

    def _associate(self, curr: np.ndarray) -> np.ndarray:
        velocities = np.zeros_like(curr)
        prev       = self._prev_obs
        for i, pt in enumerate(curr):
            dists = np.linalg.norm(prev - pt, axis=1)
            j     = int(np.argmin(dists))
            if dists[j] > self.max_match_dist:
                continue
            vel = (pt - prev[j]) / self.dt
            if np.linalg.norm(vel) > self.max_speed:
                continue
            velocities[i] = vel
        return velocities



class BARNNavigator:

    _REPLAN_INTERVAL = 30

    def __init__(self):
        self.dwa_cfg      = DWAConfig()
        self.adaptive_cfg = AdaptiveConfig()
        self.recovery_cfg = RecoveryConfig()

        self.dwa            = DWA(self.dwa_cfg)
        self.tuner          = AdaptiveTuner(self.dwa_cfg, self.adaptive_cfg)
        self.recovery       = RecoveryManager(self.recovery_cfg)
        self.global_planner = AStarPlanner()
        self.obs_tracker    = DynamicObstacleTracker(dt=0.1)

        self.state      = RobotState()
        self.goal       = None
        self.obstacles  = np.empty((0, 2))
        self.obs_vels   = np.empty((0, 2))
        self.lidar_raw  = np.array([])
        self.goal_tol   = 0.3

        self.waypoints:   list = []
        self.wp_idx:      int  = 0
        self._step_count: int  = 0

        self.start_time  = None
        self.nav_success = False

        if ROS_AVAILABLE:
            self._init_ros()


    def _init_ros(self):
        rospy.init_node("barn_navigator", anonymous=False)
        rate_hz   = rospy.get_param("~control_rate", 10)
        self.rate = rospy.Rate(rate_hz)
        self.obs_tracker.dt = 1.0 / rate_hz

        rospy.Subscriber("/scan",                  LaserScan,   self._lidar_cb)
        rospy.Subscriber("/odom",                  Odometry,    self._odom_cb)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb)

        self._cmd_pub   = rospy.Publisher("/cmd_vel",   Twist,  queue_size=1)
        self._state_pub = rospy.Publisher("/nav_state", String, queue_size=1)

        rospy.loginfo("[BARN] Navigator node started.")


    def _lidar_cb(self, msg: "LaserScan"):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[ranges == 0.0] = np.inf
        self.lidar_raw = ranges
        obs            = self._lidar_to_obstacles(msg)
        self.obstacles = obs
        self.obs_vels  = self.obs_tracker.update(obs)

    def _odom_cb(self, msg: "Odometry"):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw        = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.state.x     = p.x
        self.state.y     = p.y
        self.state.yaw   = yaw
        self.state.v     = msg.twist.twist.linear.x
        self.state.omega = msg.twist.twist.angular.z

    def _goal_cb(self, msg: "PoseStamped"):
        self.goal        = (msg.pose.position.x, msg.pose.position.y)
        self.start_time  = None
        self.waypoints   = []
        self.wp_idx      = 0
        self._step_count = 0
        self.recovery.reset()
        self.obs_tracker.reset()
        rospy.loginfo(f"[BARN] New goal: {self.goal}")


    def _lidar_to_obstacles(self, msg: "LaserScan") -> np.ndarray:
        angles = np.arange(
            msg.angle_min,
            msg.angle_max + msg.angle_increment,
            msg.angle_increment,
        )
        ranges = np.array(msg.ranges)
        valid  = np.isfinite(ranges) & (ranges < msg.range_max)
        xs = ranges[valid] * np.cos(angles[valid] + self.state.yaw) + self.state.x
        ys = ranges[valid] * np.sin(angles[valid] + self.state.yaw) + self.state.y
        return np.column_stack([xs, ys]) if xs.size else np.empty((0, 2))


    def spin(self):
        if not ROS_AVAILABLE:
            print("[BARN] Cannot spin: ROS unavailable.")
            return
        while not rospy.is_shutdown():
            self._control_step()
            self.rate.sleep()

    def _control_step(self):
        if self.goal is None:
            return

        dist = math.hypot(self.state.x - self.goal[0], self.state.y - self.goal[1])
        if dist < self.goal_tol:
            self._stop()
            if not self.nav_success:
                self.nav_success = True
                rospy.loginfo("[BARN] Goal reached!")
            return

        if self.lidar_raw.size > 0:
            context = self.tuner.update(self.lidar_raw)
        else:
            context = None

        if self.start_time is None:
            import time
            self.start_time = time.time()

        if self._step_count % self._REPLAN_INTERVAL == 0 or not self.waypoints:
            path = self.global_planner.plan(self.state, self.goal, self.obstacles)
            if path:
                self.waypoints = path
                self.wp_idx    = 0

        while self.wp_idx < len(self.waypoints) - 1:
            wpx, wpy = self.waypoints[self.wp_idx]
            if math.hypot(self.state.x - wpx, self.state.y - wpy) < 0.5:
                self.wp_idx += 1
            else:
                break

        local_goal = self.waypoints[self.wp_idx] if self.waypoints else self.goal

        v, omega, _ = self.gap_planner.compute_velocity(
            self.state, local_goal, self.lidar_raw,
            dt=self.dwa_cfg.dt,
        )
        cmd_v, cmd_omega, in_recovery = self.recovery.step(
            v, omega, self.lidar_raw, self.state, dt=self.dwa_cfg.dt
        )

        self._publish_cmd(cmd_v, cmd_omega)
        self._state_pub.publish(
            String(data=f"ctx={context} recovery={in_recovery} dist={dist:.2f}")
        )
        self._step_count += 1

    def _publish_cmd(self, v: float, omega: float):
        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = omega
        self._cmd_pub.publish(twist)

    def _stop(self):
        self._publish_cmd(0.0, 0.0)


class BaselineNavigator:
    """
    Upgraded baseline: Uses A* global planning to find a path, but uses a
    naive proportional heading controller (pure pursuit) for local execution.
    No dynamic obstacle avoidance.
    """

    _REPLAN_INTERVAL = 50

    def __init__(self, max_speed: float = 0.4, k_angular: float = 2.0):
        self.max_speed  = max_speed
        self.k_angular  = k_angular
        self._dt        = 0.1
        self.global_planner = AStarPlanner()

    def navigate(
        self,
        start:     RobotState,
        goal:      tuple,
        obstacles: np.ndarray,
        max_steps: int = 800,
    ) -> dict:
        state             = start
        steps             = 0
        collisions        = 0
        history           = []
        in_collision_last = False
        waypoints         = []
        wp_idx            = 0

        for step in range(max_steps):
            dist = math.hypot(state.x - goal[0], state.y - goal[1])
            if dist < 0.1:
                return {"success": True, "steps": steps,
                        "collisions": collisions, "final_dist": dist,
                        "history": history}

            # 1. Replan the A* path
            if step % self._REPLAN_INTERVAL == 0 or not waypoints:
                path = self.global_planner.plan(state, goal, obstacles)
                if path:
                    waypoints = path
                    wp_idx    = 0

            # 2. Track the active waypoint (Using your original, strict logic)
            while wp_idx < len(waypoints) - 1:
                wpx, wpy = waypoints[wp_idx]
                if math.hypot(state.x - wpx, state.y - wpy) < 0.5:
                    wp_idx += 1
                else:
                    break

            local_goal = waypoints[wp_idx] if waypoints else goal

            # 3. Steer directly toward local goal -- no gap logic, no collision avoidance
            goal_angle  = math.atan2(local_goal[1] - state.y, local_goal[0] - state.x)
            heading_err = goal_angle - state.yaw
            while heading_err >  math.pi: heading_err -= 2 * math.pi
            while heading_err < -math.pi: heading_err += 2 * math.pi

            omega = float(np.clip(self.k_angular * heading_err, -1.2, 1.2))
            v     = self.max_speed * max(0.0, 1.0 - abs(heading_err) / 1.2)

            state.yaw  += omega * self._dt
            state.x    += v * math.cos(state.yaw) * self._dt
            state.y    += v * math.sin(state.yaw) * self._dt
            state.v     = v
            state.omega = omega
            steps      += 1
            history.append((state.x, state.y, state.yaw))

            if obstacles.size:
                min_d        = np.min(np.linalg.norm(
                    obstacles - [state.x, state.y], axis=1))
                currently_in = min_d < 0.30
                if currently_in and not in_collision_last:
                    collisions += 1
                    if collisions >= 1:
                        return {"success": False, "steps": steps,
                                "collisions": collisions, "final_dist": dist,
                                "history": history}
                in_collision_last = currently_in
            else:
                in_collision_last = False

        final_dist = math.hypot(state.x - goal[0], state.y - goal[1])
        return {"success": False, "steps": steps,
                "collisions": collisions, "final_dist": final_dist,
                "history": history}

class SimNavigator:
    """
    Pure-Python simulation for unit-testing the navigation stack
    without a ROS installation.

    In the static sim, all obstacle velocities are zero, so the dynamic
    path is equivalent to the static one. The tracker is still exercised
    end-to-end so integration bugs surface before a ROS deployment.
    """

    _REPLAN_INTERVAL = 50

    def __init__(self, dwa_cfg: DWAConfig = None):
        self.dwa_cfg        = dwa_cfg or DWAConfig()
        self.gap_planner    = FollowGapPlanner(GapConfig())
        self.tuner          = AdaptiveTuner(self.dwa_cfg)
        self.recovery       = RecoveryManager()
        self.global_planner = AStarPlanner()
        self.obs_tracker    = DynamicObstacleTracker(dt=self.dwa_cfg.dt)

    def navigate(
        self,
        start:     RobotState,
        goal:      tuple,
        obstacles: np.ndarray,
        max_steps: int = 500,
    ) -> dict:
        state      = start
        steps      = 0
        collisions = 0
        waypoints  = []
        wp_idx     = 0
        history    = []          # for generate_video.py
        in_collision_last = False  # deduplicate: count entry into collision, not every step

        for step in tqdm(range(max_steps), desc="    ↳ Robot Step", leave=False):
            dist = math.hypot(state.x - goal[0], state.y - goal[1])
            if dist < 0.1:
                return {"success": True,  "steps": steps,
                        "collisions": collisions, "final_dist": dist,
                        "history": history}

            if step % self._REPLAN_INTERVAL == 0 or not waypoints:
                path = self.global_planner.plan(state, goal, obstacles)
                if path:
                    waypoints = path
                    wp_idx    = 0

            while wp_idx < len(waypoints) - 1:
                wpx, wpy = waypoints[wp_idx]
                if math.hypot(state.x - wpx, state.y - wpy) < 0.5:
                    wp_idx += 1
                else:
                    break

            local_goal = waypoints[wp_idx] if waypoints else goal
            lidar      = self._fake_lidar(state, obstacles)
            self.tuner.update(lidar)

            self.obs_tracker.update(obstacles)
            v, omega, _ = self.gap_planner.compute_velocity(
                state, local_goal, lidar, dt=self.dwa_cfg.dt
            )
            cmd_v, cmd_omega, _ = self.recovery.step(
                v, omega, lidar, state, dt=self.dwa_cfg.dt
            )

            dt           = self.dwa_cfg.dt
            state.yaw   += cmd_omega * dt
            state.x     += cmd_v * math.cos(state.yaw) * dt
            state.y     += cmd_v * math.sin(state.yaw) * dt
            state.v      = cmd_v
            state.omega  = cmd_omega
            steps       += 1

            history.append((state.x, state.y, state.yaw))

            # count collision only on *entry* -- not every step while overlapping
            if obstacles.size:
                min_d = np.min(np.linalg.norm(obstacles - [state.x, state.y], axis=1))
                currently_colliding = min_d < self.dwa_cfg.robot_radius
                if currently_colliding and not in_collision_last:
                    collisions += 1
                    if collisions >= 1:   # fail immediately on first collision
                        return {"success": False, "steps": steps,
                                "collisions": collisions, "final_dist": dist,
                                "history": history}
                in_collision_last = currently_colliding
            else:
                in_collision_last = False

        final_dist = math.hypot(state.x - goal[0], state.y - goal[1])
        return {"success": False, "steps": steps,
                "collisions": collisions, "final_dist": final_dist,
                "history": history}

    @staticmethod
    def _fake_lidar(state: RobotState, obstacles: np.ndarray, n_beams: int = 360) -> np.ndarray:
        angles = np.linspace(-math.pi, math.pi, n_beams)
        ranges = np.full(n_beams, 3.5)
        noise = np.random.normal(0, 0.03, size=ranges.shape)
        ranges = np.clip(ranges+noise,0.0, 3.5)
        if not obstacles.size:
            return ranges
        for i, angle in enumerate(angles):
            beam_dir = np.array([math.cos(state.yaw + angle), math.sin(state.yaw + angle)])
            for obs in obstacles:
                rel  = obs - np.array([state.x, state.y])
                proj = np.dot(rel, beam_dir)
                if proj <= 0:
                    continue
                perp = np.linalg.norm(rel - proj * beam_dir)
                if perp < 0.25:
                    ranges[i] = min(ranges[i], proj)
        return ranges



if __name__ == "__main__":
    if ROS_AVAILABLE:
        nav = BARNNavigator()
        nav.spin()
    else:
        print("Running simulation demo (no ROS)...")
        import random
        random.seed(0)
        np.random.seed(0)

        obstacles = np.random.uniform(-2, 2, (20, 2))
        start     = RobotState(x=-3.0, y=0.0, yaw=0.0)
        goal      = (3.0, 0.0)

        sim    = SimNavigator()
        result = sim.navigate(start, goal, obstacles, max_steps=800)
        print("Result:", result)
