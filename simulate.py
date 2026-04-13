"""
simulate.py  -  BARN Challenge Navigation Visualiser
=====================================================
Loads a *real* BARN world (obstacles extracted from the official
occupancy-grid .npy files or path_files) and animates the
Adaptive-DWA navigator through it.

Directory layout expected (matches the-barn-challenge repo):
  BARN_REPO_ROOT/
    jackal_helper/worlds/
      BARN/
        world_<idx>.world
        path_files/
          path_<idx>.npy          ← precomputed optimal-path grid coords
      DynaBARN/
        world_<idx>.world

Usage (from /tmp/work):
    python simulate.py --world_idx 0
    python simulate.py --world_idx 42 --save results/w42.mp4 --no-live
"""

import math
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works in Apptainer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

from dwa import DWA, DWAConfig, RobotState
from adaptive_tuner import AdaptiveTuner, NavContext
from recovery import RecoveryManager
from navigator import SimNavigator
from gap_planner import FollowGapPlanner, GapConfig
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  Colour palette (dark-mode)
# ──────────────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
GRID_CLR  = "#21262d"
OBS_CLR   = "#e74c3c"
ROBOT_CLR = "#2ecc71"
GOAL_CLR  = "#f39c12"
TRAJ_CLR  = "#3498db"
PATH_CLR  = "#9b59b6"
TEXT_CLR  = "#ecf0f1"
LIDAR_CLR = "#ffffff"

CTX_COLORS = {
    NavContext.OPEN:      "#2ecc71",
    NavContext.CLUTTERED: "#f39c12",
    NavContext.NARROW:    "#e74c3c",
}

# ──────────────────────────────────────────────────────────────────────────────
#  BARN world constants  (from run.py in the official repo)
# ──────────────────────────────────────────────────────────────────────────────
CYLINDER_RADIUS = 0.075          # [m]  radius of each occupancy cylinder
GRID_SIZE       = 30             # 30×30 cells

# Coordinate transform used in the official run.py
def path_coord_to_gazebo(x: float, y: float):
    """Convert BARN grid cell indices → Gazebo world metres."""
    r_shift = -CYLINDER_RADIUS - (GRID_SIZE * CYLINDER_RADIUS * 2)
    c_shift = CYLINDER_RADIUS + 5
    gx = x * (CYLINDER_RADIUS * 2) + r_shift
    gy = y * (CYLINDER_RADIUS * 2) + c_shift
    return gx, gy

# ──────────────────────────────────────────────────────────────────────────────
#  Load BARN world
# ──────────────────────────────────────────────────────────────────────────────
def load_barn_world(world_idx: int, barn_repo: str) -> dict:
    """
    Return a dict with:
      obstacles  : np.ndarray (N, 2)  – obstacle centres in Gazebo metres
      start      : RobotState
      goal       : (float, float)
      path_array : np.ndarray (M, 2) – reference path in Gazebo metres (or None)
    """
    worlds_dir = os.path.join(barn_repo, "jackal_helper", "worlds")

    if world_idx < 300:                                  # static BARN worlds
        path_file = os.path.join(worlds_dir, "BARN", "path_files", f"path_{world_idx}.npy")
        init_pos  = (-2.25, 3.0, 1.57)                  
        goal_off  = (0.0, 10.0)                          
    elif world_idx < 360:                                # DynaBARN worlds
        path_file = None                                 
        init_pos  = (11.0, 0.0, 3.14)
        goal_off  = (-20.0, 0.0)
    else:
        raise ValueError(f"World index {world_idx} does not exist (max 359).")

    start = RobotState(x=init_pos[0], y=init_pos[1], yaw=init_pos[2])
    goal  = (init_pos[0] + goal_off[0], init_pos[1] + goal_off[1])

    obstacles  = np.empty((0, 2))
    path_array = None

    # 1. ALWAYS try to load the reference path first
    if path_file and os.path.isfile(path_file):
        path_cells = np.load(path_file)
        path_array = np.array([
            path_coord_to_gazebo(float(p[0]), float(p[1]))
            for p in path_cells
        ])
        path_array = np.vstack([[start.x, start.y], path_array, [goal[0], goal[1]]])

    # 2. Try to load OBSTACLES from Occupancy Grid (Fastest)
    occ_file = os.path.join(worlds_dir, "BARN", f"world_{world_idx}.npy")
    world_file = os.path.join(worlds_dir, "BARN", f"world_{world_idx}.world")

    if os.path.isfile(occ_file):
        grid = np.load(occ_file)
        obstacles = _grid_to_obstacles(grid)
        print(f"[INFO] Loaded {len(obstacles)} obstacles from fast occupancy grid (.npy).")
        
    # 3. Fallback: Parse OBSTACLES from Gazebo XML (Slower, but reliable)
    elif os.path.isfile(world_file):
        obstacles = _parse_world_file(world_file)
        print(f"[INFO] Loaded {len(obstacles)} obstacles from Gazebo world file (.world).")
    else:
        print(f"[ERROR] Cannot find obstacle data (.npy or .world) for index {world_idx}.")
        sys.exit(1)
    return {
        "obstacles":  obstacles,
        "start":      start,
        "goal":       goal,
        "path_array": path_array,
        "world_idx":  world_idx,
    }

def _grid_to_obstacles(grid: np.ndarray) -> np.ndarray:
    occ = (grid > 0)
    rows, cols = np.where(occ)
    if len(rows) == 0:
        return np.empty((0, 2))
    
    obs = []
    for r, c in zip(rows, cols):
        gx, gy = path_coord_to_gazebo(float(r), float(c))
        obs.append([gx, gy])
    obs = np.array(obs)
    
    return obs

def _parse_world_file(world_file: str) -> np.ndarray:
    """
    Parse only INTERIOR obstacle cylinders.
    Skip the dense boundary wall cylinders that form the arena rectangle.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(world_file)
        world_el = tree.getroot().find("world")
        if world_el is None:
            world_el = tree.getroot()

        obs = []
        for model in world_el:
            if model.tag != "model":
                continue
            name = model.get("name", "")
            if "unit_cylinder" not in name:
                continue

            pose_el = model.find("pose")
            if pose_el is not None and pose_el.text:
                vals = [float(v) for v in pose_el.text.strip().split()]
                x, y = vals[0], vals[1]
                obs.append([x, y])

        print(f"[Parser] {world_file.split('/')[-1]}: {len(obs)} cylinders loaded")
        return np.array(obs) if obs else np.empty((0, 2))

    except Exception as e:
        print(f"[WARN] Could not parse {world_file}: {e}")
        return np.empty((0, 2))

# ──────────────────────────────────────────────────────────────────────────────
#  SimRecorder  –  runs the navigator and captures every frame
# ──────────────────────────────────────────────────────────────────────────────
class SimRecorder:

    def __init__(self, world: dict, dwa_cfg: DWAConfig = None, gap_cfg: GapConfig = None):
        self.obstacles = world["obstacles"]
        self.goal      = world["goal"]
        self.start     = world["start"]
        self.path_ref  = world.get("path_array")    # reference path (may be None)
        self.world_idx = world.get("world_idx", 0)

        self.cfg         = dwa_cfg or DWAConfig()
        self.gap_planner = FollowGapPlanner(gap_cfg or GapConfig())
        self.tuner       = AdaptiveTuner(self.cfg)
        self.recovery    = RecoveryManager()
        self.frames: list = []

    def run(self, max_steps: int = 1000):
        from navigator import AStarPlanner 
        
        state = deepcopy(self.start)
        dt    = self.cfg.dt
        print(f"[SimRecorder] Running world {self.world_idx}: start=({state.x:.2f},{state.y:.2f}) → goal={self.goal}")

        global_planner = AStarPlanner(grid_res=0.1, x_range=(-8.0, 8.0), y_range=(-2.0, 16.0))
        waypoints = []
        wp_idx = 0
        
        collisions = 0
        in_collision_last = False

        for step in tqdm(range(max_steps), desc="Navigating Maze", unit="step"):
            dist = math.hypot(state.x - self.goal[0], state.y - self.goal[1])
            if dist < 0.05:
                print(f"\n[SimRecorder] Goal reached at step {step}!")
                break

            if step % 50 == 0 or not waypoints:
                path = global_planner.plan(state, self.goal, self.obstacles)
                if path:
                    waypoints = path
                    wp_idx    = 0

            while wp_idx < len(waypoints) - 1:
                wpx, wpy = waypoints[wp_idx]
                if math.hypot(state.x - wpx, state.y - wpy) < 0.5:
                    wp_idx += 1
                else:
                    break

            local_goal = waypoints[wp_idx] if waypoints else self.goal

            lidar         = self._fake_lidar(state)
            ctx           = self.tuner.update(lidar)
            v, omega, gap_debug = self.gap_planner.compute_velocity(
                state, local_goal, lidar, dt=self.cfg.dt
            )
            traj = gap_debug  
            cmd_v, cmd_omega, in_rec = self.recovery.step(v, omega, lidar, state)

            # Check for collisions
            if self.obstacles.size:
                min_d = np.min(np.linalg.norm(self.obstacles - [state.x, state.y], axis=1))
                currently_colliding = min_d < self.cfg.robot_radius
                if currently_colliding and not in_collision_last:
                    collisions += 1
                in_collision_last = currently_colliding
            else:
                in_collision_last = False

            self.frames.append({
                "x": state.x, "y": state.y, "yaw": state.yaw,
                "v": cmd_v, "omega": cmd_omega, "traj": traj,
                "lidar": lidar, "ctx": ctx, "in_recovery": in_rec,
                "step": step, "dist": dist, "collisions": collisions
            })

            state.yaw  += cmd_omega * dt
            state.x    += cmd_v * math.cos(state.yaw) * dt
            state.y    += cmd_v * math.sin(state.yaw) * dt
            state.v     = cmd_v
            state.omega = cmd_omega

        print(f"[SimRecorder] {len(self.frames)} frames recorded.")

    def _fake_lidar(self, state: RobotState, n_beams: int = 180, max_r: float = 3.5):
        angles = np.linspace(-math.pi, math.pi, n_beams)
        ranges = np.full(n_beams, max_r)
        if not self.obstacles.size:
            return ranges
        for i, angle in enumerate(angles):
            bx = math.cos(state.yaw + angle)
            by = math.sin(state.yaw + angle)
            for ox, oy in self.obstacles:
                rx, ry = ox - state.x, oy - state.y
                proj = rx * bx + ry * by
                if proj <= 0:
                    continue
                perp = math.sqrt(max(0.0, rx*rx + ry*ry - proj*proj))
                if perp < CYLINDER_RADIUS + 0.02:
                    ranges[i] = min(ranges[i], proj)
        return ranges


# ──────────────────────────────────────────────────────────────────────────────
#  NavAnimator  –  builds and saves the animation
# ──────────────────────────────────────────────────────────────────────────────
class NavAnimator:

    N_LIDAR = 180

    def __init__(self, recorder: SimRecorder):
        self.rec    = recorder
        self.frames = recorder.frames
        self.obs    = recorder.obstacles
        self.goal   = recorder.goal
        self.start  = recorder.start
        self.path_ref = recorder.path_ref

        self.path_x: list = []
        self.path_y: list = []

        all_x = list(self.obs[:, 0]) if self.obs.size else []
        all_y = list(self.obs[:, 1]) if self.obs.size else []
        all_x += [self.start.x, self.goal[0]]
        all_y += [self.start.y, self.goal[1]]
        pad = 1.0
        self._xlim = (min(all_x) - pad, max(all_x) + pad)
        self._ylim = (min(all_y) - pad, max(all_y) + pad)

        self._build_figure()

    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 7), facecolor=BG)
        self.fig.suptitle(
            f"BARN Challenge – Adaptive DWA  |  World {self.rec.world_idx}  "
            f"|  Team 10, IIT Kharagpur",
            color=TEXT_CLR, fontsize=12, fontweight="bold", y=0.98,
        )
        gs = self.fig.add_gridspec(
            2, 3, width_ratios=[3, 1, 1],
            hspace=0.4, wspace=0.35,
            left=0.05, right=0.97, top=0.92, bottom=0.07,
        )

        self.ax      = self.fig.add_subplot(gs[:, 0], facecolor=BG)
        self.ax_spd  = self.fig.add_subplot(gs[0, 1], facecolor=BG)
        self.ax_omg  = self.fig.add_subplot(gs[0, 2], facecolor=BG)
        self.ax_ctx  = self.fig.add_subplot(gs[1, 1], facecolor=BG)
        self.ax_dist = self.fig.add_subplot(gs[1, 2], facecolor=BG)

        for ax, title, xlim, ylim in [
            (self.ax,      "Navigation Map",          self._xlim,  self._ylim),
            (self.ax_spd,  "Linear Speed [m/s]",      (0, 1),      (0, 1)),
            (self.ax_omg,  "Angular Speed [rad/s]",   (0, 1),      (0, 1)),
            (self.ax_ctx,  "Nav Context",              (0, 1),      (0, 1)),
            (self.ax_dist, "Dist to Goal [m]",         None,        (0, None)), # Removed static xlim
        ]:
            ax.set_facecolor(BG)
            ax.set_title(title, color=TEXT_CLR, fontsize=9, pad=4)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_CLR)
            ax.tick_params(colors=TEXT_CLR, labelsize=7)
            ax.grid(color=GRID_CLR, linewidth=0.5)
            if xlim: ax.set_xlim(*xlim)
            if ylim and ylim[1] is not None:
                ax.set_ylim(*ylim)

        self.ax_ctx.set_xticks([])
        self.ax_ctx.set_yticks([])

    def _draw_static(self):
        ax = self.ax
        for ox, oy in self.obs:
            c = plt.Circle(
                (ox, oy), CYLINDER_RADIUS, color=OBS_CLR, alpha=0.85, zorder=3
            )
            ax.add_patch(c)
        if self.path_ref is not None and len(self.path_ref) > 1:
            ax.plot(
                self.path_ref[:, 0], self.path_ref[:, 1],
                color="#555577", linewidth=1.2, linestyle=":",
                alpha=0.6, zorder=2, label="Ref path",
            )
        ax.plot(*self.goal, "*", color=GOAL_CLR, markersize=18, zorder=6)
        ax.annotate(
            "GOAL", self.goal,
            textcoords="offset points", xytext=(6, 6),
            color=GOAL_CLR, fontsize=8, fontweight="bold",
        )
        ax.plot(self.start.x, self.start.y, "s",
                color="#95a5a6", markersize=10, zorder=4)
        ax.annotate(
            "START", (self.start.x, self.start.y),
            textcoords="offset points", xytext=(5, 5),
            color="#95a5a6", fontsize=8,
        )

    def _init_dynamic(self):
        ax = self.ax
        self.robot_patch = plt.Circle(
            (self.start.x, self.start.y), CYLINDER_RADIUS * 2,
            color=ROBOT_CLR, alpha=0.9, zorder=7,
        )
        ax.add_patch(self.robot_patch)
        self.heading_arrow, = ax.plot([], [], color=ROBOT_CLR, linewidth=2.5, zorder=8)

        self.lidar_lines = [
            ax.plot([], [], color=LIDAR_CLR, alpha=0.07, linewidth=0.5, zorder=2)[0]
            for _ in range(self.N_LIDAR)
        ]
        self.traj_line, = ax.plot(
            [], [], color=TRAJ_CLR, linewidth=2, linestyle="--", alpha=0.8, zorder=5
        )
        self.path_line, = ax.plot(
            [], [], color=PATH_CLR, linewidth=1.5, alpha=0.6, zorder=4
        )

        cfg = self.rec.cfg
        self.spd_bar = self.ax_spd.barh(
            [0.5], [0], height=0.35, color=ROBOT_CLR
        )
        self.ax_spd.set_xlim(cfg.min_speed, cfg.max_speed)
        self.ax_spd.set_yticks([])
        self.spd_text = self.ax_spd.text(
            0.5, 0.15, "0.00", ha="center",
            color=TEXT_CLR, fontsize=9, transform=self.ax_spd.transAxes,
        )

        self.omg_bar = self.ax_omg.barh(
            [0.5], [0], height=0.35, color=TRAJ_CLR
        )
        self.ax_omg.set_xlim(-cfg.max_yaw_rate, cfg.max_yaw_rate)
        self.ax_omg.set_yticks([])
        self.omg_text = self.ax_omg.text(
            0.5, 0.15, "0.00", ha="center",
            color=TEXT_CLR, fontsize=9, transform=self.ax_omg.transAxes,
        )

        self.ctx_text = self.ax_ctx.text(
            0.5, 0.5, "CLUTTERED", ha="center", va="center",
            color=CTX_COLORS[NavContext.CLUTTERED],
            fontsize=14, fontweight="bold",
            transform=self.ax_ctx.transAxes,
        )

        self.dist_line, = self.ax_dist.plot([], [], color=GOAL_CLR, linewidth=1.5)
        max_dist = max(
            math.hypot(f["x"] - self.goal[0], f["y"] - self.goal[1])
            for f in self.frames
        ) if self.frames else 10.0
        self.ax_dist.set_ylim(0, max_dist * 1.1)
        self._dist_xs: list = []
        self._dist_ys: list = []

        self.status_text = ax.text(
            self._xlim[0] + 0.1, self._ylim[0] + 0.2, "",
            color=TEXT_CLR, fontsize=7.5,
            family="monospace", zorder=10,
        )
        
        self.collision_text = ax.text(
            self._xlim[1] - 0.5, self._ylim[1] - 0.5, "",
            color=OBS_CLR, fontsize=10, fontweight="bold",
            ha="right", va="top", zorder=10,
        )

    def _update(self, frame_idx: int):
        f   = self.frames[frame_idx]
        x, y, yaw = f["x"], f["y"], f["yaw"]
        cfg = self.rec.cfg

        self.robot_patch.set_center((x, y))
        hlen = CYLINDER_RADIUS * 3
        self.heading_arrow.set_data(
            [x, x + hlen * math.cos(yaw)],
            [y, y + hlen * math.sin(yaw)],
        )

        angles = np.linspace(-math.pi, math.pi, self.N_LIDAR)
        lidar  = f["lidar"]
        n      = min(len(lidar), self.N_LIDAR)
        for i in range(n):
            ang = yaw + angles[i]
            ex  = x + lidar[i] * math.cos(ang)
            ey  = y + lidar[i] * math.sin(ang)
            self.lidar_lines[i].set_data([x, ex], [y, ey])

        gap_debug = f["traj"]
        if isinstance(gap_debug, dict) and gap_debug.get("chosen_gap"):
            gap    = gap_debug["chosen_gap"]
            angle  = yaw + gap.deep_angle
            length = min(gap.deep_range, 2.5)
            ex     = x + length * math.cos(angle)
            ey     = y + length * math.sin(angle)
            self.traj_line.set_data([x, ex], [y, ey])
        else:
            self.traj_line.set_data([], [])

        self.path_x.append(x)
        self.path_y.append(y)
        self.path_line.set_data(self.path_x, self.path_y)

        v, omega = f["v"], f["omega"]
        self.spd_bar[0].set_width(v)
        self.spd_text.set_text(f"{v:+.3f} m/s")
        self.omg_bar[0].set_width(omega)
        self.omg_text.set_text(f"{omega:+.3f} r/s")

        ctx  = f["ctx"]
        clr  = CTX_COLORS.get(ctx, TEXT_CLR)
        self.ctx_text.set_text(ctx.name if ctx else "—")
        self.ctx_text.set_color(clr)
        self.robot_patch.set_color(clr)

        self._dist_xs.append(f["step"])
        self._dist_ys.append(f["dist"])
        self.dist_line.set_data(self._dist_xs, self._dist_ys)
        
        # DYNAMIC X-AXIS FOR DISTANCE GRAPH
        self.ax_dist.set_xlim(0, max(10, f["step"]))

        rec_flag = "[RECOVERY]" if f["in_recovery"] else ""
        self.status_text.set_text(
            f"Step {f['step']:03d} | dist={f['dist']:.2f}m | "
            f"v={v:+.2f} ω={omega:+.2f}  {rec_flag}"
        )
        
        # UPDATE COLLISION COUNTER
        cols = f.get("collisions", 0)
        if cols > 0:
            self.collision_text.set_text(f"COLLISIONS: {cols}")
        else:
            self.collision_text.set_text("")

        return (
            self.robot_patch, self.heading_arrow,
            self.traj_line, self.path_line, self.dist_line,
            *self.lidar_lines,
        )

    def animate(
        self,
        save_path:   str  = "results/simulation.gif",
        interval_ms: int  = 80,
        show_live:   bool = False,
    ):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        self._draw_static()
        self._init_dynamic()

        anim = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.frames),
            interval=interval_ms,
            blit=False,
            repeat=False,
        )

        ext = os.path.splitext(save_path)[1].lower()

        if ext == ".mp4":
            try:
                writer = animation.FFMpegWriter(fps=15, bitrate=1800)
                print(f"[Animator] Saving MP4 → {save_path} ...")
                anim.save(
                    save_path, writer=writer, dpi=120,
                    savefig_kwargs={"facecolor": BG},
                )
                print("[Animator] MP4 saved ✓")
            except Exception as e:
                print(f"[Animator] ffmpeg failed ({e}). Falling back to GIF.")
                gif_path = save_path.replace(".mp4", ".gif")
                anim.save(gif_path, writer="pillow", fps=12,
                          savefig_kwargs={"facecolor": BG})
                print(f"[Animator] GIF saved → {gif_path}")

        else:   # default: GIF
            print(f"[Animator] Saving GIF → {save_path} ...")
            anim.save(save_path, writer="pillow", fps=12,
                      savefig_kwargs={"facecolor": BG})
            print("[Animator] GIF saved ✓")

        if show_live:
            plt.show()

        plt.close(self.fig)
        return anim


# ──────────────────────────────────────────────────────────────────────────────
#  Config helpers
# ──────────────────────────────────────────────────────────────────────────────
def _default_barn_repo():
    candidates = [
        "/tmp/work/the-barn-challenge",
        os.path.join(os.path.dirname(__file__), "..", "the-barn-challenge"),
        os.path.expanduser("~/the-barn-challenge"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return "/tmp/work/the-barn-challenge"


def load_tuned_config(config_path: str = None) -> DWAConfig:
    import json
    from dataclasses import fields as dc_fields

    candidates = ([config_path] if config_path else []) + [
        "results/adaptive_best_config.json",
        "results/best_config.json",
    ]

    loaded_path = None
    raw         = {}
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    raw = json.load(f)
                loaded_path = path
                break
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")

    cfg = DWAConfig()
    if loaded_path:
        valid = {field.name for field in dc_fields(DWAConfig)}
        applied, skipped = [], []
        for k, v in raw.items():
            if k in valid:
                setattr(cfg, k, v)
                applied.append(k)
            else:
                skipped.append(k)
        print(f"[DWAConfig] Loaded from: {loaded_path}")
        if skipped:
            print(f"            Ignored  : {', '.join(skipped)}")
    else:
        print("[DWAConfig] No saved config found. Using hand-crafted defaults.")
        cfg.max_speed          = 0.7
        cfg.obstacle_cost_gain = 3.0
        cfg.to_goal_cost_gain  = 0.20
        cfg.predict_time       = 3.5
        cfg.speed_cost_gain    = 1.0

    # Always override these for smooth animation
    cfg.max_accel           = 1.5
    cfg.max_delta_yaw_rate  = 2.0
    cfg.v_resolution        = 0.05
    cfg.yaw_rate_resolution = 0.1
    cfg.robot_radius        = 0.25
    cfg.obstacle_clearance  = 0.05
    return cfg


def load_tuned_gap_config(config_path: str = None) -> GapConfig:
    import json
    from dataclasses import fields as dc_fields

    candidates = ([config_path] if config_path else []) + [
        "results/best_gap_config.json",
    ]

    loaded_path = None
    raw         = {}
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    raw = json.load(f)
                loaded_path = path
                break
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")

    cfg   = GapConfig()
    valid = {field.name for field in dc_fields(GapConfig)}
    if loaded_path:
        skipped = []
        for k, v in raw.items():
            if k in valid:
                setattr(cfg, k, v)
            else:
                skipped.append(k)
        print(f"[GapConfig] Loaded from: {loaded_path}")
        if skipped:
            print(f"            Ignored  : {', '.join(skipped)}")
    else:
        print("[GapConfig] No saved config found. Using GapConfig defaults.")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate Adaptive Gap Planner on a real BARN world and save a video."
    )
    parser.add_argument(
        "--world_idx", type=int, default=0,
        help="BARN world index (0-299 static, 300-359 DynaBARN)",
    )
    parser.add_argument(
        "--barn_repo", type=str, default=_default_barn_repo(),
        help="Path to the cloned the-barn-challenge repository root",
    )
    parser.add_argument(
        "--save", type=str, default="results/simulation.gif",
        help="Output file path (.gif or .mp4)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a GapConfig JSON "
             "(default: auto-detect results/best_gap_config.json)",
    )
    parser.add_argument(
        "--no-live", dest="show_live", action="store_false",
        help="Do not open an interactive window (useful in headless Apptainer)",
    )
    parser.set_defaults(show_live=False)
    args = parser.parse_args()

    print(f"[simulate.py] BARN repo  : {args.barn_repo}")
    print(f"[simulate.py] World index: {args.world_idx}")

    # 1. Load the real BARN world
    world = load_barn_world(args.world_idx, args.barn_repo)

    # 2. Load DWA config (used by AdaptiveTuner context detection inside SimRecorder)
    dwa_cfg = load_tuned_config()

    # 3. Load tuned GapConfig from evaluate.py output — primary local planner
    gap_cfg = load_tuned_gap_config(args.config)
    print(f"[simulate.py] gap_threshold={gap_cfg.gap_threshold:.2f}  "
          f"goal_blend={gap_cfg.goal_blend:.2f}  "
          f"max_speed={gap_cfg.max_linear_speed:.2f}  "
          f"safety_bubble={gap_cfg.safety_bubble:.2f}")

    # 4. Record simulation — inject the tuned gap planner
    recorder = SimRecorder(world, dwa_cfg=dwa_cfg, gap_cfg=gap_cfg)
    recorder.run(max_steps=args.max_steps)

    # Subsample to every 3rd frame to reduce GIF size
    recorder.frames = recorder.frames[::3]
    print(f"[Animator] Subsampled to {len(recorder.frames)} frames for GIF output.")

    if not recorder.frames:
        print("[ERROR] No frames recorded – the robot may be immediately stuck.")
        print("        Try a different --world_idx.")
        sys.exit(1)

    # 4. Animate and save
    animator = NavAnimator(recorder)
    animator.animate(
        save_path=args.save,
        interval_ms=80,
        show_live=args.show_live,
    )

    print(f"\n[Done] Video saved to : {args.save}")
    print(f"       Total frames    : {len(recorder.frames)}")
    last = recorder.frames[-1]
    print(f"       Final dist goal : {last['dist']:.2f} m")
