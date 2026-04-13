import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
from dwa import DWAConfig


class NavContext(Enum):
    OPEN       = auto()   # Low density  – speed up
    CLUTTERED  = auto()   # Medium density – balanced
    NARROW     = auto()   # High density  – slow, careful


@dataclass
class AdaptiveConfig:
    low_density_thresh:  float = 0.12
    high_density_thresh: float = 0.38

    clear_range: float = 1.5            # [m]  obstacle "nearby" threshold

    open_max_speed:          float = 1.0
    open_to_goal_gain:       float = 0.35
    open_obstacle_gain:      float = 1.0
    open_predict_time:       float = 2.4


    cluttered_max_speed:     float = 0.75
    cluttered_to_goal_gain:  float = 0.4
    cluttered_obstacle_gain: float = 1.3
    cluttered_predict_time:  float = 3.0

    narrow_max_speed:        float = 0.45
    narrow_to_goal_gain:     float = 0.45
    narrow_obstacle_gain:    float = 1.8
    narrow_predict_time:     float = 3.2


class AdaptiveTuner:


    def __init__(self, dwa_cfg: DWAConfig, adaptive_cfg: AdaptiveConfig = None):
        self.dwa_cfg  = dwa_cfg
        self.acfg     = adaptive_cfg or AdaptiveConfig()
        self._context = NavContext.CLUTTERED
        self._history: list = []          # rolling density values
        self._window  = 5                 # smoothing window


    def update(self, lidar_ranges: np.ndarray) -> NavContext:
        """
        Call once per control cycle with the raw LiDAR range array.
        Mutates self.dwa_cfg in-place and returns the detected context.
        """
        density = self._compute_density(lidar_ranges)
        self._history.append(density)
        if len(self._history) > self._window:
            self._history.pop(0)
        smooth_density = float(np.mean(self._history))

        new_ctx = self._classify(smooth_density)
        if new_ctx != self._context:
            self._context = new_ctx
            self._apply(new_ctx)

        return self._context

    @property
    def context(self) -> NavContext:
        return self._context


    def _compute_density(self, ranges: np.ndarray) -> float:
        """
        Obstacle density = fraction of scan beams closer than clear_range.
        Ignores inf / nan readings (free space or sensor dropout).
        """
        valid = ranges[np.isfinite(ranges)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(valid < self.acfg.clear_range))

    def _classify(self, density: float) -> NavContext:
        acfg = self.acfg
        if density < acfg.low_density_thresh:
            return NavContext.OPEN
        elif density > acfg.high_density_thresh:
            return NavContext.NARROW
        return NavContext.CLUTTERED

    def _apply(self, ctx: NavContext):
        """Overwrite DWAConfig fields for the selected context."""
        cfg  = self.dwa_cfg
        acfg = self.acfg

        if ctx == NavContext.OPEN:
            cfg.max_speed          = acfg.open_max_speed
            cfg.to_goal_cost_gain  = acfg.open_to_goal_gain
            cfg.obstacle_cost_gain = acfg.open_obstacle_gain
            cfg.predict_time       = acfg.open_predict_time
        elif ctx == NavContext.CLUTTERED:
            cfg.max_speed          = acfg.cluttered_max_speed
            cfg.to_goal_cost_gain  = acfg.cluttered_to_goal_gain
            cfg.obstacle_cost_gain = acfg.cluttered_obstacle_gain
            cfg.predict_time       = acfg.cluttered_predict_time
        else:  # NARROW
            cfg.max_speed          = acfg.narrow_max_speed
            cfg.to_goal_cost_gain  = acfg.narrow_to_goal_gain
            cfg.obstacle_cost_gain = acfg.narrow_obstacle_gain
            cfg.predict_time       = acfg.narrow_predict_time
