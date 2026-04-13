import csv
import json
import random
import time
import itertools
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from dwa import DWAConfig



# Evaluator: receives a DWAConfig, returns a dict of metrics
#   e.g. {"success_rate": 0.8, "avg_time": 12.3, "collisions": 1}
Evaluator = Callable[[DWAConfig], Dict[str, float]]




class ResultLogger:
    """Logs every evaluated configuration to CSV and tracks the best."""

    def __init__(self, log_dir: str = "results", objective: str = "success_rate"):
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.objective = objective
        self._best_score: float = -np.inf
        self._best_cfg:   Optional[DWAConfig] = None
        self._rows: List[dict] = []

    def log(self, cfg: DWAConfig, metrics: Dict[str, float]):
        score = metrics.get(self.objective, 0.0)
        row   = {**asdict(cfg), **metrics, "timestamp": time.time()}
        self._rows.append(row)

        if score > self._best_score:
            self._best_score = score
            self._best_cfg   = deepcopy(cfg)
            self._save_best()

    def flush(self, filename: str = "search_results.csv"):
        if not self._rows:
            return
        path = self.log_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._rows[0].keys())
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[ResultLogger] Saved {len(self._rows)} rows → {path}")

    def _save_best(self):
        if self._best_cfg is None:
            return
        path = self.log_dir / "best_config.json"
        with open(path, "w") as f:
            json.dump(asdict(self._best_cfg), f, indent=2)

    @property
    def best(self) -> Tuple[Optional[DWAConfig], float]:
        return self._best_cfg, self._best_score



def _apply_params(base_cfg: DWAConfig, params: Dict[str, Any]) -> DWAConfig:
    cfg = deepcopy(base_cfg)
    for k, v in params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"DWAConfig has no attribute '{k}'")
    return cfg



class GridSearchTuner:

    def __init__(
        self,
        base_cfg:  DWAConfig,
        grid:      Dict[str, List[Any]],
        evaluator: Evaluator,
        logger:    Optional[ResultLogger] = None,
    ):
        self.base_cfg  = base_cfg
        self.grid      = grid
        self.evaluator = evaluator
        self.logger    = logger or ResultLogger()

    def run(self) -> Tuple[Optional[DWAConfig], float]:
        keys   = list(self.grid.keys())
        combos = list(itertools.product(*[self.grid[k] for k in keys]))
        total  = len(combos)
        print(f"[GridSearch] {total} combinations to evaluate.")

        for i, values in enumerate(combos):
            params  = dict(zip(keys, values))
            cfg     = _apply_params(self.base_cfg, params)
            metrics = self.evaluator(cfg)
            self.logger.log(cfg, metrics)
            print(
                f"  [{i+1}/{total}] {params} → "
                f"success={metrics.get('success_rate', '?'):.2f}  "
                f"time={metrics.get('avg_time', '?'):.1f}s"
            )

        self.logger.flush()
        return self.logger.best


class RandomSearchTuner:

    def __init__(
        self,
        base_cfg:    DWAConfig,
        param_space: Dict[str, Tuple],
        evaluator:   Evaluator,
        n_trials:    int = 50,
        seed:        int = 42,
        logger:      Optional[ResultLogger] = None,
    ):
        self.base_cfg    = base_cfg
        self.param_space = param_space
        self.evaluator   = evaluator
        self.n_trials    = n_trials
        self.logger      = logger or ResultLogger()
        random.seed(seed)
        np.random.seed(seed)

    def _sample(self) -> Dict[str, Any]:
        params = {}
        for name, spec in self.param_space.items():
            kind = spec[0]
            if kind == "uniform":
                params[name] = random.uniform(spec[1], spec[2])
            elif kind == "choice":
                params[name] = random.choice(spec[1])
            elif kind == "log":
                params[name] = float(np.exp(random.uniform(np.log(spec[1]), np.log(spec[2]))))
            else:
                raise ValueError(f"Unknown distribution: {kind}")
        return params

    def run(self) -> Tuple[Optional[DWAConfig], float]:
        print(f"[RandomSearch] {self.n_trials} random trials.")
        for i in tqdm(range(self.n_trials), desc="Optimization Trials"):
            params  = self._sample()
            cfg     = _apply_params(self.base_cfg, params)
            metrics = self.evaluator(cfg)
            self.logger.log(cfg, metrics)
            print(
                f"  Trial {i+1}/{self.n_trials}: {params} → "
                f"success={metrics.get('success_rate', '?'):.2f}"
            )
        self.logger.flush("random_search_results.csv")
        return self.logger.best



class BayesianTuner:

    def __init__(
        self,
        base_cfg:    DWAConfig,
        param_bounds: Dict[str, Tuple[float, float]],   # name → (lo, hi)
        evaluator:   Evaluator,
        n_init:      int = 5,
        n_iter:      int = 20,
        logger:      Optional[ResultLogger] = None,
    ):
        self.base_cfg     = base_cfg
        self.param_bounds = param_bounds
        self.evaluator    = evaluator
        self.n_init       = n_init
        self.n_iter       = n_iter
        self.logger       = logger or ResultLogger()
        self._names       = list(param_bounds.keys())
        self._bounds_arr  = np.array([param_bounds[n] for n in self._names])

    def run(self) -> Tuple[Optional[DWAConfig], float]:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from scipy.stats import norm
        except ImportError:
            print("[BayesianTuner] scikit-learn / scipy not found. Falling back to random search.")
            rs = RandomSearchTuner(
                self.base_cfg,
                {n: ("uniform", lo, hi) for n, (lo, hi) in self.param_bounds.items()},
                self.evaluator,
                n_trials=self.n_init + self.n_iter,
                logger=self.logger,
            )
            return rs.run()

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.stats import norm

        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5
        )
        X_obs, y_obs = [], []

        def _evaluate_point(x_vec):
            params  = dict(zip(self._names, x_vec))
            cfg     = _apply_params(self.base_cfg, params)
            metrics = self.evaluator(cfg)
            self.logger.log(cfg, metrics)
            return metrics.get("success_rate", 0.0)


        for i in range(self.n_init):
            x = np.array([np.random.uniform(lo, hi) for lo, hi in self._bounds_arr])
            y = _evaluate_point(x)
            X_obs.append(x); y_obs.append(y)
            print(f"  [BO init {i+1}/{self.n_init}] score={y:.3f}")

        for i in range(self.n_iter):
            gp.fit(np.array(X_obs), np.array(y_obs))
            x_next = self._next_point(gp, norm, np.max(y_obs))
            y_next = _evaluate_point(x_next)
            X_obs.append(x_next); y_obs.append(y_next)
            print(f"  [BO iter {i+1}/{self.n_iter}] score={y_next:.3f}")

        self.logger.flush("bayesian_search_results.csv")
        return self.logger.best

    def _next_point(self, gp, norm, y_best: float, n_candidates: int = 1000) -> np.ndarray:
        candidates = np.random.uniform(
            self._bounds_arr[:, 0], self._bounds_arr[:, 1],
            size=(n_candidates, len(self._names))
        )
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        Z  = (mu - y_best) / sigma
        ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return candidates[np.argmax(ei)]
