"""
evaluate.py  –  BARN Challenge Batch Evaluator
================================================
Compares two algorithms:
  Baseline  : naive proportional pursuit (no obstacle avoidance, no planning)
  Adaptive  : A* global planning + Follow-the-Gap local planner

Usage:
    python evaluate.py
    python evaluate.py --n_envs 8 --barn_repo /tmp/the-barn-challenge
    python evaluate.py --n_envs 8 --tune --n_trials 10 --tune_envs 5
"""

import argparse
import csv
import json
import math
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from dataclasses import asdict
from typing import List, Optional
from tqdm import tqdm

from dwa import DWAConfig, RobotState
from navigator import SimNavigator, AStarPlanner, BaselineNavigator
from gap_planner import FollowGapPlanner, GapConfig
from simulate import load_barn_world as shared_load_barn_world

JACKAL_MAX_SPEED = 2.0


def load_barn_world(world_idx: int, barn_repo: str) -> Optional[dict]:
    if not (0 <= world_idx < 360):
        return None
    return shared_load_barn_world(world_idx, barn_repo)


def compute_ot(world: dict) -> float:
    path_arr = world.get("path_array")
    if path_arr is not None and len(path_arr) >= 2:
        pts = list(path_arr)
        length = sum(
            math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
            for i in range(len(pts)-1)
        )
        return length / JACKAL_MAX_SPEED
    start, goal, obs = world["start"], world["goal"], world["obstacles"]
    planner = AStarPlanner(grid_res=0.1, inflate_r=0.35)
    path = planner.plan(start, goal, obs)
    if path is None or len(path) < 2:
        return float("inf")
    pts = [(start.x, start.y)] + list(path)
    length = sum(
        math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        for i in range(len(pts)-1)
    )
    return length / JACKAL_MAX_SPEED


def barn_score(success: bool, at: float, ot: float) -> float:
    if not success or ot == float("inf"):
        return 0.0
    return ot / max(2.0*ot, min(at, 8.0*ot))


def run_baseline(world: dict, max_steps: int = 800) -> dict:
    nav = BaselineNavigator(max_speed=0.4)
    t0  = time.perf_counter()
    result = nav.navigate(
        deepcopy(world["start"]), world["goal"], world["obstacles"],
        max_steps=max_steps,
    )
    result["elapsed"] = time.perf_counter() - t0
    ot = compute_ot(world)
    at = result["steps"] * 0.1
    result["ot"] = ot
    result["at"] = at
    result["barn_score"] = barn_score(result["success"], at, ot)
    return result


def run_adaptive(gap_cfg: GapConfig, world: dict, max_steps: int = 800) -> dict:
    sim = SimNavigator(dwa_cfg=DWAConfig())
    sim.gap_planner = FollowGapPlanner(gap_cfg)
    t0  = time.perf_counter()
    result = sim.navigate(
        deepcopy(world["start"]), world["goal"], world["obstacles"],
        max_steps=max_steps,
    )
    result["elapsed"] = time.perf_counter() - t0
    ot = compute_ot(world)
    at = result["steps"] * 0.1
    result["ot"] = ot
    result["at"] = at
    result["barn_score"] = barn_score(result["success"], at, ot)
    return result


def _collect(results: list) -> dict:
    scores      = [r["barn_score"]   for r in results]
    successes   = [float(r["success"]) for r in results]
    times       = [r["steps"]        for r in results if r["success"]]
    collisions  = [r["collisions"]   for r in results]
    final_dists = [r["final_dist"]   for r in results]
    ots         = [r["ot"]           for r in results if r["ot"] != float("inf")]
    ats         = [r["at"]           for r in results if r["success"]]

    # Path efficiency: how close to optimal were successful runs
    efficiencies = []
    for r in results:
        if r["success"] and r["ot"] != float("inf") and r["ot"] > 0:
            efficiencies.append(r["ot"] / r["at"])  # 1.0 = perfect, lower = slower

    # Timeout rate: ran out of steps without succeeding or colliding
    timeouts = [
        float(not r["success"] and r["collisions"] == 0)
        for r in results
    ]

    return {
        "barn_score":       float(np.mean(scores))       if scores       else 0.0,
        "success_rate":     float(np.mean(successes))    if successes    else 0.0,
        "collision_rate":   float(np.mean([float(r["collisions"] > 0) for r in results])),
        "timeout_rate":     float(np.mean(timeouts))     if timeouts     else 0.0,
        "avg_collisions":   float(np.mean(collisions))   if collisions   else 0.0,
        "avg_time_steps":   float(np.mean(times))        if times        else float("inf"),
        "avg_time_sec":     float(np.mean(times)) * 0.1  if times        else float("inf"),
        "avg_final_dist":   float(np.mean(final_dists))  if final_dists  else float("inf"),
        "path_efficiency":  float(np.mean(efficiencies)) if efficiencies else 0.0,
        "avg_ot_sec":       float(np.mean(ots))          if ots          else float("inf"),
        "n_envs":           len(results),
    }

def evaluate_baseline(barn_repo: str, world_indices: List[int],
                      max_steps: int = 800) -> dict:
    results = []
    for idx in tqdm(world_indices, desc="  [Baseline]", leave=False):
        world = load_barn_world(idx, barn_repo)
        if world is None:
            continue
        results.append(run_baseline(world, max_steps))
    return _collect(results)


def evaluate_adaptive(gap_cfg: GapConfig, barn_repo: str,
                      world_indices: List[int], max_steps: int = 800) -> dict:
    results = []
    for idx in tqdm(world_indices, desc="  [Adaptive]", leave=False):
        world = load_barn_world(idx, barn_repo)
        if world is None:
            continue
        results.append(run_adaptive(gap_cfg, world, max_steps))
    return _collect(results)


def tune_gap_config(barn_repo: str, world_indices: List[int],
                    n_trials: int = 15) -> GapConfig:
    param_space = {
        "gap_threshold":    ("uniform", 0.8,  1.8),
        "goal_blend":       ("uniform", 0.5,  0.85),
        "k_angular":        ("uniform", 1.5,  3.5),
        "max_linear_speed": ("uniform", 0.5,  1.0),
        "safety_bubble":    ("uniform", 0.25, 0.5),
        "min_gap_width":    ("choice",  [5, 8, 10, 12]),
    }

    best_score = -float("inf")
    best_cfg   = GapConfig()
    rows       = []
    os.makedirs("results", exist_ok=True)

    for trial in range(n_trials):
        cfg = GapConfig()
        for name, spec in param_space.items():
            if spec[0] == "uniform":
                setattr(cfg, name, random.uniform(spec[1], spec[2]))
            elif spec[0] == "choice":
                setattr(cfg, name, random.choice(spec[1]))

        metrics = evaluate_adaptive(cfg, barn_repo, world_indices)
        score   = metrics["barn_score"]
        print(f"  Trial {trial+1}/{n_trials}: score={score:.4f}  "
              f"success={metrics['success_rate']:.1%}  "
              f"gap_thresh={cfg.gap_threshold:.2f}  "
              f"goal_blend={cfg.goal_blend:.2f}  "
              f"speed={cfg.max_linear_speed:.2f}")

        rows.append({**asdict(cfg), **metrics})
        if score > best_score:
            best_score = score
            best_cfg   = deepcopy(cfg)

    if rows:
        csv_path = "results/gap_search_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Tuning] Saved {len(rows)} trials → {csv_path}")

    print(f"[Tuning] Best BARN score: {best_score:.4f}")
    return best_cfg


def plot_comparison(baseline: dict, adaptive: dict,
                    save_path: str = "results/comparison.png",
                    world_indices: List[int] = None):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    BLUE   = "#4e79a7"
    ORANGE = "#f28e2b"
    labels = ["Baseline\n(Pursuit)", "Adaptive\n(Gap Planner)"]

    subtitle = ""
    if world_indices:
        subtitle = f"  (n={len(world_indices)} random evaluation worlds)"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#f7f7f7")
    axes = axes.flatten() # FIX: Assign the flattened array back to `axes`
    fig.suptitle(
        f"BARN Challenge 2026  |  Baseline vs Follow-the-Gap{subtitle}\n"
        f"Team 10, IIT Kharagpur",
        fontsize=12, fontweight="bold",
    )

    def _bar(ax, vals, title, ylabel, fmt):
        bars = ax.bar(labels, vals, color=[BLUE, ORANGE],
                      edgecolor="white", linewidth=1.2, width=0.5, zorder=3)
        top = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, top * 1.25)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + top*0.02,
                    fmt(v), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    _bar(axes[0],
         [baseline["barn_score"], adaptive["barn_score"]],
         "BARN Score  (higher = better)",
         "OT / clip(AT, 2OT, 8OT)",
         lambda v: f"{v:.3f}")

    _bar(axes[1],
         [baseline["success_rate"]*100, adaptive["success_rate"]*100],
         "Success Rate",
         "% worlds solved",
         lambda v: f"{v:.1f}%")

    axes[1].set_ylim(0, 115)

    _bar(axes[2],
    	 [baseline["path_efficiency"]*100, adaptive["path_efficiency"]*100],
     	 "Path Efficiency  (higher = better)",
     	 "OT / AT  (%)",
     	 lambda v: f"{v:.1f}%")

    _bar(axes[3],
         [baseline["avg_collisions"], adaptive["avg_collisions"]],
         "Avg Collisions / Run  (lower = better)",
         "collision events",
         lambda v: f"{v:.2f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Plot] Saved → {save_path}")
    plt.close()


def _default_barn_repo():
    for c in [
        "/tmp/work/the-barn-challenge",
        "/tmp/the-barn-challenge",
        os.path.expanduser("~/the-barn-challenge"),
    ]:
        if os.path.isdir(c):
            return c
    return "/tmp/work/the-barn-challenge"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--barn_repo",  type=str, default=_default_barn_repo())
    parser.add_argument("--n_envs",     type=int, default=8)
    parser.add_argument("--tune",       action="store_true")
    parser.add_argument("--n_trials",   type=int, default=15)
    parser.add_argument("--tune_envs",  type=int, default=5)
    parser.add_argument("--max_steps",  type=int, default=800)
    args = parser.parse_args()

    # Calculate total required disjoint environments
    total_needed = args.n_envs + (args.tune_envs if args.tune else 0)
    if total_needed > 300:
        print("[WARN] Requested more environments than available (300). Capping.")
        total_needed = 300

    # Randomly select mutually exclusive worlds for tuning and evaluation
    all_sampled = random.sample(range(300), total_needed)
    
    if args.tune:
        tune_indices = all_sampled[:args.tune_envs]
        eval_indices = all_sampled[args.tune_envs:]
    else:
        tune_indices = []
        eval_indices = all_sampled[:args.n_envs]

    print(f"\n[evaluate.py]  BARN repo  : {args.barn_repo}")
    if args.tune:
        print(f"               Tune Worlds  : {tune_indices}")
    print(f"               Eval Worlds  : {eval_indices}")
    print(f"               Max steps    : {args.max_steps}")

    os.makedirs("results", exist_ok=True)

    # 1. Baseline (evaluated on eval_indices)
    print("\n[1/3] Evaluating BASELINE (naive proportional pursuit) ...")
    baseline_metrics = evaluate_baseline(
        args.barn_repo, eval_indices, max_steps=args.max_steps
    )
    print(f"      Baseline → {baseline_metrics}")

    # 2. Tune or use defaults (tuned purely on tune_indices)
    if args.tune:
        print("\n[2/3] Tuning Follow-the-Gap parameters ...")
        best_gap_cfg = tune_gap_config(
            args.barn_repo, tune_indices,
            n_trials=args.n_trials,
        )
    else:
        print("\n[2/3] Using default GapConfig (pass --tune to optimise).")
        best_gap_cfg = GapConfig()

    # 3. Adaptive (evaluated on eval_indices)
    print("\n[3/3] Evaluating ADAPTIVE (A* + Follow-the-Gap) ...")
    adaptive_metrics = evaluate_adaptive(
        best_gap_cfg, args.barn_repo, eval_indices, max_steps=args.max_steps
    )
    print(f"      Adaptive  → {adaptive_metrics}")

    print("\n" + "="*60)
    print(f"  BARN score      :  {baseline_metrics['barn_score']:.4f}"
      f"  →  {adaptive_metrics['barn_score']:.4f}")
    print(f"  Success rate    :  {baseline_metrics['success_rate']:.1%}"
      f"  →  {adaptive_metrics['success_rate']:.1%}")
    print(f"  Collision rate  :  {baseline_metrics['collision_rate']:.1%}"
      f"  →  {adaptive_metrics['collision_rate']:.1%}") 
    print(f"  Timeout rate    :  {baseline_metrics['timeout_rate']:.1%}"
      f"  →  {adaptive_metrics['timeout_rate']:.1%}")
    print(f"  Avg time (s)    :  {baseline_metrics['avg_time_sec']:.1f}"
      f"  →  {adaptive_metrics['avg_time_sec']:.1f}")
    print(f"  Path efficiency :  {baseline_metrics['path_efficiency']:.3f}"
      f"  →  {adaptive_metrics['path_efficiency']:.3f}")
    print(f"  Avg final dist  :  {baseline_metrics['avg_final_dist']:.2f}m"
      f"  →  {adaptive_metrics['avg_final_dist']:.2f}m")
    print(f"  Avg OT (s)      :  {baseline_metrics['avg_ot_sec']:.1f}"
      f"  →  {adaptive_metrics['avg_ot_sec']:.1f}")
    print(f"  Avg collisions  :  {baseline_metrics['avg_collisions']:.2f}"
      f"  →  {adaptive_metrics['avg_collisions']:.2f}")
    print("="*60)
    # Save
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    with open("results/adaptive_metrics.json", "w") as f:
        json.dump(adaptive_metrics, f, indent=2)
    with open("results/best_gap_config.json", "w") as f:
        json.dump(asdict(best_gap_cfg), f, indent=2)

    plot_comparison(baseline_metrics, adaptive_metrics,
                    world_indices=eval_indices)
    print("\n[Done] All results saved in results/")
