# BARN Challenge Navigation – Team 10
**AI61006: Artificial Intelligence for Cyber Physical Systems**  
IIT Kharagpur · Spring 2026

---

## Team Members
| Roll No.     | Name         |
|--------------|--------------|
| 25AI60R02    | Ravindra Mina|
| 25AI60R12    | Raghav Kapil |
| 25AI60R20    | Sarthak Dey  |
| 25AI60R11    | Ritish Bhatt |
| 25AI60R23    | Aishik Das   |

---

## Project Structure

```
barn_nav/
├── dwa.py                  # Core DWA algorithm + config
├── adaptive_tuner.py       # Obstacle-density-based parameter adaptation
├── recovery.py             # Stuck detection + recovery behaviours
├── hyperparameter_tuning.py# Grid / random / Bayesian search
├── navigator.py            # ROS node + pure-Python SimNavigator
├── evaluate.py             # Batch evaluation + comparison plots
└── README.md
```

---

## Setup

### 1 — ROS + Gazebo (Full Pipeline)
```bash
# Install ROS Noetic (Ubuntu 20.04)
sudo apt-get install ros-noetic-desktop-full

# Clone BARN Challenge repo
git clone https://github.com/Daffan/the-barn-challenge.git
cd the-barn-challenge && pip install -r requirements.txt

# Copy this package into your catkin workspace
cp -r barn_nav ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make
source devel/setup.bash

# Launch BARN environment + navigator
roslaunch barn_nav barn_nav.launch
```

### 2 — Simulation-Only (No ROS Required)
```bash
pip install numpy matplotlib scikit-learn scipy
python evaluate.py
```

---

## Key Modules

### `DWAConfig` (dwa.py)
All hyperparameters in one dataclass. Key fields:

| Parameter            | Default | Effect                              |
|----------------------|---------|-------------------------------------|
| `max_speed`          | 0.5 m/s | Maximum forward velocity            |
| `obstacle_cost_gain` | 1.0     | Penalises proximity to obstacles    |
| `to_goal_cost_gain`  | 0.15    | Penalises heading away from goal    |
| `predict_time`       | 3.0 s   | Trajectory simulation horizon       |
| `robot_radius`       | 0.3 m   | Safety exclusion zone               |

### `AdaptiveTuner` (adaptive_tuner.py)
Classifies each LiDAR scan into one of three contexts and adjusts
DWAConfig in-place:

| Context   | Density    | Behaviour                        |
|-----------|------------|----------------------------------|
| OPEN      | < 15%      | Faster speed, lower caution      |
| CLUTTERED | 15%–45%    | Balanced (baseline)              |
| NARROW    | > 45%      | Slow, high obstacle cost         |

### `RecoveryManager` (recovery.py)
Three-phase recovery sequence:
1. **ROTATE** – spin in-place toward clearest direction
2. **BACKUP** – reverse briefly
3. **ARC** – forward arc to escape the stuck region

### Hyperparameter Search (hyperparameter_tuning.py)
```python
from hyperparameter_tuning import RandomSearchTuner, ResultLogger
from dwa import DWAConfig

logger = ResultLogger(objective="success_rate")
tuner  = RandomSearchTuner(
    base_cfg    = DWAConfig(),
    param_space = {
        "max_speed":          ("uniform", 0.2, 0.8),
        "obstacle_cost_gain": ("log",     0.5, 3.0),
        "predict_time":       ("uniform", 1.5, 5.0),
    },
    evaluator   = my_evaluator_fn,
    n_trials    = 50,
    logger      = logger,
)
best_cfg, best_score = tuner.run()
```

---

## Running the Evaluation

```bash
python evaluate.py
```

Outputs:
- `results/search_results.csv`      — all trial metrics
- `results/best_config.json`        — best DWA configuration found
- `results/adaptive_best_config.json`
- `results/comparison.png`          — bar chart: Baseline vs Adaptive

---

## References
1. BARN Challenge — https://people.cs.gmu.edu/~xiao/Research/BARN_Challenge/
2. GitHub — https://github.com/Daffan/the-barn-challenge
3. Fox, Burgard, Thrun. *The Dynamic Window Approach to Collision Avoidance.* IEEE RA, 1997.
# Automation_navigation_in_cluttered_environment
