"""
planning/mppi.py
SafeDyn-VLN Guard: MPPI (Model Predictive Path Integral) controller.

Implements batched trajectory sampling over H horizon with K samples.
Each sample is a sequence of [v, omega] actions.
Returns best trajectory, best first action, best mode, and score.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MPPIConfig:
    """Configuration for MPPI controller."""
    K: int = 50                     # number of samples
    H: int = 10                     # horizon steps
    dt: float = 0.05                # time step
    lambda_mppi: float = 1.0        # temperature parameter
    noise_std_v: float = 0.15       # noise std for linear velocity
    noise_std_omega: float = 0.3    # noise std for angular velocity
    v_min: float = 0.0
    v_max: float = 0.8
    omega_min: float = -1.2
    omega_max: float = 1.2
    robot_radius: float = 0.25
    seed: Optional[int] = None      # deterministic seed for tests
    cost_clearance_weight: float = 10.0
    cost_goal_weight: float = 5.0
    cost_smoothness_weight: float = 1.0
    cost_tube_weight: float = 15.0


@dataclass
class MPPIResult:
    """Result of MPPI optimization."""
    best_action: Dict[str, float]
    best_trajectory: np.ndarray    # (H, 2) actions [v, omega]
    best_mode: str
    best_score: float
    all_scores: List[float]
    cost_components: Dict[str, float]
    samples_evaluated: int
    valid_samples: int


def simulate_action_sequence(
    start_pos: np.ndarray,
    start_yaw: float,
    actions: np.ndarray,
    dt: float,
    v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate robot forward given a sequence of actions.

    Args:
        start_pos: [x, z]
        start_yaw: yaw angle
        actions: (H, 2) array of [v, omega] per step
        dt: time step
        v_max: max linear velocity

    Returns:
        positions: (H+1, 2) array of [x, z] per step
        yaws: (H+1,) array of yaw per step
    """
    H = actions.shape[0]
    positions = np.zeros((H + 1, 2))
    yaws = np.zeros(H + 1)
    positions[0] = start_pos
    yaws[0] = start_yaw

    for i in range(H):
        v = np.clip(actions[i, 0], 0.0, v_max)
        omega = actions[i, 1]
        yaw = yaws[i]
        pos = positions[i]
        positions[i + 1] = pos + np.array([v * np.sin(yaw), -v * np.cos(yaw)]) * dt
        yaws[i + 1] = yaw + omega * dt

    return positions, yaws


def score_trajectory(
    positions: np.ndarray,
    yaws: np.ndarray,
    actions: np.ndarray,
    goal_pos: np.ndarray,
    b_plan: List[Any],
    robot_radius: float,
    config: MPPIConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    Score a trajectory for cost minimization (lower is better).

    Components:
      - tube cost: distance to nearest B_plan element (penalize proximity)
      - goal cost: distance to goal (reward progress)
      - smoothness cost: action magnitude (penalize large controls)
    """
    H = actions.shape[0]
    goal_pos = np.asarray(goal_pos, dtype=np.float64)

    # Tube cost: min clearance across trajectory
    tube_cost = 0.0
    if b_plan:
        tube_by_step: Dict[int, List[Any]] = {}
        for te in b_plan:
            s = int(getattr(te, 'step', 0))
            tube_by_step.setdefault(s, []).append(te)

        for i in range(1, H + 1):
            if i >= len(positions):
                break
            rp = positions[i]
            for te in tube_by_step.get(i, []):
                dist = float(np.linalg.norm(rp - np.asarray(te.center)))
                clearance = dist - robot_radius - float(te.radius)
                if clearance < 0:
                    tube_cost += config.cost_tube_weight * abs(clearance) * 10.0
                else:
                    tube_cost += config.cost_tube_weight / (clearance + 0.1)

    # Goal cost: final distance to goal
    final_pos = positions[min(H, len(positions) - 1)]
    goal_dist = float(np.linalg.norm(final_pos - goal_pos))
    goal_cost = config.cost_goal_weight * goal_dist

    # Smoothness cost
    smoothness_cost = config.cost_smoothness_weight * float(
        np.sum(actions[:, 0]**2 + actions[:, 1]**2)
    )

    total = tube_cost + goal_cost + smoothness_cost
    components = {
        "tube": tube_cost,
        "goal": goal_cost,
        "smoothness": smoothness_cost,
        "total": total,
    }
    return total, components


def mppi_optimize(
    start_pos: np.ndarray,
    start_yaw: float,
    nominal_action: np.ndarray,
    goal_pos: np.ndarray,
    b_plan: List[Any],
    config: MPPIConfig,
    mode: str = "forward",
) -> MPPIResult:
    """
    Run MPPI optimization for one mode.

    Args:
        start_pos: current robot [x, z]
        start_yaw: current yaw
        nominal_action: nominal [v, omega] to center samples around
        goal_pos: goal position [x, z]
        b_plan: planning tube elements
        config: MPPI configuration
        mode: 'forward', 'left', 'right', 'yield', 'stop'

    Returns:
        MPPIResult with best action, trajectory, mode, score
    """
    rng = np.random.default_rng(config.seed)
    K, H = config.K, config.H
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(1, 2)

    # Mode-specific biases
    mode_bias = np.zeros(2)
    if mode == "left":
        mode_bias[1] = 0.3   # bias toward positive omega (turn left)
    elif mode == "right":
        mode_bias[1] = -0.3  # bias toward negative omega (turn right)
    elif mode == "yield":
        nominal = np.array([[0.1, 0.0]])  # slow down
    elif mode == "stop":
        nominal = np.array([[0.0, 0.0]])

    # Generate noise
    noise_v = rng.normal(0, config.noise_std_v, (K, H))
    noise_omega = rng.normal(0, config.noise_std_omega, (K, H))
    noise = np.stack([noise_v, noise_omega], axis=-1)  # (K, H, 2)

    # Generate sample trajectories
    base_actions = np.tile(nominal, (K, H, 1))  # (K, H, 2)
    base_actions[:, :, 1] += mode_bias[1]
    sample_actions = base_actions + noise  # (K, H, 2)

    # Clip actions
    sample_actions[:, :, 0] = np.clip(sample_actions[:, :, 0], config.v_min, config.v_max)
    sample_actions[:, :, 1] = np.clip(sample_actions[:, :, 1], config.omega_min, config.omega_max)

    # Score each sample
    scores = np.zeros(K)
    cost_components_all = {"tube": 0.0, "goal": 0.0, "smoothness": 0.0}
    valid_count = 0

    for k in range(K):
        positions, yaws = simulate_action_sequence(
            start_pos, start_yaw, sample_actions[k], config.dt, config.v_max,
        )
        score, components = score_trajectory(
            positions, yaws, sample_actions[k], goal_pos,
            b_plan, config.robot_radius, config,
        )
        scores[k] = score
        for key in cost_components_all:
            cost_components_all[key] += components[key]
        valid_count += 1

    # Weighted average for best action (importance sampling)
    weights = np.exp(-scores / max(config.lambda_mppi, 1e-6))
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights_normalized = weights / weights_sum
    else:
        weights_normalized = np.ones(K) / K

    # Best action = first step of best sample
    best_idx = int(np.argmin(scores))
    best_action_arr = sample_actions[best_idx, 0]
    best_action = {
        "linear_velocity": float(best_action_arr[0]),
        "angular_velocity": float(best_action_arr[1]),
    }
    best_trajectory = sample_actions[best_idx]

    return MPPIResult(
        best_action=best_action,
        best_trajectory=best_trajectory,
        best_mode=mode,
        best_score=float(scores[best_idx]),
        all_scores=scores.tolist(),
        cost_components={k: v / max(K, 1) for k, v in cost_components_all.items()},
        samples_evaluated=K,
        valid_samples=valid_count,
    )
