"""
planning/mode_conditioned.py
SafeDyn-VLN Guard: Mode-Conditioned Shielded MPPI.

Runs MPPI optimization across multiple modes: forward, left, right, yield, stop.
Each mode samples K action sequences over H horizon.
Selects best mode by score, with hysteresis for mode switching.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from safedyn.planning.mppi import MPPIConfig, MPPIResult, mppi_optimize


@dataclass
class ModeConditionedConfig:
    """Configuration for mode-conditioned MPPI."""
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    modes: Tuple[str, ...] = ("forward", "left", "right", "yield", "stop")
    mode_hysteresis: float = 0.1    # minimum score improvement to switch modes
    switch_penalty: float = 0.05    # penalty for switching modes
    current_mode: str = "forward"
    current_mode_steps: int = 0


@dataclass
class ModeConditionedResult:
    """Result of mode-conditioned MPPI."""
    best_action: Dict[str, float]
    best_mode: str
    best_score: float
    mode_scores: Dict[str, float]
    mode_results: Dict[str, MPPIResult]
    switched: bool
    previous_mode: str
    hysteresis_applied: bool
    total_samples: int


def mode_conditioned_mppi(
    start_pos: np.ndarray,
    start_yaw: float,
    nominal_action: Dict[str, float],
    goal_pos: np.ndarray,
    b_plan: List[Any],
    config: ModeConditionedConfig,
) -> ModeConditionedResult:
    """
    Run mode-conditioned MPPI across all modes.

    For each mode:
      1. Run MPPI with K samples over H horizon
      2. Score the best trajectory
      3. Select mode with best score (with hysteresis)

    Returns best action, mode, and all mode results.
    """
    nominal_arr = np.array([
        nominal_action.get("linear_velocity", 0.0),
        nominal_action.get("angular_velocity", 0.0),
    ], dtype=np.float64)

    mode_results: Dict[str, MPPIResult] = {}
    mode_scores: Dict[str, float] = {}

    for mode in config.modes:
        # Create per-mode config with different seed offsets
        mode_config = MPPIConfig(
            K=config.mppi.K,
            H=config.mppi.H,
            dt=config.mppi.dt,
            lambda_mppi=config.mppi.lambda_mppi,
            noise_std_v=config.mppi.noise_std_v,
            noise_std_omega=config.mppi.noise_std_omega,
            v_min=config.mppi.v_min,
            v_max=config.mppi.v_max,
            omega_min=config.mppi.omega_min,
            omega_max=config.mppi.omega_max,
            robot_radius=config.mppi.robot_radius,
            seed=(config.mppi.seed if config.mppi.seed is not None
                  else None),
            cost_clearance_weight=config.mppi.cost_clearance_weight,
            cost_goal_weight=config.mppi.cost_goal_weight,
            cost_smoothness_weight=config.mppi.cost_smoothness_weight,
            cost_tube_weight=config.mppi.cost_tube_weight,
        )

        result = mppi_optimize(
            start_pos=start_pos,
            start_yaw=start_yaw,
            nominal_action=nominal_arr,
            goal_pos=goal_pos,
            b_plan=b_plan,
            config=mode_config,
            mode=mode,
        )
        mode_results[mode] = result
        mode_scores[mode] = result.best_score

    # Select best mode with hysteresis
    previous_mode = config.current_mode
    best_mode = previous_mode
    best_score = mode_scores.get(previous_mode, float("inf"))
    hysteresis_applied = False

    for mode in config.modes:
        if mode == previous_mode:
            continue
        score = mode_scores[mode]
        # Apply switch penalty
        adjusted_score = score + config.switch_penalty
        # Switch only if significantly better
        if adjusted_score < best_score - config.mode_hysteresis:
            best_mode = mode
            best_score = score
            hysteresis_applied = True

    # If current mode is no longer feasible, force switch
    if previous_mode in mode_results:
        prev_result = mode_results[previous_mode]
        # Check if best action is essentially stop
        if (prev_result.best_action.get("linear_velocity", 0.0) < 0.01
                and prev_result.best_action.get("angular_velocity", 0.0) < 0.01):
            # Current mode stuck — find best non-stop mode
            for mode in config.modes:
                if mode == "stop":
                    continue
                if mode_scores[mode] < best_score:
                    best_mode = mode
                    best_score = mode_scores[mode]
                    hysteresis_applied = True

    switched = best_mode != previous_mode

    # Get best action from selected mode
    if best_mode in mode_results:
        best_action = mode_results[best_mode].best_action
    else:
        best_action = {"linear_velocity": 0.0, "angular_velocity": 0.0}

    total_samples = sum(r.samples_evaluated for r in mode_results.values())

    return ModeConditionedResult(
        best_action=best_action,
        best_mode=best_mode,
        best_score=best_score,
        mode_scores=mode_scores,
        mode_results=mode_results,
        switched=switched,
        previous_mode=previous_mode,
        hysteresis_applied=hysteresis_applied,
        total_samples=total_samples,
    )
