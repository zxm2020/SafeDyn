"""
planning/myopic_risk.py
SafeDyn-VLN Guard: Myopic Risk Shaping for planning cost.

Near-term risk has high weight; far-term risk is discounted.
Uses B_plan only (not B_cert) for cost shaping.
Supports adaptive horizon.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class MyopicRiskConfig:
    """Configuration for myopic risk shaping."""
    near_term_horizon: int = 5      # steps considered "near-term"
    near_term_weight: float = 2.0   # weight for near-term risk
    far_term_discount: float = 0.8  # discount factor per step beyond near-term
    clearance_power: float = 2.0    # power law exponent for clearance cost
    min_clearance_threshold: float = 0.1  # minimum clearance to consider
    adaptive_horizon_enabled: bool = True
    max_adaptive_horizon: int = 20
    min_adaptive_horizon: int = 5


def compute_myopic_risk(
    positions: np.ndarray,
    b_plan: List[Any],
    robot_radius: float,
    config: MyopicRiskConfig,
    robot_speed: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute myopic risk cost for a trajectory.

    Near-term steps (0..near_term_horizon) get full weight.
    Far-term steps get exponentially discounted weight.

    Args:
        positions: (H+1, 2) trajectory positions
        b_plan: planning tube elements
        robot_radius: robot collision radius
        config: myopic risk configuration
        robot_speed: current robot speed (for adaptive horizon)

    Returns:
        dict with total_risk, per_step_risks, adaptive_horizon, components
    """
    H = len(positions) - 1
    if H <= 0:
        return {
            "total_risk": 0.0,
            "per_step_risks": [],
            "adaptive_horizon": H,
            "components": {"near_term": 0.0, "far_term": 0.0},
        }

    # Adaptive horizon: reduce horizon when speed is low
    adaptive_H = H
    if config.adaptive_horizon_enabled:
        # At low speed, near-term is more important
        speed_factor = min(1.0, robot_speed / 0.4)
        adaptive_H = max(
            config.min_adaptive_horizon,
            int(config.min_adaptive_horizon + speed_factor * (H - config.min_adaptive_horizon))
        )
        adaptive_H = min(adaptive_H, min(config.max_adaptive_horizon, H))

    if not b_plan:
        return {
            "total_risk": 0.0,
            "per_step_risks": [],
            "adaptive_horizon": adaptive_H,
            "components": {"near_term": 0.0, "far_term": 0.0},
        }

    # Group tube elements by step
    tube_by_step: Dict[int, List[Any]] = {}
    for te in b_plan:
        s = int(getattr(te, 'step', 0))
        tube_by_step.setdefault(s, []).append(te)

    per_step_risks = []
    near_term_risk = 0.0
    far_term_risk = 0.0

    for i in range(1, min(adaptive_H + 1, len(positions))):
        rp = positions[i]
        step_risk = 0.0

        for te in tube_by_step.get(i, []):
            dist = float(np.linalg.norm(rp - np.asarray(te.center)))
            clearance = dist - robot_radius - float(te.radius)

            if clearance < config.min_clearance_threshold:
                # Risk increases as clearance decreases
                risk = (config.min_clearance_threshold - clearance + 0.01) ** config.clearance_power
            else:
                # Exponential decay of risk with clearance
                risk = np.exp(-2.0 * clearance)

            step_risk += risk

        # Apply time-based weighting
        if i <= config.near_term_horizon:
            weight = config.near_term_weight
            near_term_risk += step_risk * weight
        else:
            steps_beyond = i - config.near_term_horizon
            weight = config.near_term_weight * (config.far_term_discount ** steps_beyond)
            far_term_risk += step_risk * weight

        per_step_risks.append(step_risk * weight)

    total_risk = near_term_risk + far_term_risk

    return {
        "total_risk": float(total_risk),
        "per_step_risks": per_step_risks,
        "adaptive_horizon": adaptive_H,
        "components": {
            "near_term": float(near_term_risk),
            "far_term": float(far_term_risk),
        },
    }


def shape_planning_cost(
    base_cost: float,
    myopic_risk: Dict[str, Any],
    risk_weight: float = 1.0,
) -> float:
    """
    Combine base trajectory cost with myopic risk cost.

    Returns weighted total cost.
    """
    return base_cost + risk_weight * myopic_risk.get("total_risk", 0.0)
