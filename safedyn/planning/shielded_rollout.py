"""
planning/shielded_rollout.py
SafeDyn-VLN Guard: Shielded Rollout for fast safety check inside MPPI.

Uses B_plan (not B_cert) for fast approximate clearance check during
trajectory sampling. B_cert is only used by CertifiedAccept for final
certification.

Logs predicted shield interventions.
Does not use B_cert as planner panic cost.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ShieldConfig:
    """Configuration for shielded rollout."""
    clearance_threshold: float = 0.05  # minimum clearance to pass shield
    intervention_log_max: int = 100    # max log entries
    use_b_plan_only: bool = True       # never use B_cert in planner


@dataclass
class ShieldIntervention:
    """Record of a shield intervention during rollout."""
    step: int
    position: np.ndarray
    nearest_tube_step: int
    clearance: float
    intervention_type: str  # "clearance_violation", "boundary_violation"


@dataclass
class ShieldedRolloutResult:
    """Result of shielded rollout check."""
    passed: bool
    min_clearance: float
    interventions: List[ShieldIntervention]
    steps_checked: int
    clearance_profile: List[float]


def shielded_rollout_check(
    positions: np.ndarray,
    yaws: np.ndarray,
    b_plan: List[Any],
    robot_radius: float,
    config: ShieldConfig,
) -> ShieldedRolloutResult:
    """
    Fast approximate safety check using B_plan during MPPI rollout.

    This is NOT a safety certification — it's a planning-time cost signal.
    B_cert is never used here.

    Args:
        positions: (H+1, 2) trajectory positions
        yaws: (H+1,) trajectory yaws
        b_plan: planning tube elements
        robot_radius: robot collision radius
        config: shield configuration

    Returns:
        ShieldedRolloutResult with pass/fail, clearance info, interventions
    """
    H = len(positions) - 1
    if H <= 0:
        return ShieldedRolloutResult(
            passed=True, min_clearance=float("inf"),
            interventions=[], steps_checked=0, clearance_profile=[],
        )

    # Group tube by step
    tube_by_step: Dict[int, List[Any]] = {}
    for te in b_plan:
        s = int(getattr(te, 'step', 0))
        tube_by_step.setdefault(s, []).append(te)

    min_clearance = float("inf")
    interventions: List[ShieldIntervention] = []
    clearance_profile: List[float] = []
    passed = True

    for i in range(1, min(H + 1, len(positions))):
        rp = positions[i]
        step_clearance = float("inf")

        for te in tube_by_step.get(i, []):
            dist = float(np.linalg.norm(rp - np.asarray(te.center)))
            clearance = dist - robot_radius - float(te.radius)
            step_clearance = min(step_clearance, clearance)

            if clearance < config.clearance_threshold:
                interventions.append(ShieldIntervention(
                    step=i,
                    position=rp.copy(),
                    nearest_tube_step=int(te.step),
                    clearance=clearance,
                    intervention_type="clearance_violation",
                ))
                passed = False

        clearance_profile.append(step_clearance)
        min_clearance = min(min_clearance, step_clearance)

    # Truncate log if too long
    if len(interventions) > config.intervention_log_max:
        interventions = interventions[:config.intervention_log_max]

    return ShieldedRolloutResult(
        passed=passed,
        min_clearance=float(min_clearance),
        interventions=interventions,
        steps_checked=min(H, len(positions) - 1),
        clearance_profile=clearance_profile,
    )


def shielded_rollout_score(
    result: ShieldedRolloutResult,
    clearance_weight: float = 10.0,
) -> float:
    """
    Convert shielded rollout result to a cost score (lower is better).

    Positive clearance reduces cost; violations increase cost heavily.
    """
    if result.min_clearance < 0:
        return clearance_weight * abs(result.min_clearance) * 10.0
    return -clearance_weight * result.min_clearance
