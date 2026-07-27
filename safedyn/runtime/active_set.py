"""
runtime/active_set.py
SafeDyn-VLN Guard: Active entity/constraint culling.

Culls tube elements and constraints to only those relevant to the
current planning horizon. Reduces computation by excluding distant
or expired constraints.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class ActiveSetConfig:
    """Configuration for active set culling."""
    max_distance: float = 10.0       # max distance to consider relevant
    min_relevance: float = 0.1       # minimum relevance score
    max_active_constraints: int = 10  # max constraints in active set
    robot_radius: float = 0.25


def compute_relevance(
    robot_pos: np.ndarray,
    tube_center: np.ndarray,
    tube_radius: float,
    max_distance: float,
) -> float:
    """
    Compute relevance score for a tube element.

    Relevance = 1 / (distance + epsilon) * radius_factor
    """
    dist = float(np.linalg.norm(np.asarray(robot_pos) - np.asarray(tube_center)))
    if dist > max_distance:
        return 0.0
    distance_factor = 1.0 / (dist + 0.1)
    radius_factor = min(1.0, tube_radius / 0.5)
    return float(distance_factor * radius_factor)


def cull_active_set(
    robot_pos: np.ndarray,
    b_plan: List[Any],
    b_cert: List[Any],
    config: ActiveSetConfig,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """
    Cull B_plan and B_cert to active set.

    Returns (active_b_plan, active_b_cert, metadata).
    """
    robot_pos = np.asarray(robot_pos, dtype=np.float64)

    # Cull B_plan
    active_plan = []
    for te in b_plan:
        relevance = compute_relevance(
            robot_pos,
            np.asarray(te.center),
            float(te.radius),
            config.max_distance,
        )
        if relevance >= config.min_relevance:
            active_plan.append((relevance, te))
    active_plan.sort(key=lambda x: x[0], reverse=True)
    active_plan = [te for _, te in active_plan[:config.max_active_constraints]]

    # Cull B_cert
    active_cert = []
    for te in b_cert:
        relevance = compute_relevance(
            robot_pos,
            np.asarray(te.center),
            float(te.radius),
            config.max_distance,
        )
        if relevance >= config.min_relevance:
            active_cert.append((relevance, te))
    active_cert.sort(key=lambda x: x[0], reverse=True)
    active_cert = [te for _, te in active_cert[:config.max_active_constraints]]

    metadata = {
        "original_plan_count": len(b_plan),
        "original_cert_count": len(b_cert),
        "active_plan_count": len(active_plan),
        "active_cert_count": len(active_cert),
        "culling_ratio": (len(active_plan) + len(active_cert)) / max(1, len(b_plan) + len(b_cert)),
    }

    return active_plan, active_cert, metadata
