"""
risk/planning_tube.py
Phase 2: Planning tube B_plan construction.
Coordinate system: x-z-yaw.
B_plan is used for PLANNING COST ONLY — it does NOT provide safety certification.
Rules:
  - May use smaller margin than B_cert.
  - May use shorter horizon.
  - May use variance clamping.
  - May use myopic discount.
  - Must NOT be used for safety certification.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PlanningTubeElement:
    """
    One element of the planning tube.
    Corresponds to one future time step.
    """
    step: int              # time step index relative to now
    center: np.ndarray     # [x, z] tube center
    radius: float          # tube radius for planning cost
    covariance_radius: float  # uncertainty radius (for cost only)
    margin: float          # planning margin applied
    source: str = "planning"


def build_planning_tube(
    tracks: List[Any],           # List[TemporalEntityState]
    robot_pos: np.ndarray,        # [x, z]
    robot_radius: float,
    dt: float,
    horizon_steps: int = 20,
    margin: float = 0.20,
    covariance_scale: float = 1.0,
) -> List[PlanningTubeElement]:
    """
    Build planning tubes around tracked entities.
    Each tube element is a circle in the x-z plane.

    Args:
        tracks: current tracked entities (from temporal filter)
        robot_pos: robot [x, z]
        robot_radius: collision radius
        dt: time step (s)
        horizon_steps: number of future steps to project
        margin: planning margin (m) — conservative but smaller than certified
        covariance_scale: multiplier on Kalman uncertainty

    Returns:
        List of PlanningTubeElement for all tracks × horizon steps.
    """
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    tube_elements: List[PlanningTubeElement] = []

    for track in tracks:
        pos = np.asarray(track.position, dtype=np.float64)  # [x, z]
        vel = np.asarray(track.velocity, dtype=np.float64)   # [vx, vz]

        # Planning tube radius = entity radius + inflation + planning margin
        entity_radius = float(track.radius)
        uncertainty = float(track.uncertainty_radius) - entity_radius  # uncertainty inflation
        base_radius = entity_radius + uncertainty * float(covariance_scale) + float(margin)

        for step_idx in range(1, horizon_steps + 1):
            t = float(step_idx) * float(dt)
            # Predict entity position (constant velocity)
            center = pos + vel * t
            # Planning radius stays roughly constant
            r = float(base_radius)

            tube_elements.append(PlanningTubeElement(
                step=step_idx,
                center=center.copy(),
                radius=r,
                covariance_radius=float(uncertainty),
                margin=float(margin),
                source="planning",
            ))

    return tube_elements


def tube_overlaps_robot(
    robot_pos: np.ndarray,
    robot_radius: float,
    tube_elem: PlanningTubeElement,
) -> bool:
    """
    Check if robot circle overlaps a planning tube element.
    Returns True if overlap.
    """
    dist = float(np.linalg.norm(np.asarray(robot_pos) - tube_elem.center))
    combined = float(robot_radius) + tube_elem.radius
    return bool(dist < combined)


def tube_to_dict(elem: PlanningTubeElement) -> dict:
    return {
        "step": elem.step,
        "center": elem.center.tolist(),
        "radius": elem.radius,
        "covariance_radius": elem.covariance_radius,
        "margin": elem.margin,
        "source": elem.source,
    }
