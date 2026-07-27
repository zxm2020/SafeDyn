"""
safety/exact_rollout.py
Phase 3A: Exact rollout validation using B_cert.
Coordinate system: x-z-yaw.

Validates whether an action candidate causes the robot circle to overlap
any certified tube (B_cert) element over a short simulation horizon.

Uses the current robot unicycle dynamics:
  - x += v * sin(yaw) * dt
  - z += -v * cos(yaw) * dt
  - yaw += omega * dt
  - v clamped to [0, max_v] per action

Only B_cert is used for geometric overlap checks.
B_plan is NOT used for certification.

Returns True if the action is rollout-safe (no overlap in any step).
"""

import numpy as np
from typing import List, Dict, Any, Tuple


# ── Robot dynamics ────────────────────────────────────────────────────────

def simulate_robot_path(
    start_pos: np.ndarray,   # [x, z]
    start_yaw: float,
    action: Dict[str, Any],
    dt: float,
    max_steps: int,
    max_linear_velocity: float = 0.8,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Simulate robot state over `max_steps` using unicycle dynamics.

    Returns:
      positions: list of [x, z] arrays per step
      yaws: list of yaw angles per step
    """
    pos = np.asarray(start_pos, dtype=np.float64).copy()
    yaw = float(start_yaw)

    positions = [pos.copy()]
    yaws = [yaw]

    v = float(action.get("linear_velocity", 0.0))
    omega = float(action.get("angular_velocity", 0.0))

    # Clamp velocity to [0, max]
    v = max(0.0, min(abs(v), max_linear_velocity))

    for _ in range(max_steps):
        pos = pos + np.array([v * np.sin(yaw), -v * np.cos(yaw)], dtype=np.float64) * dt
        yaw = yaw + omega * dt
        positions.append(pos.copy())
        yaws.append(yaw)

    return positions, yaws


def robot_overlaps_tube_elem(
    robot_pos: np.ndarray,
    robot_radius: float,
    tube_elem,
) -> bool:
    """
    Check if robot circle (center, radius) overlaps a tube element.
    """
    dist = float(np.linalg.norm(np.asarray(robot_pos) - np.asarray(tube_elem.center)))
    combined = float(robot_radius) + float(tube_elem.radius)
    return bool(dist < combined)


def action_passes_rollout(
    start_pos: np.ndarray,
    start_yaw: float,
    action: Dict[str, Any],
    b_cert: List,
    dt: float,
    horizon_steps: int,
    robot_radius: float,
    max_linear_velocity: float = 0.8,
) -> bool:
    """
    Check if `action` causes robot to overlap any B_cert element.

    Returns True if rollout-safe (no overlap).
    Returns False if any overlap detected.
    """
    if not b_cert:
        return True   # No tubes → trivially safe

    positions, _ = simulate_robot_path(
        start_pos, start_yaw, action, dt,
        max_steps=horizon_steps,
        max_linear_velocity=max_linear_velocity,
    )

    # Group tube elements by step
    tube_by_step: dict = {}
    for te in b_cert:
        step = int(te.step)
        if step not in tube_by_step:
            tube_by_step[step] = []
        tube_by_step[step].append(te)

    # Check each future step
    for step_idx in range(1, horizon_steps + 1):
        if step_idx >= len(positions):
            break
        rp = positions[step_idx]
        for te in tube_by_step.get(step_idx, []):
            if robot_overlaps_tube_elem(rp, robot_radius, te):
                return False

    return True


# ── Batch validation ──────────────────────────────────────────────────────

def find_rollout_safe_candidates(
    start_pos: np.ndarray,
    start_yaw: float,
    candidate_actions: List[Dict[str, Any]],
    b_cert: List,
    dt: float,
    horizon_steps: int,
    robot_radius: float,
    max_linear_velocity: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Find all candidate actions that pass exact rollout.

    Returns list of actions (dicts) that are rollout-safe.
    Empty list → emergency_stop.
    """
    safe = []
    for action in candidate_actions:
        if action_passes_rollout(
            start_pos, start_yaw, action, b_cert, dt,
            horizon_steps, robot_radius, max_linear_velocity,
        ):
            safe.append(action)
    return safe
