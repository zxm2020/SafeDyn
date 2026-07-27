"""
safety/backup_validator.py
Phase 3B-2 Fix: Moving-Entity-Aware Backup Validator.
Coordinate system: x-z-yaw.

CRITICAL ASSUMPTION:
  The dynamic entity is assumed to continue moving at its current velocity
  indefinitely. It does NOT stop when the robot stops.

  stop is feasible ONLY if the robot's stationary position does not overlap
  with ANY future B_cert element over the prediction horizon.

Rules:
  - B_cert is used for all safety checks.
  - B_plan is NOT used for safety certification.
  - Entity tracking lag is a known limitation (see Phase 3B-3).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class BackupResult:
    """Result of evaluating one backup candidate."""
    name: str
    is_feasible: bool
    min_clearance: float
    score: float
    rollout_min_distance: float
    goal_progress: float
    control_change: float
    first_violation_step: int = -1
    reason: str = ""
    future_tube_step_used: int = -1
    # Phase 3B-2b: Additional debug fields
    initial_clearance: float = 0.0
    final_clearance: float = 0.0
    clearance_trend: float = 0.0
    min_ttc: float = float("inf")
    moving_toward_entity: bool = False
    forward_toward_entity_score: float = 0.0


# ── Entity path simulation ───────────────────────────────────────────────────────

def simulate_entity_path(
    start_pos,
    velocity,
    dt: float,
    horizon_steps: int,
) -> List:
    """
    Simulate entity at constant velocity over horizon.
    Entity NEVER stops for robot (conservative safe assumption).
    """
    pos = np.asarray(start_pos, dtype=np.float64).copy()
    positions = [pos.copy()]
    for _ in range(horizon_steps):
        pos = pos + np.asarray(velocity, dtype=np.float64) * float(dt)
        positions.append(pos.copy())
    return positions


def _combined_radius(robot_radius: float, cert_tube: List) -> float:
    """Compute conservative combined collision radius."""
    if cert_tube:
        return float(robot_radius) + max(float(te.radius) for te in cert_tube)
    return float(robot_radius) + 0.2


def compute_min_clearance(
    robot_positions: List,
    entity_positions: List,
    robot_radius: float,
    cert_tube: List,
) -> float:
    """
    Minimum clearance across all timesteps.
    clearance = center_distance - combined_radius.
    Negative = overlap (infeasible).
    """
    if not entity_positions:
        return float("inf")
    combined_r = _combined_radius(robot_radius, cert_tube)
    min_cl = float("inf")
    n = min(len(robot_positions), len(entity_positions))
    for i in range(n):
        d = float(np.linalg.norm(robot_positions[i] - entity_positions[i]))
        cl = d - combined_r
        if cl < min_cl:
            min_cl = cl
    return min_cl


def compute_rollout_clearance_profile(
    robot_start_pos,
    robot_start_yaw: float,
    robot_action: Dict[str, Any],
    entity_start_pos,
    entity_velocity,
    cert_tube: List,
    robot_radius: float,
    dt: float,
    horizon_steps: int,
    max_linear_velocity: float = 0.8,
) -> Tuple[float, float, float, float, float, float, int, str, int]:
    """
    Compute clearance profile for robot executing robot_action against future B_cert.

    Returns: (min_clearance, initial_clearance, final_clearance, min_ttc,
              min_closing_speed, min_center_distance, first_violation_step, reason, constraining_step)

    Clearance negative → infeasible.
    """
    from safedyn.safety.exact_rollout import simulate_robot_path

    # Simulate robot trajectory
    robot_positions, robot_yaws = simulate_robot_path(
        np.asarray(robot_start_pos, dtype=np.float64),
        float(robot_start_yaw),
        robot_action,
        float(dt),
        max_steps=horizon_steps,
        max_linear_velocity=float(max_linear_velocity),
    )

    # Simulate entity trajectory (for TTC calculation)
    entity_positions = simulate_entity_path(
        np.asarray(entity_start_pos, dtype=np.float64),
        np.asarray(entity_velocity, dtype=np.float64),
        float(dt),
        horizon_steps,
    )

    # Group cert tube elements by step
    tube_by_step: dict = {}
    if cert_tube:
        for te in cert_tube:
            step = int(te.step)
            if step not in tube_by_step:
                tube_by_step[step] = []
            tube_by_step[step].append(te)

    min_cl = float("inf")
    min_d = float("inf")
    initial_clearance = float("inf")
    final_clearance = float("inf")
    min_ttc = float("inf")
    min_closing_speed = 0.0
    first_violation_step = -1
    constraining_step = -1
    reason = ""

    # Check each future step
    for i in range(min(len(robot_positions), horizon_steps + 1)):
        rp = robot_positions[i]
        step_tubes = tube_by_step.get(i, [])

        # Track initial and final clearance
        if i == 0 and step_tubes:
            for te in step_tubes:
                d = float(np.linalg.norm(rp - np.asarray(te.center)))
                cl = d - (robot_radius + float(te.radius))
                if cl < initial_clearance:
                    initial_clearance = cl

        # Check against B_cert tubes at this step
        for te in step_tubes:
            tube_dist = float(np.linalg.norm(rp - np.asarray(te.center)))
            tube_cl = tube_dist - (robot_radius + float(te.radius))

            if tube_dist < min_d:
                min_d = tube_dist

            if tube_cl < min_cl:
                min_cl = tube_cl
                constraining_step = i

            # Track first violation
            if first_violation_step < 0 and tube_cl <= 0:
                first_violation_step = i
                reason = f"overlap_with_future_tube_step_{i}"

        # TTC calculation using entity positions
        if i < len(entity_positions):
            d_to_entity = float(np.linalg.norm(rp - entity_positions[i]))
            combined_r = _combined_radius(robot_radius, cert_tube)
            clearance_to_entity = d_to_entity - combined_r

            # Track final clearance (at end of horizon)
            if i == min(len(robot_positions), horizon_steps + 1) - 1:
                final_clearance = clearance_to_entity

            # Calculate closing speed (relative velocity along line of sight)
            if i > 0 and i < len(entity_positions):
                robot_vel = np.array([robot_action.get("linear_velocity", 0.0) * np.sin(robot_yaws[i-1]),
                                     -robot_action.get("linear_velocity", 0.0) * np.cos(robot_yaws[i-1])])
                rel_vel = np.asarray(entity_velocity) - robot_vel
                rel_pos = entity_positions[i] - rp
                if np.linalg.norm(rel_pos) > 1e-6:
                    closing_speed = np.dot(rel_vel, rel_pos) / np.linalg.norm(rel_pos)
                    if closing_speed > 0:
                        ttc = clearance_to_entity / closing_speed if closing_speed > 1e-3 else float("inf")
                        if ttc > 0 and ttc < min_ttc:
                            min_ttc = ttc
                        if closing_speed > min_closing_speed:
                            min_closing_speed = closing_speed

    if min_cl <= 0 and not reason:
        reason = "overlap_with_future_tube"

    # Handle case where no tubes were checked
    if initial_clearance == float("inf"):
        initial_clearance = min_cl
    if final_clearance == float("inf"):
        final_clearance = min_cl

    return (float(min_cl), float(initial_clearance), float(final_clearance),
            float(min_ttc), float(min_closing_speed), float(min_d),
            first_violation_step, reason, constraining_step)


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_backup_score(
    candidate,
    min_clearance: float,
    initial_clearance: float,
    final_clearance: float,
    clearance_trend: float,
    min_ttc: float,
    moving_toward_entity: bool,
    nominal_action: Dict[str, Any],
    max_linear_velocity: float = 0.8,
    ttc_threshold: float = 2.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Phase 3B-2b: TTC-aware and Direction-aware backup scoring.

    Score components:
      + 10.0 * min_clearance
      + 5.0 * clearance_trend (positive = moving away is good)
      + 2.0 * min_ttc (higher TTC is better)
      - 5.0 * moving_toward_entity_penalty
      - reverse_penalty
      - turn_penalty
      - smoothness_penalty

    Returns: (score, score_components dict)
    """
    score_components = {}

    # Clearance component (most important)
    clearance_component = 10.0 * float(min_clearance)
    score_components["clearance"] = clearance_component

    # Clearance trend component (positive = moving away)
    trend_component = 5.0 * float(clearance_trend)
    score_components["clearance_trend"] = trend_component

    # TTC component (higher is better, but cap at 10s)
    ttc_capped = min(float(min_ttc), 10.0)
    ttc_component = 2.0 * ttc_capped
    score_components["ttc"] = ttc_component

    # Moving toward entity penalty
    moving_toward_penalty = 0.0
    if moving_toward_entity:
        moving_toward_penalty = 5.0
    score_components["moving_toward_penalty"] = -moving_toward_penalty

    # Reverse penalty (small - reverse is generally safe but slow)
    reverse_penalty = 0.0
    if "reverse" in candidate.name:
        reverse_penalty = 0.5
    score_components["reverse_penalty"] = -reverse_penalty

    # Turn penalty (small)
    turn_penalty = 0.0
    if "turn" in candidate.name:
        turn_penalty = 0.3
    score_components["turn_penalty"] = -turn_penalty

    # Smoothness penalty (control change from nominal)
    dv = abs(float(candidate.linear_velocity)
             - float(nominal_action.get("linear_velocity", 0.0)))
    do = abs(float(candidate.angular_velocity)
             - float(nominal_action.get("angular_velocity", 0.0)))
    smoothness_penalty = 1.0 * (dv / max_linear_velocity + do / 1.2)
    score_components["smoothness_penalty"] = -smoothness_penalty

    # TTC threshold hard penalty: if TTC < threshold, penalize (even for stop)
    ttc_threshold_penalty = 0.0
    v = float(candidate.linear_velocity)
    if min_ttc < ttc_threshold:
        if v > 0.1:
            # Heavy penalty for forward motion when TTC is low
            ttc_threshold_penalty = 8.0 * (ttc_threshold - min_ttc) / ttc_threshold
        elif candidate.name == "stop":
            # Medium penalty for stop when TTC is low (entity will hit stopped robot)
            ttc_threshold_penalty = 5.0 * (ttc_threshold - min_ttc) / ttc_threshold
    score_components["ttc_threshold_penalty"] = -ttc_threshold_penalty

    total_score = (clearance_component + trend_component + ttc_component
                   - moving_toward_penalty - reverse_penalty - turn_penalty
                   - smoothness_penalty - ttc_threshold_penalty)

    score_components["total"] = float(total_score)
    return float(total_score), score_components


# ── Phase 3C: Encounter-aware scoring ───────────────────────────────────────────

def compute_backup_score_encounter_aware(
    candidate,
    min_clearance: float,
    initial_clearance: float,
    final_clearance: float,
    clearance_trend: float,
    min_ttc: float,
    moving_toward_entity: bool,
    nominal_action: Dict[str, Any],
    encounter_type: str,
    previous_backup_name: Optional[str],
    goal_direction: Optional[np.ndarray],
    max_linear_velocity: float = 0.8,
    ttc_threshold: float = 2.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Phase 3C: Encounter-type-aware and task-aware backup scoring.

    Uses encounter-specific weights and adds task progress terms.
    """
    from safedyn.safety.encounter_classifier import get_encounter_scoring_weights

    weights = get_encounter_scoring_weights(encounter_type)
    score_components = {}

    # Clearance component (encounter-weighted)
    clearance_component = weights.get("clearance_weight", 10.0) * float(min_clearance)
    score_components["clearance"] = clearance_component

    # Clearance trend component
    trend_weight = weights.get("trend_weight", 5.0)
    trend_component = trend_weight * float(clearance_trend)
    score_components["clearance_trend"] = trend_component

    # TTC component
    ttc_weight = weights.get("ttc_weight", 2.0)
    ttc_capped = min(float(min_ttc), 10.0)
    ttc_component = ttc_weight * ttc_capped
    score_components["ttc"] = ttc_component

    # Encounter-specific penalties and bonuses
    encounter_penalty = 0.0
    v = float(candidate.linear_velocity)

    # Forward penalty (encounter-specific)
    if v > 0.1:
        if moving_toward_entity:
            encounter_penalty += weights.get("forward_penalty", 2.0)
        else:
            encounter_penalty += weights.get("forward_penalty", 2.0) * 0.3

    # Reverse bonus/penalty
    if "reverse" in candidate.name:
        if "reverse_bonus" in weights:
            encounter_penalty -= weights["reverse_bonus"]  # negative = bonus
        if "reverse_penalty" in weights:
            encounter_penalty += weights["reverse_penalty"]

    # Turn bonus/penalty
    if "turn" in candidate.name:
        if "turn_bonus" in weights:
            encounter_penalty -= weights["turn_bonus"]
        if "turn_left_bonus" in weights and "left" in candidate.name:
            encounter_penalty -= weights["turn_left_bonus"]
        if "turn_right_bonus" in weights and "right" in candidate.name:
            encounter_penalty -= weights["turn_right_bonus"]
        if "turn_left_penalty" in weights and "left" in candidate.name:
            encounter_penalty += weights["turn_left_penalty"]
        if "turn_right_penalty" in weights and "right" in candidate.name:
            encounter_penalty += weights["turn_right_penalty"]

    # Stop bonus/penalty
    if candidate.name == "stop":
        if "stop_bonus" in weights:
            encounter_penalty -= weights["stop_bonus"]
        if "stop_penalty" in weights:
            encounter_penalty += weights["stop_penalty"]

    score_components["encounter_penalty"] = -encounter_penalty

    # Side consistency bonus (avoid left/right oscillation)
    side_consistency_bonus = 0.0
    if previous_backup_name and "side_consistency_weight" in weights:
        prev_left = "left" in previous_backup_name
        curr_left = "left" in candidate.name
        prev_right = "right" in previous_backup_name
        curr_right = "right" in candidate.name

        if (prev_left and curr_left) or (prev_right and curr_right):
            side_consistency_bonus = weights["side_consistency_weight"]
        elif (prev_left and curr_right) or (prev_right and curr_left):
            side_consistency_bonus = -weights["side_consistency_weight"]

    score_components["side_consistency"] = side_consistency_bonus

    # Task progress component (if goal direction provided)
    progress_component = 0.0
    if goal_direction is not None and v > 0.1:
        # Compute alignment with goal
        robot_yaw = np.arctan2(goal_direction[0], -goal_direction[1])
        action_yaw = robot_yaw + candidate.angular_velocity * 0.5  # approximate

        # Forward progress
        if "reverse" not in candidate.name:
            progress_component = 0.5 * v / max_linear_velocity

    score_components["progress"] = progress_component

    # Smoothness penalty
    dv = abs(v - float(nominal_action.get("linear_velocity", 0.0)))
    do = abs(float(candidate.angular_velocity) - float(nominal_action.get("angular_velocity", 0.0)))
    smoothness_penalty = 1.0 * (dv / max_linear_velocity + do / 1.2)
    score_components["smoothness"] = -smoothness_penalty

    # TTC threshold penalty
    ttc_threshold_penalty = 0.0
    if min_ttc < ttc_threshold:
        if v > 0.1:
            ttc_threshold_penalty = 8.0 * (ttc_threshold - min_ttc) / ttc_threshold
        elif candidate.name == "stop":
            ttc_threshold_penalty = 5.0 * (ttc_threshold - min_ttc) / ttc_threshold

    score_components["ttc_threshold"] = -ttc_threshold_penalty

    total_score = (clearance_component + trend_component + ttc_component
                   + progress_component + side_consistency_bonus
                   - encounter_penalty - smoothness_penalty - ttc_threshold_penalty)

    score_components["total"] = float(total_score)
    return float(total_score), score_components

def is_moving_toward_entity(
    robot_action: Dict[str, Any],
    robot_yaw: float,
    robot_pos: np.ndarray,
    entity_pos: np.ndarray,
    entity_velocity: np.ndarray,
) -> bool:
    """
    Check if robot action is moving toward the entity (head-on or closing scenario).

    Returns True if:
      - Robot has forward velocity (v > 0)
      - Entity is in front of robot (within ±90 degrees)
      - Relative motion is closing (distance decreasing)
    """
    v = float(robot_action.get("linear_velocity", 0.0))
    if v <= 0.1:
        return False  # Not moving forward significantly

    # Vector from robot to entity
    to_entity = np.asarray(entity_pos) - np.asarray(robot_pos)
    distance = np.linalg.norm(to_entity)
    if distance < 1e-6:
        return False

    # Angle to entity in robot frame
    # yaw=0 means facing -z, so angle is measured from -z axis
    angle_to_entity = np.arctan2(to_entity[0], -to_entity[1])  # x-z-yaw frame
    angle_diff = angle_to_entity - robot_yaw
    # Normalize to [-pi, pi]
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    # Entity is "in front" if within ±90 degrees
    entity_in_front = abs(angle_diff) < np.pi / 2

    if not entity_in_front:
        return False

    # Check if distance is decreasing (closing)
    # Project velocities onto line connecting robot and entity
    direction = to_entity / distance  # unit vector from robot to entity

    robot_vel = np.array([v * np.sin(robot_yaw), -v * np.cos(robot_yaw)])

    # Rate of change of distance = dot(entity_vel - robot_vel, direction)
    # Negative means closing (distance decreasing)
    distance_rate = np.dot(np.asarray(entity_velocity) - robot_vel, direction)

    # Closing if distance_rate < 0 (distance decreasing)
    return distance_rate < -0.1  # threshold to avoid noise


def evaluate_candidate(
    candidate,
    robot_start_pos,
    robot_start_yaw: float,
    nominal_action: Dict[str, Any],
    entity_start_pos,
    entity_velocity,
    cert_tube: List,
    robot_radius: float,
    dt: float,
    horizon_steps: int,
    max_linear_velocity: float = 0.8,
    encounter_type: str = "crossing",
    previous_backup_name: Optional[str] = None,
    goal_direction: Optional[np.ndarray] = None,
) -> BackupResult:
    """
    Phase 3C: Evaluate one backup candidate with encounter-aware and task-aware scoring.

    Returns BackupResult with is_feasible, min_clearance, score, reason, and debug fields.
    """
    action_dict = {"linear_velocity": float(candidate.linear_velocity),
                   "angular_velocity": float(candidate.angular_velocity)}

    # Compute clearance profile
    (min_cl, initial_cl, final_cl, min_ttc, min_closing_speed, min_d,
     first_violation, reason, constraining_step) = compute_rollout_clearance_profile(
        robot_start_pos, robot_start_yaw, action_dict,
        entity_start_pos, entity_velocity,
        cert_tube, robot_radius, dt, horizon_steps, max_linear_velocity,
    )

    clearance_trend = final_cl - initial_cl

    # Check if moving toward entity
    moving_toward = is_moving_toward_entity(
        action_dict, robot_start_yaw,
        np.asarray(robot_start_pos), np.asarray(entity_start_pos), np.asarray(entity_velocity)
    )

    # Compute score with encounter-aware scoring
    score, score_components = compute_backup_score_encounter_aware(
        candidate, min_cl, initial_cl, final_cl, clearance_trend, min_ttc,
        moving_toward, nominal_action, encounter_type, previous_backup_name,
        goal_direction, max_linear_velocity,
    )

    is_feasible = float(min_cl) > 0.0

    # Add specific reason for infeasibility
    if not is_feasible:
        if candidate.name == "stop":
            if first_violation >= 0:
                reason = f"stop_blocked_by_future_tube_at_step_{first_violation}"
            else:
                reason = "stop_blocked_by_future_tube"
        elif not reason:
            reason = "clearance_negative"

    # Forward toward entity penalty reason
    forward_toward_score = 0.0
    if moving_toward and candidate.linear_velocity > 0.1:
        forward_toward_score = score_components.get("moving_toward_penalty", 0.0)

    return BackupResult(
        name=str(candidate.name),
        is_feasible=is_feasible,
        min_clearance=float(min_cl),
        score=float(score),
        rollout_min_distance=float(min_d),
        goal_progress=score_components.get("progress", 0.0),
        control_change=0.0,
        first_violation_step=first_violation,
        reason=reason,
        future_tube_step_used=constraining_step,
        initial_clearance=float(initial_cl),
        final_clearance=float(final_cl),
        clearance_trend=float(clearance_trend),
        min_ttc=float(min_ttc),
        moving_toward_entity=moving_toward,
        forward_toward_entity_score=float(forward_toward_score),
    )


# ── Top-level: evaluate all candidates ─────────────────────────────────────────

def evaluate_all_candidates(
    candidates: List,
    robot_start_pos,
    robot_start_yaw: float,
    nominal_action: Dict[str, Any],
    entity_start_pos,
    entity_velocity,
    cert_tube: List,
    robot_radius: float,
    dt: float,
    horizon_steps: int,
    max_linear_velocity: float = 0.8,
) -> Tuple[List[BackupResult], List[BackupResult]]:
    """
    Evaluate all candidates.

    Returns: (feasible_results, all_results)
    """
    all_results = []
    feasible_results = []

    for c in candidates:
        result = evaluate_candidate(
            c, robot_start_pos, robot_start_yaw, nominal_action,
            entity_start_pos, entity_velocity, cert_tube,
            robot_radius, dt, horizon_steps, max_linear_velocity,
        )
        all_results.append(result)
        if result.is_feasible:
            feasible_results.append(result)

    return feasible_results, all_results


def find_best_safe_backup(
    feasible_results: List[BackupResult],
    prefer_non_stop: bool = True,
) -> Tuple[Optional[BackupResult], str, List[BackupResult]]:
    """
    Select the best safe backup candidate.

    Args:
        feasible_results: List of feasible backup candidates
        prefer_non_stop: If True, prefer non-stop backups when available

    Returns (best_result, status, all_feasible):
      status = "safe_candidate_found"  → best_result is valid
      status = "no_safe_candidate"     → best_result is None, emergency_stop
      all_feasible = list of all feasible candidates for logging
    """
    if not feasible_results:
        return None, "no_safe_candidate", []

    # Sort by score descending
    feasible_results.sort(key=lambda r: r.score, reverse=True)

    # If prefer_non_stop, try to find best non-stop candidate first
    if prefer_non_stop:
        non_stop_feasible = [r for r in feasible_results if r.name != "stop"]
        if non_stop_feasible:
            # Sort non-stop by score
            non_stop_feasible.sort(key=lambda r: r.score, reverse=True)
            best = non_stop_feasible[0]
            if best.min_clearance > 0.0:
                return best, "safe_candidate_found_non_stop", feasible_results

    # Otherwise, use best overall (may be stop)
    best = feasible_results[0]
    if best.min_clearance > 0.0:
        if best.name == "stop":
            return best, "safe_candidate_found_stop", feasible_results
        return best, "safe_candidate_found", feasible_results

    return None, "no_safe_candidate", feasible_results
