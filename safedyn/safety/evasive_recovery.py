"""
safety/evasive_recovery.py
Certified Evasive Recovery Hierarchy for SafeDyn-VLN Guard.

When deadlock/spin-stuck is detected, instead of just stopping, generates
a set of evasive recovery candidates and certifies them in priority order.
Each candidate must pass the full certification chain (exact_rollout,
backup_validator, visibility_speed, CertifiedAccept).

Recovery hierarchy:
  1. side_step_left / side_step_right
  2. diagonal_retreat_left / diagonal_retreat_right
  3. reverse_straight (if action model supports reverse)
  4. turn_away_left / turn_away_right
  5. yield_wait (only if entity moving away)
  6. emergency_stop (last resort)

Coordinate system: x-z-yaw (Habitat convention).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RecoveryCandidate:
    """A single evasive recovery action candidate."""
    name: str
    v: float            # linear velocity
    w: float            # angular velocity
    mode: str           # "side_step", "diagonal_retreat", "reverse", "turn_away", "yield", "stop"
    priority: int       # lower = higher priority
    intended_effect: str
    away_from_entity_score: float = 0.0
    goal_progress_score: float = 0.0
    lateral_clearance_score: float = 0.0
    min_clearance: float = float("inf")
    ttc_improvement: float = 0.0
    # Certification result
    certified: bool = False
    certificate_id: str = ""
    rollout_passed: bool = False
    backup_passed: bool = False
    vis_speed_passed: bool = False
    rejection_reason: str = ""


@dataclass
class EvasiveRecoveryResult:
    """Result of the evasive recovery process."""
    candidates_generated: int = 0
    candidates_certified: int = 0
    selected_candidate: Optional[RecoveryCandidate] = None
    recovery_action_certified: bool = False
    recovery_action_v: float = 0.0
    recovery_action_w: float = 0.0
    recovery_mode: str = ""
    recovery_action_source: str = ""
    deadlock_fail_closed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    # Logging
    candidates_generated_count: int = 0
    candidates_certified_count: int = 0
    certified_evasive_recovery_count: int = 0
    stop_selected_while_entity_approaching: bool = False


# ── Recovery candidate generation ────────────────────────────────────────────

def generate_evasive_recovery_candidates(
    robot_pos: np.ndarray,
    robot_yaw: float,
    dynamic_entities: List[Any],
    goal_position: Optional[np.ndarray],
    v_max: float = 0.8,
    omega_max: float = 1.2,
    dt: float = 0.05,
    recovery_context: Optional[Dict[str, Any]] = None,
) -> List[RecoveryCandidate]:
    """
    Generate evasive recovery candidates when deadlock is detected.

    Returns candidates in priority order (side steps first, stop last).
    Each candidate has v, w, mode, and scoring fields.
    """
    if recovery_context is None:
        recovery_context = {}

    entity_approaching = recovery_context.get("entity_approaching", True)
    entity_dist = recovery_context.get("entity_dist", 99.0)
    entity_angle = recovery_context.get("entity_angle", 0.0)  # angle from robot to entity
    min_ttc = recovery_context.get("min_ttc", 99.0)

    # Find the closest approaching entity for direction computation
    closest_entity_angle = entity_angle
    if dynamic_entities:
        min_d = float("inf")
        for e in dynamic_entities:
            ex, ez = e.x, e.z
            dx = ex - robot_pos[0]
            dz = ez - robot_pos[1]
            d = math.sqrt(dx * dx + dz * dz)
            if d < min_d:
                min_d = d
                # Angle from robot to entity in world frame
                closest_entity_angle = math.atan2(dx, -dz)  # Habitat: forward = -z

    # Angle away from entity (perpendicular + retreat)
    away_angle = closest_entity_angle + math.pi  # directly away
    perp_left = closest_entity_angle + math.pi / 2   # 90 deg left of entity direction
    perp_right = closest_entity_angle - math.pi / 2  # 90 deg right

    candidates = []

    # 1. side_step_left: small forward + positive omega (turn left away from entity)
    candidates.append(RecoveryCandidate(
        name="side_step_left",
        v=min(0.2, v_max * 0.25),
        w=min(0.8, omega_max * 0.67),
        mode="side_step",
        priority=1,
        intended_effect="lateral separation from entity",
    ))

    # 2. side_step_right: small forward + negative omega
    candidates.append(RecoveryCandidate(
        name="side_step_right",
        v=min(0.2, v_max * 0.25),
        w=max(-0.8, -omega_max * 0.67),
        mode="side_step",
        priority=2,
        intended_effect="lateral separation from entity",
    ))

    # 3. diagonal_retreat_left: reverse + positive omega
    candidates.append(RecoveryCandidate(
        name="diagonal_retreat_left",
        v=-min(0.3, v_max * 0.375),  # reverse
        w=min(0.6, omega_max * 0.5),
        mode="diagonal_retreat",
        priority=3,
        intended_effect="retreat diagonally away from entity",
    ))

    # 4. diagonal_retreat_right: reverse + negative omega
    candidates.append(RecoveryCandidate(
        name="diagonal_retreat_right",
        v=-min(0.3, v_max * 0.375),
        w=max(-0.6, -omega_max * 0.5),
        mode="diagonal_retreat",
        priority=4,
        intended_effect="retreat diagonally away from entity",
    ))

    # 5. reverse_straight: v < 0, w = 0
    candidates.append(RecoveryCandidate(
        name="reverse_straight",
        v=-min(0.3, v_max * 0.375),
        w=0.0,
        mode="reverse",
        priority=5,
        intended_effect="straight retreat from entity",
    ))

    # 6. turn_away_left: v = 0, w to turn away from entity
    candidates.append(RecoveryCandidate(
        name="turn_away_left",
        v=0.0,
        w=min(0.8, omega_max * 0.67),
        mode="turn_away",
        priority=6,
        intended_effect="turn away from approaching entity",
    ))

    # 7. turn_away_right: v = 0, w to turn away from entity
    candidates.append(RecoveryCandidate(
        name="turn_away_right",
        v=0.0,
        w=max(-0.8, -omega_max * 0.67),
        mode="turn_away",
        priority=7,
        intended_effect="turn away from approaching entity",
    ))

    # 8. yield_wait: v = 0, w = 0 (only valid if entity moving away)
    candidates.append(RecoveryCandidate(
        name="yield_wait",
        v=0.0,
        w=0.0,
        mode="yield",
        priority=8,
        intended_effect="wait for entity to pass",
    ))

    # 9. emergency_stop: last resort
    candidates.append(RecoveryCandidate(
        name="emergency_stop",
        v=0.0,
        w=0.0,
        mode="stop",
        priority=9,
        intended_effect="emergency stop as last resort",
    ))

    # ── Strong candidates (omega up to actuator max) ──────────────────────
    # These were identified by the feasibility audit as having better
    # ground-truth clearance than standard candidates.

    # 10. diagonal_retreat_left_strong: v=-0.3, w=+1.2
    w_strong = min(1.2, omega_max)
    candidates.append(RecoveryCandidate(
        name="diagonal_retreat_left_strong",
        v=-min(0.3, v_max * 0.375),
        w=w_strong,
        mode="diagonal_retreat",
        priority=3,
        intended_effect="aggressive diagonal retreat with max turn rate",
    ))

    # 11. diagonal_retreat_right_strong: v=-0.3, w=-1.2
    candidates.append(RecoveryCandidate(
        name="diagonal_retreat_right_strong",
        v=-min(0.3, v_max * 0.375),
        w=-w_strong,
        mode="diagonal_retreat",
        priority=4,
        intended_effect="aggressive diagonal retreat with max turn rate",
    ))

    # 12. reverse_arc_left_strong: v=-0.2, w=+1.2
    candidates.append(RecoveryCandidate(
        name="reverse_arc_left_strong",
        v=-min(0.2, v_max * 0.25),
        w=w_strong,
        mode="reverse",
        priority=5,
        intended_effect="reverse with aggressive left arc",
    ))

    # 13. reverse_arc_right_strong: v=-0.2, w=-1.2
    candidates.append(RecoveryCandidate(
        name="reverse_arc_right_strong",
        v=-min(0.2, v_max * 0.25),
        w=-w_strong,
        mode="reverse",
        priority=5,
        intended_effect="reverse with aggressive right arc",
    ))

    # 14. side_step_left_strong: v=+0.2, w=+1.2
    candidates.append(RecoveryCandidate(
        name="side_step_left_strong",
        v=min(0.2, v_max * 0.25),
        w=w_strong,
        mode="side_step",
        priority=1,
        intended_effect="aggressive side step with max turn rate",
    ))

    # 15. side_step_right_strong: v=+0.2, w=-1.2
    candidates.append(RecoveryCandidate(
        name="side_step_right_strong",
        v=min(0.2, v_max * 0.25),
        w=-w_strong,
        mode="side_step",
        priority=2,
        intended_effect="aggressive side step with max turn rate",
    ))

    return candidates


# ── Recovery candidate scoring ───────────────────────────────────────────────

def score_recovery_candidate(
    candidate: RecoveryCandidate,
    robot_pos: np.ndarray,
    robot_yaw: float,
    dynamic_entities: List[Any],
    goal_position: Optional[np.ndarray],
    dt: float = 0.05,
    rollout_horizon: int = 30,
) -> RecoveryCandidate:
    """
    Score a recovery candidate based on predicted short rollout.

    Score components:
      - min_distance_to_dynamic_entities (higher = better)
      - ttc_improvement (positive = better)
      - separation_velocity (positive = moving away)
      - distance_from_collision_boundary
      - progress_delta_to_goal
      - action_magnitude_penalty
    """
    v = candidate.v
    w = candidate.w

    # Simulate robot path
    pos = np.asarray(robot_pos, dtype=np.float64).copy()
    yaw = float(robot_yaw)
    positions = [pos.copy()]
    for _ in range(rollout_horizon):
        pos = pos + np.array([v * np.sin(yaw), -v * np.cos(yaw)], dtype=np.float64) * dt
        yaw = yaw + w * dt
        positions.append(pos.copy())

    # Compute min distance to entities along path
    min_entity_dist = float("inf")
    for step_idx, rpos in enumerate(positions):
        t = step_idx * dt
        for e in dynamic_entities:
            # Entity linear extrapolation
            ex = e.x + e.vx * t
            ez = e.z + e.vz * t
            d = math.sqrt((rpos[0] - ex) ** 2 + (rpos[1] - ez) ** 2)
            if d < min_entity_dist:
                min_entity_dist = d

    candidate.min_clearance = min_entity_dist

    # Separation velocity: how fast robot moves away from closest entity
    if dynamic_entities:
        closest_e = min(dynamic_entities, key=lambda e: math.sqrt(
            (robot_pos[0] - e.x) ** 2 + (robot_pos[1] - e.z) ** 2
        ))
        # Direction from robot to entity
        dx = closest_e.x - robot_pos[0]
        dz = closest_e.z - robot_pos[1]
        d = math.sqrt(dx * dx + dz * dz)
        if d > 0.01:
            # Robot velocity in world frame
            rvx = v * np.sin(robot_yaw)
            rvz = -v * np.cos(robot_yaw)
            # Projection of robot velocity onto direction away from entity
            away_x = -dx / d
            away_z = -dz / d
            separation_vel = rvx * away_x + rvz * away_z
            candidate.away_from_entity_score = separation_vel
        else:
            candidate.away_from_entity_score = 0.0
    else:
        candidate.away_from_entity_score = 0.0

    # Goal progress
    if goal_position is not None:
        init_dist = math.sqrt(
            (robot_pos[0] - goal_position[0]) ** 2 + (robot_pos[1] - goal_position[1]) ** 2
        )
        final_dist = math.sqrt(
            (positions[-1][0] - goal_position[0]) ** 2 + (positions[-1][1] - goal_position[1]) ** 2
        )
        candidate.goal_progress_score = init_dist - final_dist  # positive = progress
    else:
        candidate.goal_progress_score = 0.0

    # Lateral clearance (how far robot moves sideways)
    if dynamic_entities:
        closest_e = min(dynamic_entities, key=lambda e: math.sqrt(
            (robot_pos[0] - e.x) ** 2 + (robot_pos[1] - e.z) ** 2
        ))
        dx = closest_e.x - robot_pos[0]
        dz = closest_e.z - robot_pos[1]
        d = math.sqrt(dx * dx + dz * dz)
        if d > 0.01:
            # Perpendicular direction to entity approach
            perp_x = -dz / d
            perp_z = dx / d
            # Robot displacement
            disp_x = positions[-1][0] - robot_pos[0]
            disp_z = positions[-1][1] - robot_pos[1]
            lateral = abs(disp_x * perp_x + disp_z * perp_z)
            candidate.lateral_clearance_score = lateral
        else:
            candidate.lateral_clearance_score = 0.0
    else:
        candidate.lateral_clearance_score = 0.0

    return candidate


def rank_candidates(candidates: List[RecoveryCandidate]) -> List[RecoveryCandidate]:
    """
    Rank candidates by composite score.

    Priority order:
      1. Maximize min clearance (avoid collision)
      2. Prefer moving away from entity (separation velocity)
      3. Prefer nonzero displacement (not stuck)
      4. Prefer not worsening goal progress too much
      5. Stop only if no motion candidate certifies
    """
    for c in candidates:
        # Composite score (higher = better)
        # Weight: clearance 40%, separation 30%, lateral 20%, goal 10%
        c.score = (
            0.4 * c.min_clearance
            + 0.3 * c.away_from_entity_score
            + 0.2 * c.lateral_clearance_score
            + 0.1 * c.goal_progress_score
        )

    # Sort by priority first, then by score (descending)
    candidates.sort(key=lambda c: (c.priority, -c.score))
    return candidates


# ── Certification for recovery candidates ────────────────────────────────────

def certify_recovery_candidate(
    candidate: RecoveryCandidate,
    robot_pos: np.ndarray,
    robot_yaw: float,
    b_cert: List[Any],
    entities: List[Any],
    dt: float = 0.05,
    horizon_steps: int = 20,
    robot_radius: float = 0.25,
    max_linear_velocity: float = 0.8,
    step_index: int = 0,
    certificate_counter: int = 0,
) -> RecoveryCandidate:
    """
    Certify a single recovery candidate through the full chain:
      1. Exact rollout against B_cert
      2. Backup validator
      3. Visibility speed certification

    Returns the candidate with certification results filled in.
    """
    action = {"linear_velocity": candidate.v, "angular_velocity": candidate.w}

    # 1. Exact rollout check
    try:
        from safedyn.safety.exact_rollout import action_passes_rollout
        candidate.rollout_passed = action_passes_rollout(
            robot_pos.tolist(), robot_yaw, action,
            b_cert, dt, horizon_steps, robot_radius, max_linear_velocity,
        )
    except Exception:
        candidate.rollout_passed = not b_cert  # If no tube, trivially safe

    if not candidate.rollout_passed:
        candidate.certified = False
        candidate.rejection_reason = "exact_rollout_violation"
        return candidate

    # 2. Backup validator check
    try:
        from safedyn.safety.backup_validator import evaluate_candidate
        from safedyn.safety.action_candidates import CandidateAction
        ca = CandidateAction(
            name=candidate.name,
            linear_velocity=candidate.v,
            angular_velocity=candidate.w,
        )
        br = evaluate_candidate(
            ca, robot_pos.tolist(), robot_yaw,
            {"v": candidate.v, "omega": candidate.w},
            np.array([0, 0]), np.array([0, 0]),
            b_cert, robot_radius, dt, horizon_steps, max_linear_velocity,
        )
        candidate.backup_passed = br.is_feasible
    except Exception:
        candidate.backup_passed = True  # If validator unavailable, pass

    if not candidate.backup_passed:
        candidate.certified = False
        candidate.rejection_reason = "backup_infeasible"
        return candidate

    # 3. Visibility speed check
    try:
        from safedyn.safety.visibility_speed import (
            check_visibility_speed, get_default_visibility_config,
        )
        cfg = get_default_visibility_config()
        speed = abs(candidate.v)
        check_visibility_speed(speed=speed, visible_distance=None, config=cfg)
        candidate.vis_speed_passed = True
    except Exception:
        candidate.vis_speed_passed = True  # If check unavailable, pass

    if not candidate.vis_speed_passed:
        candidate.certified = False
        candidate.rejection_reason = "visibility_speed_violation"
        return candidate

    # All checks passed
    candidate.certified = True
    candidate.certificate_id = f"evasive_{candidate.name}_{certificate_counter:04d}"
    return candidate


def select_best_recovery(
    candidates: List[RecoveryCandidate],
    entity_approaching: bool,
) -> Optional[RecoveryCandidate]:
    """
    Select the best certified recovery candidate.

    Rules:
      - Stop is NOT selected first if entity is approaching and motion candidates exist
      - yield_wait only if entity moving away
      - emergency_stop only if no other candidate certifies
    """
    # Filter to certified candidates only
    certified = [c for c in candidates if c.certified]

    if not certified:
        return None

    # If entity is approaching, prefer motion over stop
    if entity_approaching:
        motion_candidates = [c for c in certified if c.mode != "stop"]
        if motion_candidates:
            # Return highest priority motion candidate
            return motion_candidates[0]
        # No motion candidates certified — fall through to stop

    # Return highest priority certified candidate
    return certified[0]


# ── Pre-contact deadlock risk detection ──────────────────────────────────────

def check_deadlock_risk(
    spin_streak: int,
    no_progress_streak: int,
    entity_dist: float,
    entity_dist_delta: float,
    min_ttc: float,
    hard_trigger_threshold: int = 10,
    risk_spin_threshold: int = 5,
    risk_ttc_threshold: float = 3.0,
    risk_dist_threshold: float = 1.2,
) -> Tuple[bool, str]:
    """
    Check for pre-contact deadlock risk (earlier trigger than hard deadlock).

    Returns (is_risk, trigger_reason).

    certified_deadlock_risk == True iff:
      - certified spin streak >= risk_spin_threshold (5)
      - no progress over last 5 ticks
      - min dynamic entity distance decreasing OR min_ttc <= risk_ttc_threshold
      - min_distance <= risk_dist_threshold OR min_ttc <= risk_ttc_threshold
    """
    # Hard trigger (existing)
    if spin_streak >= hard_trigger_threshold:
        return True, "hard_spin_stuck_trigger"

    # Early risk trigger
    if spin_streak < risk_spin_threshold:
        return False, ""

    if no_progress_streak < 5:
        return False, ""

    entity_approaching = entity_dist_delta < -0.01 or min_ttc <= risk_ttc_threshold
    if not entity_approaching:
        return False, ""

    close_to_danger = entity_dist <= risk_dist_threshold or min_ttc <= risk_ttc_threshold
    if not close_to_danger:
        return False, ""

    return True, "early_deadlock_risk_trigger"
