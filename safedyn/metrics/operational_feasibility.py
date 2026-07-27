"""evaluation/operational_feasibility.py

Method-independent operational-bounds feasibility sweep.

Given a scenario (robot start, goal, dynamic entities), robot dynamics
bounds, and a horizon, this module enumerates a small deterministic
trajectory family and reports whether at least one trajectory keeps
the robot clear of every entity's constant-velocity extrapolation and
makes forward progress. It is used ONLY as an offline scenario
calibration: is this scenario physically within the robot's operational
envelope? It is NOT an online policy and never sees method outcomes.

The trajectory family is declared in the manifest (`feasibility.trajectory_family`)
and mirrored here so unit tests are self-contained.

Coordinate system: x-z-yaw (Habitat convention). Robot forward is -z
when yaw = 0. Positive omega turns the robot toward +x (the project's
"left" convention — see safety/action_candidates.py where slow_left has
omega = +0.8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryCandidate:
    """One trajectory in the feasibility family."""
    name: str
    phase_defs: List[Tuple[float, float, int]]  # list of (v, omega, ticks)


@dataclass
class FeasibilityResult:
    """Method-independent scenario feasibility result."""
    scenario_feasible: bool
    feasible_trajectory_exists: bool
    feasible_mode: Optional[str]
    feasible_min_clearance: float
    feasible_progress: float
    reason_if_infeasible: str
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # -- Per-mode extensions (spec section B) ---------------------------
    # These are set by ``sweep_feasibility`` in addition to the legacy
    # aggregate fields above so downstream code that only reads
    # ``scenario_feasible`` continues to work.
    forward_feasible: bool = False
    stop_feasible: bool = False
    side_left_feasible: bool = False
    side_right_feasible: bool = False
    any_side_feasible: bool = False
    best_feasible_mode: Optional[str] = None
    best_side_mode: Optional[str] = None
    forward_min_clearance: float = 0.0
    side_min_clearance: float = 0.0
    stop_deadlock_risk: bool = False
    forward_collision_or_nearmiss: bool = False
    side_progress: float = 0.0
    side_gap_certifiable: bool = False
    include_in_side_commit_main: bool = False


# --------------------------------------------------------------------------
# Default trajectory family (matches the manifest's feasibility block).
# --------------------------------------------------------------------------

def build_default_family(v_max: float, omega_max: float, horizon_ticks: int) -> List[TrajectoryCandidate]:
    """Return the discrete trajectory family used by the sweep.

    Each candidate is a sequence of (v, omega, ticks). Ticks in the
    sequence sum to `horizon_ticks` — no candidate can extend past the
    total horizon.
    """
    v_slow = 0.4 * v_max
    v_arc = 0.55 * v_max
    v_yield = 0.30 * v_max
    v_rev = -0.25 * v_max
    o_arc = 0.75 * omega_max
    half = horizon_ticks // 2
    other = horizon_ticks - half
    return [
        TrajectoryCandidate("forward",             [(v_max,   0.0,     horizon_ticks)]),
        TrajectoryCandidate("stop",                [(0.0,     0.0,     horizon_ticks)]),
        TrajectoryCandidate("slow_forward",        [(v_slow,  0.0,     horizon_ticks)]),
        TrajectoryCandidate("left_arc",            [(v_arc,   +o_arc,  horizon_ticks)]),
        TrajectoryCandidate("right_arc",           [(v_arc,   -o_arc,  horizon_ticks)]),
        TrajectoryCandidate("yield_left",          [(v_yield, +o_arc,  horizon_ticks)]),
        TrajectoryCandidate("yield_right",         [(v_yield, -o_arc,  horizon_ticks)]),
        TrajectoryCandidate("stop_then_left",      [(0.0, 0.0, half), (v_arc, +o_arc, other)]),
        TrajectoryCandidate("stop_then_right",     [(0.0, 0.0, half), (v_arc, -o_arc, other)]),
        TrajectoryCandidate("slow_reverse_then_left",
                            [(v_rev, +0.5 * omega_max, half),
                             (v_arc, +o_arc, other)]),
        TrajectoryCandidate("slow_reverse_then_right",
                            [(v_rev, -0.5 * omega_max, half),
                             (v_arc, -o_arc, other)]),
    ]


# Mode-classification: which side each candidate belongs to.
_LEFT_CANDIDATES = {"left_arc", "yield_left", "stop_then_left", "slow_reverse_then_left"}
_RIGHT_CANDIDATES = {"right_arc", "yield_right", "stop_then_right", "slow_reverse_then_right"}
_SIDE_CANDIDATES = _LEFT_CANDIDATES | _RIGHT_CANDIDATES


# --------------------------------------------------------------------------
# Rollout helpers.
# --------------------------------------------------------------------------

def _simulate(
    start_pos: np.ndarray,
    start_yaw: float,
    phases: List[Tuple[float, float, int]],
    dt: float,
) -> List[np.ndarray]:
    pos = np.asarray(start_pos, dtype=np.float64).copy()
    yaw = float(start_yaw)
    positions = [pos.copy()]
    for v, omega, ticks in phases:
        for _ in range(int(ticks)):
            pos = pos + np.array([v * np.sin(yaw), -v * np.cos(yaw)]) * dt
            yaw = yaw + omega * dt
            positions.append(pos.copy())
    return positions


def _entity_positions(entity, horizon_ticks: int, dt: float) -> List[np.ndarray]:
    if hasattr(entity, "position") and callable(entity.position):
        p0 = np.asarray(entity.position(), dtype=np.float64)
        v0 = np.asarray(entity.velocity(), dtype=np.float64)
    elif isinstance(entity, dict):
        p0 = np.asarray(entity.get("position", [0.0, 0.0]), dtype=np.float64)
        v0 = np.asarray(entity.get("velocity", [0.0, 0.0]), dtype=np.float64)
    else:
        p0 = np.asarray(getattr(entity, "position", [0.0, 0.0]), dtype=np.float64)
        v0 = np.asarray(getattr(entity, "velocity", [0.0, 0.0]), dtype=np.float64)
    return [p0 + v0 * (dt * float(i)) for i in range(horizon_ticks + 1)]


def _entity_radius(entity) -> float:
    if isinstance(entity, dict):
        return float(entity.get("radius", 0.25))
    return float(getattr(entity, "radius", 0.25))


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------

def sweep_feasibility(
    scenario: Dict[str, Any],
    robot: Dict[str, Any],
    horizon_ticks: int = 60,
    safety_margin_m: float = 0.10,
    progress_min_m: float = 0.30,
    family: Optional[List[TrajectoryCandidate]] = None,
) -> FeasibilityResult:
    """Enumerate the trajectory family and return the best-scoring
    feasible candidate.

    A candidate is *feasible* if:
      - `min clearance >= safety_margin_m` over the entire rollout
      - forward goal-projected progress >= `progress_min_m`

    The score used to pick a winner among feasible candidates is
    `progress + 0.25 * min_clearance`. This function is used only for
    scenario calibration and NEVER sees method outcomes.
    """
    dt = float(robot.get("dt", 0.05))
    v_max = float(robot.get("v_max", robot.get("max_linear_velocity", 0.8)))
    omega_max = float(robot.get("omega_max", robot.get("max_angular_velocity", 1.2)))
    robot_radius = float(robot.get("robot_radius", robot.get("radius", 0.25)))

    start = list(scenario.get("robot_start", [0.0, 0.0, 0.0]))
    goal = list(scenario.get("goal", [0.0, -10.0]))
    entities = list(scenario.get("entities", []) or [])
    start_pos = np.array([float(start[0]), float(start[1])], dtype=np.float64)
    start_yaw = float(start[2]) if len(start) > 2 else 0.0
    goal_vec = np.array(goal, dtype=np.float64) - start_pos
    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm < 1e-6:
        goal_unit = np.array([0.0, -1.0])
    else:
        goal_unit = goal_vec / goal_norm

    fam = family or build_default_family(v_max, omega_max, horizon_ticks)

    # Precompute entity trajectories.
    entity_paths = [_entity_positions(e, horizon_ticks, dt) for e in entities]
    entity_radii = [_entity_radius(e) for e in entities]

    all_candidates: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_name: Optional[str] = None
    best_min_clear = -float("inf")
    best_progress = -float("inf")
    best_feasible = False

    for cand in fam:
        # Cap phases so total ticks match horizon (defensive).
        total = sum(int(t) for _, _, t in cand.phase_defs)
        phases = cand.phase_defs
        if total != horizon_ticks:
            # Pad/truncate the last phase.
            if phases and total < horizon_ticks:
                v, w, t = phases[-1]
                phases = list(phases[:-1]) + [(v, w, t + (horizon_ticks - total))]
            elif phases and total > horizon_ticks:
                v, w, t = phases[-1]
                phases = list(phases[:-1]) + [(v, w, max(0, t - (total - horizon_ticks)))]
        positions = _simulate(start_pos, start_yaw, phases, dt)

        # Min clearance over the rollout.
        min_clear = float("inf")
        for i, rp in enumerate(positions):
            if i == 0:
                continue
            for ep, er in zip(entity_paths, entity_radii):
                if i >= len(ep):
                    continue
                d = float(np.linalg.norm(rp - ep[i]))
                clear = d - float(robot_radius) - float(er)
                if clear < min_clear:
                    min_clear = clear
        if min_clear == float("inf"):
            min_clear = 10.0

        progress = float(np.dot(positions[-1] - start_pos, goal_unit))
        feasible = (min_clear >= safety_margin_m) and (progress >= progress_min_m)
        score = progress + 0.25 * max(0.0, min_clear)

        all_candidates.append({
            "name": cand.name,
            "min_clearance_m": float(min_clear),
            "progress_m": float(progress),
            "feasible": bool(feasible),
            "score": float(score),
        })

        if feasible and score > best_score:
            best_score = score
            best_name = cand.name
            best_min_clear = min_clear
            best_progress = progress
            best_feasible = True

    if best_feasible and best_name is not None:
        _pm = _compute_per_mode_fields(
            all_candidates=all_candidates,
            safety_margin_m=safety_margin_m,
            progress_min_m=progress_min_m,
        )
        return FeasibilityResult(
            scenario_feasible=True,
            feasible_trajectory_exists=True,
            feasible_mode=best_name,
            feasible_min_clearance=float(best_min_clear),
            feasible_progress=float(best_progress),
            reason_if_infeasible="",
            all_candidates=all_candidates,
            **_pm,
        )

    # No candidate was fully feasible. Report why using the best-clearance
    # candidate as a diagnostic reference.
    best_clear = max((c["min_clearance_m"] for c in all_candidates), default=-1.0)
    reasons = []
    if best_clear < safety_margin_m:
        reasons.append(
            f"no_candidate_clears_safety_margin(best={best_clear:.3f}m "
            f"< {safety_margin_m:.3f}m)"
        )
    best_prog = max((c["progress_m"] for c in all_candidates), default=0.0)
    if best_prog < progress_min_m:
        reasons.append(
            f"no_candidate_makes_progress(best={best_prog:.3f}m "
            f"< {progress_min_m:.3f}m)"
        )
    # If the per-mode aggregation shows no side recovery, add that
    # diagnostic reason so the caller can distinguish
    # "beyond_operational_bounds" from "forward-only feasible".
    _pm = _compute_per_mode_fields(
        all_candidates=all_candidates,
        safety_margin_m=safety_margin_m,
        progress_min_m=progress_min_m,
    )
    if not _pm["any_side_feasible"]:
        reasons.append("no_side_recovery_path")
    if not reasons:
        reasons.append("no_feasible_trajectory_found")
    return FeasibilityResult(
        scenario_feasible=False,
        feasible_trajectory_exists=False,
        feasible_mode=None,
        feasible_min_clearance=float(best_clear if best_clear != -1.0 else 0.0),
        feasible_progress=float(best_prog),
        reason_if_infeasible=";".join(reasons),
        all_candidates=all_candidates,
        **_pm,
    )


def _compute_per_mode_fields(
    all_candidates: List[Dict[str, Any]],
    safety_margin_m: float,
    progress_min_m: float,
) -> Dict[str, Any]:
    """Aggregate per-candidate results into the per-mode feasibility
    fields declared in ``FeasibilityResult``.
    """
    by_name = {c["name"]: c for c in all_candidates}

    def _get(name: str, key: str, default: float) -> float:
        return float(by_name.get(name, {}).get(key, default))

    forward = by_name.get("forward", {})
    stop = by_name.get("stop", {})

    forward_feasible = bool(forward.get("feasible", False))
    forward_min_clear = float(forward.get("min_clearance_m", 0.0))
    forward_progress = float(forward.get("progress_m", 0.0))
    forward_collision_or_nearmiss = (
        forward_min_clear < safety_margin_m
    )

    # A stop is "surviving-feasible" if it stays clear; but by
    # definition it makes zero progress, so we tag deadlock_risk.
    stop_min_clear = float(stop.get("min_clearance_m", 0.0))
    stop_progress = float(stop.get("progress_m", 0.0))
    stop_feasible = stop_min_clear >= safety_margin_m
    stop_deadlock_risk = bool(stop_feasible and stop_progress < progress_min_m)

    # Side-mode aggregation.
    left_feasible_cands = [
        c for c in all_candidates
        if c["name"] in _LEFT_CANDIDATES and c["feasible"]
    ]
    right_feasible_cands = [
        c for c in all_candidates
        if c["name"] in _RIGHT_CANDIDATES and c["feasible"]
    ]
    side_left_feasible = len(left_feasible_cands) > 0
    side_right_feasible = len(right_feasible_cands) > 0
    any_side_feasible = side_left_feasible or side_right_feasible

    side_all_feasible = left_feasible_cands + right_feasible_cands
    if side_all_feasible:
        best_side = max(side_all_feasible, key=lambda c: c["score"])
        best_side_mode = str(best_side["name"])
        side_min_clearance = float(best_side["min_clearance_m"])
        side_progress = float(best_side["progress_m"])
    else:
        best_side_mode = None
        # Diagnostic: even if no side candidate is feasible, report
        # the *best* side clearance/progress for the audit sheet.
        side_all = [
            c for c in all_candidates
            if c["name"] in _SIDE_CANDIDATES
        ]
        if side_all:
            best_side_diag = max(side_all, key=lambda c: c["score"])
            side_min_clearance = float(best_side_diag["min_clearance_m"])
            side_progress = float(best_side_diag["progress_m"])
        else:
            side_min_clearance = 0.0
            side_progress = 0.0

    # Best mode across all candidates by score, feasible-only.
    feasible_all = [c for c in all_candidates if c["feasible"]]
    if feasible_all:
        best_feasible_mode = str(max(feasible_all, key=lambda c: c["score"])["name"])
    else:
        best_feasible_mode = None

    side_gap_certifiable = bool(
        any_side_feasible and side_min_clearance >= safety_margin_m
    )
    include_in_side_commit_main = bool(
        (not forward_feasible)
        and any_side_feasible
        and side_min_clearance >= safety_margin_m
        and side_progress > 0.0
    )

    return {
        "forward_feasible": bool(forward_feasible),
        "stop_feasible": bool(stop_feasible),
        "side_left_feasible": bool(side_left_feasible),
        "side_right_feasible": bool(side_right_feasible),
        "any_side_feasible": bool(any_side_feasible),
        "best_feasible_mode": best_feasible_mode,
        "best_side_mode": best_side_mode,
        "forward_min_clearance": float(forward_min_clear),
        "side_min_clearance": float(side_min_clearance),
        "stop_deadlock_risk": bool(stop_deadlock_risk),
        "forward_collision_or_nearmiss": bool(forward_collision_or_nearmiss),
        "side_progress": float(side_progress),
        "side_gap_certifiable": bool(side_gap_certifiable),
        "include_in_side_commit_main": bool(include_in_side_commit_main),
    }


__all__ = [
    "TrajectoryCandidate",
    "FeasibilityResult",
    "build_default_family",
    "sweep_feasibility",
]
