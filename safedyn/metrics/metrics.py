"""
evaluation/metrics.py
Phase 1: Full metrics for baselines and batch evaluation.
Coordinate system: x-z-yaw.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class MetricsTracker:
    """
    Tracks per-step and aggregate metrics for a SafeDyn-VLN Guard episode.

    Metric categories:
      Safety     → collision, near_miss, min_distance, collision_step, near_miss_count
      Task       → success, timeout, final_distance, path_length, goal_progress
      Anti-freezing → freeze_steps, freeze_rate, deadlock, average_speed
      Guard      → intervention_count, intervention_rate, fallback_count,
                    executed_stop_count
      Runtime    → episode_wall_time, mean_step_time, p95_step_time

    Phase 1 notes:
      - certified / QP / backup validator fields are placeholder (False / "not_applicable")
      - certification is not implemented yet
    """

    def __init__(self, goal_tolerance: float = 0.3):
        self.goal_tolerance = goal_tolerance
        self.reset()

    def reset(self) -> None:
        """Reset all counters for a new episode."""
        self.robot_state_goal_ref: Optional[np.ndarray] = None

        # ── Safety ───────────────────────────────────────────────────────────
        self.collision_steps: List[int] = []
        self.near_miss_steps: List[int] = []
        self.min_distances: List[float] = []

        # ── Task ───────────────────────────────────────────────────────────
        self.goal_reached = False
        self.timeout = False
        self.final_distance: Optional[float] = None
        self.robot_positions: List[np.ndarray] = []   # [x, z] per step
        self._initial_dist: Optional[float] = None
        self._path_length: float = 0.0
        self._step_speeds: List[float] = []

        # ── Anti-freezing ──────────────────────────────────────────────────
        self._frozen_steps = 0
        self._in_freeze = False

        # ── Guard behavior ────────────────────────────────────────────────
        self.intervention_steps: List[int] = []
        self.fallback_steps: List[int] = []
        self.stop_steps: List[int] = []
        self.sources: List[str] = []

        # ── Runtime ───────────────────────────────────────────────────────
        self._step_times_ms: List[float] = []

        # ── Tube tracking (Phase 2) ────────────────────────────────────────
        self._tube_violations_steps: List[int] = []
        self._tube_violations_total: int = 0

        # ── Rollout / Fallback (Phase 3A) ───────────────────────────────
        self._rollout_passed_steps: List[int] = []
        self._fallback_stop_steps: List[int] = []
        self._fallback_yield_left_steps: List[int] = []
        self._fallback_yield_right_steps: List[int] = []
        self._fallback_slow_reverse_steps: List[int] = []
        self._emergency_stop_steps: List[int] = []

        # ── CBF-QP (Phase 3B-1) ───────────────────────────────────────
        self._cbf_qp_intervention_steps: List[int] = []
        self._cbf_qp_solve_success_steps: List[int] = []
        self._cbf_qp_solve_failure_steps: List[int] = []
        self._filtered_action_steps: List[int] = []
        self._rollout_after_qp_pass_steps: List[int] = []

        # ── Backup Validator (Phase 3B-2) ───────────────────────────────
        self._backup_checked_steps: List[int] = []
        self._backup_feasible_steps: List[int] = []
        self._backup_infeasible_steps: List[int] = []
        self._safe_alternative_found_steps: List[int] = []
        self._backup_all_failed_steps: List[int] = []
        self._backup_min_clearances: List[float] = []
        self._stop_infeasible_steps: List[int] = []
        self._non_stop_backup_selected_steps: List[int] = []
        self._moving_toward_selected_steps: List[int] = []
        self._backup_clearance_trends: List[float] = []
        self._backup_min_ttcs: List[float] = []
        # Phase 3C: Encounter tracking
        self._encounter_types: Dict[str, int] = {}
        self._mode_switch_steps: List[int] = []
        self._side_switch_steps: List[int] = []
        self._previous_backup_policy: Optional[str] = None
        self._backup_policy_steps: Dict[str, List[int]] = {
            "stop": [],
            "slow_forward": [],
            "slow_left": [],
            "slow_right": [],
            "turn_left_in_place": [],
            "turn_right_in_place": [],
            "slow_reverse": [],
            "reverse_left": [],
            "reverse_right": [],
        }

        # ── Per-step log ────────────────────────────────────────────────
        self.steps: List[Dict[str, Any]] = []
        self.certifications: List[Dict[str, Any]] = []

    # ── Recording ────────────────────────────────────────────────────────────────

    def set_goal_reference(self, goal_pos: np.ndarray) -> None:
        """Set goal [x, z] for progress tracking."""
        self.robot_state_goal_ref = np.asarray(goal_pos, dtype=np.float64)

    def record_step(
        self,
        step: int,
        collision: bool,
        near_miss: bool,
        min_distance: float,
        certification: Dict[str, Any],
        u_vln: Dict[str, Any],
        u_exec: Dict[str, Any],
        robot_state,
        step_time_ms: float = 0.0,
        tube_violations: int = 0,
        num_tracks: int = 0,
        plan_radii: Optional[List[float]] = None,
        cert_radii: Optional[List[float]] = None,
    ) -> None:
        """Record all metrics for a single simulation step."""
        # ── Safety ───────────────────────────────────────────────────────
        self.min_distances.append(float(min_distance))
        if collision:
            self.collision_steps.append(step)
        if near_miss:
            self.near_miss_steps.append(step)

        # ── Guard behavior ───────────────────────────────────────────────
        self.sources.append(str(certification.get("source", "")))
        source = self.sources[-1]
        if certification.get("intervened", False):
            self.intervention_steps.append(step)
        if source in ("cautious", "yield", "emergency_stop", "single_step_shield_stop"):
            self.fallback_steps.append(step)
        if certification.get("tube_stop", False):
            self.fallback_steps.append(step)

        # ── Rollout / Fallback tracking (Phase 3A) ─────────────────────
        if certification.get("rollout_passed", False):
            self._rollout_passed_steps.append(step)

        fb_name = str(certification.get("fallback_type", ""))
        if fb_name:
            self.fallback_steps.append(step)
            if fb_name == "stop":
                self._fallback_stop_steps.append(step)
            elif fb_name == "yield_left":
                self._fallback_yield_left_steps.append(step)
            elif fb_name == "yield_right":
                self._fallback_yield_right_steps.append(step)
            elif fb_name == "slow_reverse":
                self._fallback_slow_reverse_steps.append(step)
        if source == "emergency_stop":
            self.fallback_steps.append(step)
            self._emergency_stop_steps.append(step)

        # ── CBF-QP tracking (Phase 3B-1) ─────────────────────────
        # qp_intervened=True means the filtered action differs from nominal
        qp_intervened = bool(certification.get("qp_intervened", False))
        if qp_intervened:
            self._cbf_qp_intervention_steps.append(step)
            self._filtered_action_steps.append(step)
        if source == "cbf_qp_filtered":
            # Filtered action passed rollout → QP/filter succeeded
            self._cbf_qp_solve_success_steps.append(step)
            self._rollout_after_qp_pass_steps.append(step)
        # Solve failure: we tried a filter, rollout still failed, fell back
        if qp_intervened and source.startswith("fallback_"):
            self._cbf_qp_solve_failure_steps.append(step)

        # ── Backup Validator tracking (Phase 3B-2) ─────────────────────
        if certification.get("backup_checked", False):
            self._backup_checked_steps.append(step)
        if certification.get("backup_feasible", False):
            self._backup_feasible_steps.append(step)
        else:
            if certification.get("backup_checked", False):
                self._backup_infeasible_steps.append(step)
        if certification.get("safe_alternative_found", False):
            self._safe_alternative_found_steps.append(step)
        if certification.get("emergency_stop", False) and certification.get("backup_checked", False):
            self._backup_all_failed_steps.append(step)

        # Track stop infeasibility and non-stop backup selection
        if certification.get("stop_infeasible", False):
            self._stop_infeasible_steps.append(step)
        if certification.get("non_stop_selected", False):
            self._non_stop_backup_selected_steps.append(step)
        if certification.get("moving_toward_selected", False):
            self._moving_toward_selected_steps.append(step)

        # Track clearance trend and TTC
        clearance_trend = certification.get("selected_clearance_trend")
        if clearance_trend is not None and clearance_trend != float("inf"):
            self._backup_clearance_trends.append(float(clearance_trend))
        min_ttc = certification.get("selected_min_ttc")
        if min_ttc is not None and min_ttc != float("inf"):
            self._backup_min_ttcs.append(float(min_ttc))

        # Track backup min clearance
        backup_min_cl = certification.get("backup_min_clearance")
        if backup_min_cl is not None and backup_min_cl != float("inf") and backup_min_cl != -float("inf"):
            self._backup_min_clearances.append(float(backup_min_cl))

        # Track selected backup policy
        selected_policy = certification.get("selected_backup_policy")
        if selected_policy and selected_policy in self._backup_policy_steps:
            self._backup_policy_steps[selected_policy].append(step)

        # Phase 3C: Track encounter types and mode switches
        encounter_type = certification.get("encounter_type")
        if encounter_type:
            if encounter_type not in self._encounter_types:
                self._encounter_types[encounter_type] = 0
            self._encounter_types[encounter_type] += 1

        # Track mode switches (backup policy changes)
        if selected_policy and self._previous_backup_policy:
            if selected_policy != self._previous_backup_policy:
                self._mode_switch_steps.append(step)
                # Check side switch
                prev_left = "left" in self._previous_backup_policy
                curr_right = "right" in selected_policy
                prev_right = "right" in self._previous_backup_policy
                curr_left = "left" in selected_policy
                if (prev_left and curr_right) or (prev_right and curr_left):
                    self._side_switch_steps.append(step)
        self._previous_backup_policy = selected_policy

        exec_v = float(u_exec.get("linear_velocity", 0.0))
        if abs(exec_v) < 0.01:
            self.stop_steps.append(step)

        # ── Anti-freezing ────────────────────────────────────────────────
        omega_exec = float(u_exec.get("angular_velocity", 0.0))
        if abs(exec_v) < 0.01 and abs(omega_exec) < 0.01:
            self._frozen_steps += 1
            self._in_freeze = True
        else:
            self._in_freeze = False
        self._step_speeds.append(float(exec_v))

        # ── Position tracking ────────────────────────────────────────────
        pos = np.array([float(robot_state.x), float(robot_state.z)], dtype=np.float64)
        self.robot_positions.append(pos)

        if len(self.robot_positions) >= 2:
            dpos = float(np.linalg.norm(pos - self.robot_positions[-2]))
            self._path_length += dpos

        # ── Goal detection ──────────────────────────────────────────────
        if self.robot_state_goal_ref is not None:
            dist = float(np.linalg.norm(pos - self.robot_state_goal_ref))
            if dist < self.goal_tolerance:
                self.goal_reached = True
            self.final_distance = dist

        # ── Runtime ────────────────────────────────────────────────────
        self._step_times_ms.append(float(step_time_ms))

        # ── Tube tracking (Phase 2) ──────────────────────────────────────
        if tube_violations > 0:
            self._tube_violations_steps.append(step)
        self._tube_violations_total += tube_violations

        # ── Certification log ───────────────────────────────────────────
        self.certifications.append(certification)
        self.steps.append({
            "step": step,
            "collision": collision,
            "near_miss": near_miss,
            "min_distance": float(min_distance),
            "source": source,
        })

    def record_initial_distance(self, dist: float) -> None:
        self._initial_dist = float(dist)

    # ── Summary ────────────────────────────────────────────────────────────────

    def compute_summary(self) -> Dict[str, Any]:
        """Compute all aggregate metrics for the episode."""
        n = max(len(self.steps), 1)

        # Safety
        collision = len(self.collision_steps) > 0
        near_miss = len(self.near_miss_steps) > 0
        collision_step = self.collision_steps[0] if collision else -1
        near_miss_count = len(self.near_miss_steps)
        min_dist_overall = float(min(self.min_distances)) if self.min_distances else float("inf")
        collision_rate = len(self.collision_steps) / n
        near_miss_rate = near_miss_count / n

        # Task
        success = 1.0 if self.goal_reached else 0.0
        timeout = (not self.goal_reached and len(self.steps) >= 1)
        final_dist = self.final_distance if self.final_distance is not None else -1.0
        initial_dist = self._initial_dist if self._initial_dist is not None else 0.0
        path_length = self._path_length
        optimal_path = max(initial_dist, 1e-6)
        spl = success * (optimal_path / max(path_length, 1e-6)) if success else 0.0

        # Anti-freezing
        freeze_rate = self._frozen_steps / n
        deadlock = self._in_freeze and not self.goal_reached
        avg_speed = float(np.mean(self._step_speeds)) if self._step_speeds else 0.0

        # Guard behavior
        intervention_count = len(self.intervention_steps)
        intervention_rate = intervention_count / n
        fallback_count = len(self.fallback_steps)
        fallback_rate = fallback_count / n
        stop_count = len(self.stop_steps)
        stop_rate = stop_count / n

        # Runtime
        mean_step_time = float(np.mean(self._step_times_ms)) if self._step_times_ms else 0.0
        p95_step_time = float(np.percentile(self._step_times_ms, 95)) if self._step_times_ms else 0.0

        return {
            # Safety
            "collision": 1 if collision else 0,
            "collision_rate": collision_rate,
            "collision_step": collision_step,
            "near_miss": near_miss_count,
            "near_miss_rate": near_miss_rate,
            "min_distance": min_dist_overall,
            # Task
            "success": success,
            "success_rate": success,
            "goal_reached": 1.0 if self.goal_reached else 0.0,
            "timeout": 1 if timeout else 0,
            "final_distance": final_dist,
            "path_length": path_length,
            "spl": spl,
            # Anti-freezing
            "freeze_steps": self._frozen_steps,
            "freeze_rate": freeze_rate,
            "deadlock": 1 if deadlock else 0,
            "average_speed": avg_speed,
            # Guard behavior
            "intervention_count": intervention_count,
            "intervention_rate": intervention_rate,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "stop_count": stop_count,
            "stop_rate": stop_rate,
            # Runtime
            "mean_step_time_ms": mean_step_time,
            "p95_step_time_ms": p95_step_time,
            # Phase 1 placeholders (certification not implemented)
            "certified_count": 0,
            "certified_rate": 0.0,
            "qp_intervention_rate": 0.0,
            "backup_feasible_rate": 0.0,
            "visibility_certified_rate": 0.0,
            "certification_implemented": False,
            "proposed_full_implemented": False,
            # Tube metrics (Phase 2)
            "tube_violations_total": self._tube_violations_total,
            "tube_violations_steps": len(self._tube_violations_steps),
            "tube_violation_rate": len(self._tube_violations_steps) / n,
            # Rollout metrics (Phase 3A)
            "rollout_passed_count": len(self._rollout_passed_steps),
            "rollout_passed_rate": len(self._rollout_passed_steps) / n,
            "fallback_stop_count": len(self._fallback_stop_steps),
            "fallback_yield_left_count": len(self._fallback_yield_left_steps),
            "fallback_yield_right_count": len(self._fallback_yield_right_steps),
            "fallback_slow_reverse_count": len(self._fallback_slow_reverse_steps),
            "emergency_stop_count": len(self._emergency_stop_steps),
            # CBF-QP metrics (Phase 3B-1)
            "cbf_qp_intervention_count": len(self._cbf_qp_intervention_steps),
            "cbf_qp_intervention_rate": len(self._cbf_qp_intervention_steps) / n,
            "cbf_qp_solve_success_count": len(self._cbf_qp_solve_success_steps),
            "cbf_qp_solve_failure_count": len(self._cbf_qp_solve_failure_steps),
            "filtered_action_count": len(self._filtered_action_steps),
            "rollout_after_qp_pass_count": len(self._rollout_after_qp_pass_steps),
            "rollout_after_qp_pass_rate": (
                len(self._rollout_after_qp_pass_steps) / n
                if self._filtered_action_steps else 0.0
            ),
            # Backup Validator metrics (Phase 3B-2)
            "backup_checked_count": len(self._backup_checked_steps),
            "backup_checked_rate": len(self._backup_checked_steps) / n,
            "backup_feasible_count": len(self._backup_feasible_steps),
            "backup_feasible_rate": len(self._backup_feasible_steps) / n if self._backup_checked_steps else 0.0,
            "backup_infeasible_count": len(self._backup_infeasible_steps),
            "safe_alternative_found_count": len(self._safe_alternative_found_steps),
            "safe_alternative_found_rate": len(self._safe_alternative_found_steps) / n if self._backup_checked_steps else 0.0,
            "backup_all_failed_count": len(self._backup_all_failed_steps),
            "backup_min_clearance_mean": float(np.mean(self._backup_min_clearances)) if self._backup_min_clearances else 0.0,
            "backup_min_clearance_min": float(np.min(self._backup_min_clearances)) if self._backup_min_clearances else 0.0,
            "backup_policy_stop_count": len(self._backup_policy_steps["stop"]),
            "backup_policy_slow_forward_count": len(self._backup_policy_steps["slow_forward"]),
            "backup_policy_slow_left_count": len(self._backup_policy_steps["slow_left"]),
            "backup_policy_slow_right_count": len(self._backup_policy_steps["slow_right"]),
            "backup_policy_turn_left_count": len(self._backup_policy_steps["turn_left_in_place"]),
            "backup_policy_turn_right_count": len(self._backup_policy_steps["turn_right_in_place"]),
            "backup_policy_reverse_count": len(self._backup_policy_steps["slow_reverse"]),
            "backup_policy_reverse_left_count": len(self._backup_policy_steps["reverse_left"]),
            "backup_policy_reverse_right_count": len(self._backup_policy_steps["reverse_right"]),
            # Phase 3B-2 Fix metrics
            "stop_infeasible_count": len(self._stop_infeasible_steps),
            "stop_infeasible_rate": len(self._stop_infeasible_steps) / n if self._backup_checked_steps else 0.0,
            "non_stop_backup_selected_count": len(self._non_stop_backup_selected_steps),
            "non_stop_backup_selected_rate": len(self._non_stop_backup_selected_steps) / n if self._backup_checked_steps else 0.0,
            # Phase 3B-2b metrics
            "backup_moving_toward_selected_count": len(self._moving_toward_selected_steps),
            "backup_clearance_trend_mean": float(np.mean(self._backup_clearance_trends)) if self._backup_clearance_trends else 0.0,
            "backup_min_ttc_mean": float(np.mean(self._backup_min_ttcs)) if self._backup_min_ttcs else 0.0,
            # Phase 3C metrics
            "encounter_type_counts": self._encounter_types,
            "mode_switch_count": len(self._mode_switch_steps),
            "side_switch_count": len(self._side_switch_steps),
            # Counts
            "total_steps": n,
        }

    # ── Convenience accessors ────────────────────────────────────────────────

    @property
    def has_collision(self) -> bool:
        return len(self.collision_steps) > 0

    @property
    def has_near_miss(self) -> bool:
        return len(self.near_miss_steps) > 0
