"""
planning/side_commitment.py
SafeDyn-VLN Guard: Stable Side Commitment with encounter-frame convention.

Encounter-frame left/right convention:
  - At encounter frame, commit to left or right escape direction
  - Mode hysteresis: switch only when current mode invalid or alternative
    significantly better
  - Persists for commitment_horizon steps

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class SideCommitmentConfig:
    """Configuration for side commitment."""
    commitment_horizon: int = 10       # steps to persist
    switch_threshold: float = 0.15     # score margin needed to switch
    grace_steps: int = 5               # grace period after expiry
    min_progress_advantage: float = 0.1 # min goal progress advantage to commit


@dataclass
class SideCommitmentState:
    """Mutable state for side commitment persistence."""
    committed_mode: str = "center"
    remaining_steps: int = 0
    grace_steps: int = 0
    last_committed_action: Optional[Dict[str, float]] = None
    last_selected_variant: Optional[str] = None
    commit_frame_step: int = -1
    switch_count: int = 0

    def tick(self) -> None:
        """Advance one step."""
        if self.remaining_steps > 0:
            self.remaining_steps -= 1
            self.grace_steps = 0
        if self.remaining_steps <= 0:
            if self.committed_mode in ("left", "right"):
                if self.grace_steps < 5:
                    self.grace_steps += 1
                else:
                    self.committed_mode = "center"
                    self.grace_steps = 0
                    self.last_committed_action = None

    def commit(self, mode: str, horizon: int, action: Optional[Dict[str, float]] = None,
               variant: Optional[str] = None, step: int = -1) -> None:
        """Commit to a side mode."""
        if self.committed_mode != mode:
            self.switch_count += 1
        self.committed_mode = mode
        self.remaining_steps = horizon
        self.grace_steps = 0
        self.last_committed_action = dict(action) if action else None
        self.last_selected_variant = variant
        self.commit_frame_step = step


def should_switch_mode(
    current_mode: str,
    current_score: float,
    candidate_mode: str,
    candidate_score: float,
    config: SideCommitmentConfig,
    current_feasible: bool = True,
) -> bool:
    """
    Determine if mode should switch.

    Switch only when:
      1. Current mode is infeasible, OR
      2. Candidate is significantly better (by switch_threshold)
    """
    if not current_feasible:
        return True

    if current_mode == "center":
        # From center, any side that's better by threshold
        return candidate_score < current_score - config.switch_threshold

    # From a committed side, need larger margin to switch
    return candidate_score < current_score - config.switch_threshold * 2.0


def evaluate_side_commitment(
    center_action: Dict[str, float],
    left_action: Dict[str, float],
    right_action: Dict[str, float],
    center_score: float,
    left_score: float,
    right_score: float,
    state: SideCommitmentState,
    config: SideCommitmentConfig,
    center_feasible: bool = True,
    left_feasible: bool = True,
    right_feasible: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate side commitment given current state and candidate scores.

    Returns dict with:
      selected_mode, selected_action, switched, hysteresis_applied,
      commitment_info
    """
    state.tick()

    # If committed and still feasible, stay committed
    if (state.committed_mode == "left" and left_feasible
            and state.remaining_steps > 0):
        return {
            "selected_mode": "left",
            "selected_action": dict(left_action),
            "switched": False,
            "hysteresis_applied": True,
            "commitment_info": {
                "remaining_steps": state.remaining_steps,
                "grace_steps": state.grace_steps,
            },
        }
    if (state.committed_mode == "right" and right_feasible
            and state.remaining_steps > 0):
        return {
            "selected_mode": "right",
            "selected_action": dict(right_action),
            "switched": False,
            "hysteresis_applied": True,
            "commitment_info": {
                "remaining_steps": state.remaining_steps,
                "grace_steps": state.grace_steps,
            },
        }

    # Evaluate candidates
    candidates = [
        ("center", center_score, center_action, center_feasible),
        ("left", left_score, left_action, left_feasible),
        ("right", right_score, right_action, right_feasible),
    ]

    # Filter to feasible candidates
    feasible = [(m, s, a, f) for m, s, a, f in candidates if f]
    if not feasible:
        return {
            "selected_mode": "center",
            "selected_action": dict(center_action),
            "switched": state.committed_mode != "center",
            "hysteresis_applied": False,
            "commitment_info": {"remaining_steps": 0, "grace_steps": 0},
        }

    # Pick best feasible candidate
    best = min(feasible, key=lambda x: x[1])
    best_mode, best_score, best_action, _ = best

    # Check hysteresis
    current_score = {"center": center_score, "left": left_score,
                     "right": right_score}.get(state.committed_mode, float("inf"))
    current_feasible_flag = {"center": center_feasible, "left": left_feasible,
                             "right": right_feasible}.get(state.committed_mode, False)

    if best_mode == state.committed_mode:
        # Stay on current mode
        return {
            "selected_mode": best_mode,
            "selected_action": dict(best_action),
            "switched": False,
            "hysteresis_applied": True,
            "commitment_info": {
                "remaining_steps": state.remaining_steps,
                "grace_steps": state.grace_steps,
            },
        }

    should = should_switch_mode(
        state.committed_mode, current_score,
        best_mode, best_score, config, current_feasible_flag,
    )

    if should and best_mode in ("left", "right"):
        state.commit(best_mode, config.commitment_horizon, best_action,
                     step=state.commit_frame_step + 1)
        return {
            "selected_mode": best_mode,
            "selected_action": dict(best_action),
            "switched": True,
            "hysteresis_applied": False,
            "commitment_info": {
                "remaining_steps": state.remaining_steps,
                "grace_steps": 0,
            },
        }

    return {
        "selected_mode": "center",
        "selected_action": dict(center_action),
        "switched": state.committed_mode != "center",
        "hysteresis_applied": True,
        "commitment_info": {
            "remaining_steps": state.remaining_steps,
            "grace_steps": state.grace_steps,
        },
    }
