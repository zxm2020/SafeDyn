"""
safety/action_candidates.py
Phase 3B-2: Candidate action set for backup safe alternative selection.
Coordinate system: x-z-yaw.

All candidates use [linear_velocity, angular_velocity] format.
Names are used for logging and scoring.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class CandidateAction:
    """A candidate backup action."""
    name: str
    linear_velocity: float
    angular_velocity: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity,
        }


# ── Candidate set ────────────────────────────────────────────────────────────

CANDIDATES: List[CandidateAction] = [
    CandidateAction(
        name="stop",
        linear_velocity=0.0,
        angular_velocity=0.0,
        description="Full stop — may be infeasible if entity keeps approaching",
    ),
    CandidateAction(
        name="slow_forward",
        linear_velocity=0.2,
        angular_velocity=0.0,
        description="Slow forward — minimal progress",
    ),
    CandidateAction(
        name="slow_left",
        linear_velocity=0.2,
        angular_velocity=0.8,
        description="Slow forward + turn left",
    ),
    CandidateAction(
        name="slow_right",
        linear_velocity=0.2,
        angular_velocity=-0.8,
        description="Slow forward + turn right",
    ),
    CandidateAction(
        name="turn_left_in_place",
        linear_velocity=0.0,
        angular_velocity=1.0,
        description="Turn left in place",
    ),
    CandidateAction(
        name="turn_right_in_place",
        linear_velocity=0.0,
        angular_velocity=-1.0,
        description="Turn right in place",
    ),
    CandidateAction(
        name="slow_reverse",
        linear_velocity=-0.2,
        angular_velocity=0.0,
        description="Slow reverse — retreat",
    ),
    CandidateAction(
        name="reverse_left",
        linear_velocity=-0.2,
        angular_velocity=0.6,
        description="Reverse + turn left",
    ),
    CandidateAction(
        name="reverse_right",
        linear_velocity=-0.2,
        angular_velocity=-0.6,
        description="Reverse + turn right",
    ),
]

# Map name → CandidateAction for fast lookup
CANDIDATE_MAP: Dict[str, CandidateAction] = {c.name: c for c in CANDIDATES}


def get_all_candidates() -> List[CandidateAction]:
    """Return all candidate actions."""
    return CANDIDATES


def get_candidate_by_name(name: str) -> CandidateAction:
    """Return candidate by name, raise KeyError if not found."""
    return CANDIDATE_MAP[name]


def candidate_to_dict(c: CandidateAction) -> Dict[str, Any]:
    return c.to_dict()
