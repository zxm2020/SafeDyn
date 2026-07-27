"""
tracking/temporal_filter.py
Phase 2: Temporal entity state filter for tracking + confidence management.
Coordinate system: x-z-yaw.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TemporalEntityState:
    """
    Filtered entity state with temporal consistency scoring.
    Used as the interface between Kalman tracker and tube builders.
    """
    track_id: str
    position: np.ndarray     # [x, z] estimate
    velocity: np.ndarray     # [vx, vz] estimate
    radius: float           # collision radius
    uncertainty_radius: float  # inflated radius for certified tube
    confidence: float       # 0..1
    confidence_level: str    # 'high' or 'low'
    age: int               # steps since first seen
    missed_steps: int       # consecutive missed
    to_dict_called: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "position": self.position.tolist(),      # [x, z]
            "velocity": self.velocity.tolist(),       # [vx, vz]
            "radius": self.radius,
            "uncertainty_radius": self.uncertainty_radius,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "age": self.age,
            "missed_steps": self.missed_steps,
        }


def build_temporal_states(tracks) -> List[TemporalEntityState]:
    """
    Convert a list of KalmanTrack objects to TemporalEntityState objects.
    """
    states = []
    for t in tracks:
        states.append(TemporalEntityState(
            track_id=t.track_id,
            position=t.position(),
            velocity=t.velocity(),
            radius=t.radius,
            uncertainty_radius=t.uncertainty_radius(),
            confidence=t.confidence,
            confidence_level=t.confidence_level(),
            age=t.age,
            missed_steps=t.missed_steps,
        ))
    return states


def nearest_track_distance(robot_pos: np.ndarray, tracks) -> float:
    """Minimum Euclidean distance from robot to any tracked entity (x-z plane)."""
    if not tracks:
        return float("inf")
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    min_d = float("inf")
    for t in tracks:
        d = float(np.linalg.norm(robot_pos - t.position()))
        if d < min_d:
            min_d = d
    return min_d
