"""
tracking/track_cache.py
Phase 2: Track cache for multi-step temporal consistency.
Stores per-episode track history for tube logging.
Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TrackSnapshot:
    """Per-step snapshot of all tracked entities."""
    step: int
    tracks: List[dict]   # serialized KalmanTrack.to_dict() per track
    num_tracks: int
    mean_uncertainty_radius: float


class TrackCache:
    """
    Stores per-step track state snapshots for episode logging.
    Also computes running tube statistics.
    """

    def __init__(self):
        self.snapshots: List[TrackSnapshot] = []
        self._tube_plan_radii: List[float] = []
        self._tube_cert_radii: List[float] = []
        self._violations: List[int] = []

    def record_step(
        self,
        step: int,
        tracks: list,
        plan_tube: list,
        cert_tube: list,
    ) -> None:
        """Record one simulation step."""
        plan_radii = [float(e.radius) for e in plan_tube] if plan_tube else []
        cert_radii = [float(e.radius) for e in cert_tube] if cert_tube else []
        mean_plan = float(np.mean(plan_radii)) if plan_radii else 0.0
        mean_cert = float(np.mean(cert_radii)) if cert_radii else 0.0
        violation = 1 if cert_radii and plan_radii and any(c < p for c, p in zip(cert_radii, plan_radii)) else 0

        self._tube_plan_radii.extend(plan_radii)
        self._tube_cert_radii.extend(cert_radii)
        self._violations.append(violation)

        track_dicts = []
        for t in tracks:
            track_dicts.append({
                "track_id": t.track_id,
                "position": t.position().tolist(),
                "radius": t.radius,
                "confidence": t.confidence,
            })

        self.snapshots.append(TrackSnapshot(
            step=step,
            tracks=track_dicts,
            num_tracks=len(tracks),
            mean_uncertainty_radius=mean_plan,
        ))

    def get_summary(self) -> dict:
        """Compute tube statistics for episode summary."""
        n = max(len(self.snapshots), 1)
        return {
            "num_tracks_mean": np.mean([s.num_tracks for s in self.snapshots]) if self.snapshots else 0.0,
            "num_tracks_max": max([s.num_tracks for s in self.snapshots]) if self.snapshots else 0,
            "mean_plan_tube_radius": float(np.mean(self._tube_plan_radii) if self._tube_plan_radii else 0.0),
            "mean_cert_tube_radius": float(np.mean(self._tube_cert_radii) if self._tube_cert_radii else 0.0),
            "cert_ge_plan_violations": sum(self._violations),
            "cert_ge_plan_violation_rate": float(sum(self._violations) / n),
        }
