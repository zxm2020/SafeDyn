"""
tracking/kalman.py
Phase 2: Linear Kalman filter for 2D position-velocity tracking on the x-z plane.
Coordinate system: x-z-yaw.
State: [p_x, p_z, v_x, v_z]^T
Measurement: [z_x, z_z] (noisy position observation)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────────────

# Process noise for constant-velocity model (m/s per sqrt(s))
DEFAULT_PROCESS_NOISE = 0.05
# Observation noise for position measurement (m)
DEFAULT_OBS_NOISE = 0.03


# ── Kalman track ──────────────────────────────────────────────────────────

@dataclass
class KalmanTrack:
    """
    Single tracked entity using a linear Kalman filter on the x-z plane.
    State: [p_x, p_z, v_x, v_z]^T
    """
    track_id: str
    state: np.ndarray              # [p_x, p_z, v_x, v_z]
    covariance: np.ndarray          # 4×4 covariance matrix
    radius: float = 0.2           # collision radius of this entity
    confidence: float = 1.0         # 0..1 (decays on missed observations)
    age: int = 0                  # total steps since first observation
    missed_steps: int = 0          # consecutive steps without observation
    max_missed: int = 20          # drop track after this many missed steps
    low_confidence_threshold: float = 0.4
    _PERSIST_AT_NEARBY: bool = field(default=False, repr=False)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=np.float64).reshape(4)
        self.covariance = np.asarray(self.covariance, dtype=np.float64).reshape(4, 4)
        self._PERSIST_AT_NEARBY = False  # don't delete nearby low-conf tracks

    # ── State accessors ───────────────────────────────────────────────────

    def position(self) -> np.ndarray:
        """Return [p_x, p_z] position estimate."""
        return self.state[:2].copy()

    def velocity(self) -> np.ndarray:
        """Return [v_x, v_z] velocity estimate."""
        return self.state[2:].copy()

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity()))

    def uncertainty_radius(self) -> float:
        """
        Uncertainty-inflated radius: sqrt(largest eigenvalue of position covariance)
        + entity radius.  Used by certified tube construction.
        """
        pos_cov = self.covariance[:2, :2]
        max_var = float(np.max(np.linalg.eigvalsh(pos_cov)))
        return float(np.sqrt(max_var)) + self.radius

    def confidence_level(self) -> str:
        """Return 'high' / 'low' / 'dropped'."""
        if self.confidence < self.low_confidence_threshold:
            return "low"
        return "high"

    # ── Prediction step ───────────────────────────────────────────────

    def predict(self, dt: float, process_noise: float = DEFAULT_PROCESS_NOISE) -> None:
        """
        Advance the track by dt seconds using the constant-velocity model.
        Covariance grows with process noise.
        """
        # State transition matrix (constant velocity)
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Process noise covariance (tunable per tracked entity)
        q = float(process_noise)
        Q = np.array([
            [dt**4/4, 0.0,        dt**3/2, 0.0       ],
            [0.0,        dt**4/4, 0.0,        dt**3/2],
            [dt**3/2, 0.0,        dt**2,      0.0       ],
            [0.0,        dt**3/2, 0.0,        dt**2      ],
        ], dtype=np.float64) * q**2

        # Predict
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        self.age += 1

    # ── Observation update ─────────────────────────────────────────────

    def update(self, z: np.ndarray, obs_noise: float = DEFAULT_OBS_NOISE) -> None:
        """
        Update track with a noisy position observation [z_x, z_z].
        Resets missed_steps counter and restores confidence.
        """
        z = np.asarray(z, dtype=np.float64).reshape(2)

        # Measurement matrix (observe position only)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance
        r = float(obs_noise)
        R = np.eye(2, dtype=np.float64) * r**2

        # Kalman update
        S = H @ self.covariance @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.covariance @ H.T @ S_inv          # 4×2
        y = z - H @ self.state                      # innovation
        self.state = self.state + K @ y
        I4 = np.eye(4, dtype=np.float64)
        self.covariance = (I4 - K @ H) @ self.covariance

        # Reset missed counter, restore confidence
        self.missed_steps = 0
        self.confidence = min(1.0, self.confidence + 0.1)

    # ── Missed observation ─────────────────────────────────────────────

    def mark_missed(self, confidence_decay: float = 0.9) -> None:
        """
        Record that no observation was received for this step.
        Confidence decays, covariance grows (already handled by predict).
        """
        self.missed_steps += 1
        self.confidence = float(self.confidence) * float(confidence_decay)

    def should_drop(self) -> bool:
        """
        Drop track if missed_steps exceeded max_missed AND track is far from robot.
        Nearby low-confidence tracks are kept to avoid free-space false optimism.
        """
        return self.missed_steps > self.max_missed and not self._PERSIST_AT_NEARBY

    def to_dict(self) -> dict:
        """Serializable snapshot for logging."""
        return {
            "track_id": self.track_id,
            "position": self.position().tolist(),     # [x, z]
            "velocity": self.velocity().tolist(),       # [vx, vz]
            "radius": self.radius,
            "confidence": self.confidence,
            "age": self.age,
            "missed_steps": self.missed_steps,
            "uncertainty_radius": self.uncertainty_radius(),
            "confidence_level": self.confidence_level(),
        }


# ── Tracker ──────────────────────────────────────────────────────────────

class KalmanTracker:
    """
    Manages a set of KalmanTrack objects.
    Handles prediction, association, and update on each simulation step.
    Coordinate system: x-z-yaw.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        observation_noise: float = DEFAULT_OBS_NOISE,
        confidence_decay: float = 0.9,
        max_missed_steps: int = 20,
        low_confidence_threshold: float = 0.4,
        assignment_distance_threshold: float = 1.5,
    ):
        self.process_noise = float(process_noise)
        self.observation_noise = float(observation_noise)
        self.confidence_decay = float(confidence_decay)
        self.max_missed_steps = int(max_missed_steps)
        self.low_conf_threshold = float(low_confidence_threshold)
        self.assign_thresh = float(assignment_distance_threshold)
        self._tracks: Dict[str, KalmanTrack] = {}
        self._next_id = 0

    @property
    def tracks(self) -> Dict[str, KalmanTrack]:
        return self._tracks

    def predict(self, dt: float) -> None:
        """Predict all tracks one step forward."""
        for track in self._tracks.values():
            track.predict(dt, self.process_noise)
            track.mark_missed(self.confidence_decay)

    def update(self, observations: List[dict], robot_pos: np.ndarray) -> None:
        """
        Update tracks with noisy observations.
        observations: list of dicts with keys: position [x,z], radius
        Coordinate system: x-z plane.
        """
        obs_xy = [np.array(o["position"], dtype=np.float64) for o in observations]
        track_ids = list(self._tracks.keys())
        assigned = set()

        # Associate each observation to nearest unmatched track
        for obs, obs_pos in zip(observations, obs_xy):
            best_id = None
            best_dist = float("inf")

            for tid in track_ids:
                if tid in assigned:
                    continue
                track_pos = self._tracks[tid].position()
                d = float(np.linalg.norm(obs_pos - track_pos))
                if d < best_dist and d < self.assign_thresh:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                # Update matched track
                self._tracks[best_id].update(obs_pos, self.observation_noise)
                assigned.add(best_id)
                # Restore radius from observation
                self._tracks[best_id].radius = float(obs.get("radius", 0.2))
            else:
                # New track
                new_id = f"track_{self._next_id}"
                self._next_id += 1
                self._tracks[new_id] = KalmanTrack(
                    track_id=new_id,
                    state=np.array([obs_pos[0], obs_pos[1], 0.0, 0.0], dtype=np.float64),
                    covariance=np.eye(4, dtype=np.float64) * (self.observation_noise**2),
                    radius=float(obs.get("radius", 0.2)),
                    confidence=1.0,
                    max_missed=self.max_missed_steps,
                    low_confidence_threshold=self.low_conf_threshold,
                )

        # Mark unassigned tracks as missed
        for tid in track_ids:
            if tid not in assigned:
                self._tracks[tid].mark_missed(self.confidence_decay)

        # Drop expired tracks
        to_drop = [tid for tid, t in self._tracks.items() if t.should_drop()]
        for tid in to_drop:
            del self._tracks[tid]

    def get_active_tracks(self) -> List[KalmanTrack]:
        """Return all tracks (including low-confidence, excluding dropped)."""
        return list(self._tracks.values())

    def get_high_conf_tracks(self) -> List[KalmanTrack]:
        return [t for t in self._tracks.values() if t.confidence_level() == "high"]

    def get_all_track_dicts(self) -> List[dict]:
        """Return serializable list for logging."""
        return [t.to_dict() for t in self._tracks.values()]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
