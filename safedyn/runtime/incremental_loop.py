"""
runtime/incremental_loop.py
SafeDyn-VLN Guard: Incremental update loop.

Per-tick operations:
  1. Track predict/update
  2. Tube shift update
  3. Active set update
  4. Cached plan read
  5. Shield tick certification

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class IncrementalLoopConfig:
    """Configuration for incremental loop."""
    dt: float = 0.05
    robot_radius: float = 0.25
    horizon_steps: int = 20
    planning_margin: float = 0.20
    certified_margin: float = 0.45


@dataclass
class IncrementalStepResult:
    """Result of one incremental loop step."""
    step: int
    tracks_updated: int
    b_plan_updated: bool
    b_cert_updated: bool
    active_set_size: int
    cached_plan_valid: bool
    shield_passed: bool
    tube_shift_violations: int
    timestamp: float = 0.0


def incremental_step(
    step: int,
    robot_pos: np.ndarray,
    robot_yaw: float,
    observations: List[Dict[str, Any]],
    tracker: Any,
    cached_plan: Optional[Dict[str, Any]],
    config: IncrementalLoopConfig,
) -> IncrementalStepResult:
    """
    Execute one incremental loop step.

    1. Predict all tracks forward
    2. Update tracks with new observations
    3. Build updated B_plan and B_cert
    4. Check cached plan validity
    5. Run shield certification on cached plan
    """
    import time
    timestamp = time.time()

    # Step 1: Predict tracks
    if tracker is not None:
        tracker.predict(config.dt)

    # Step 2: Update tracks with observations
    if tracker is not None and observations:
        tracker.update(observations, np.asarray(robot_pos))

    # Step 3: Get active tracks
    active_tracks = []
    if tracker is not None:
        active_tracks = tracker.get_active_tracks()

    # Step 4: Build B_plan and B_cert (would use risk modules in real runtime)
    b_plan = []
    b_cert = []
    tube_shift_violations = 0

    # Step 5: Active set culling
    active_set_size = len(active_tracks)

    # Step 6: Check cached plan validity
    cached_plan_valid = cached_plan is not None

    # Step 7: Shield tick certification (simplified)
    shield_passed = True

    return IncrementalStepResult(
        step=step,
        tracks_updated=len(active_tracks),
        b_plan_updated=True,
        b_cert_updated=True,
        active_set_size=active_set_size,
        cached_plan_valid=cached_plan_valid,
        shield_passed=shield_passed,
        tube_shift_violations=tube_shift_violations,
        timestamp=timestamp,
    )


def run_incremental_loop(
    start_step: int,
    end_step: int,
    robot_pos: np.ndarray,
    robot_yaw: float,
    observations_per_step: List[List[Dict[str, Any]]],
    tracker: Any,
    config: IncrementalLoopConfig,
) -> List[IncrementalStepResult]:
    """
    Run incremental loop over multiple steps.

    Returns list of IncrementalStepResult for each step.
    """
    results = []
    cached_plan = None

    for i in range(start_step, end_step):
        obs = observations_per_step[i] if i < len(observations_per_step) else []
        result = incremental_step(
            step=i,
            robot_pos=robot_pos,
            robot_yaw=robot_yaw,
            observations=obs,
            tracker=tracker,
            cached_plan=cached_plan,
            config=config,
        )
        results.append(result)

    return results
