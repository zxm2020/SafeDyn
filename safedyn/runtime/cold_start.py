"""
runtime/cold_start.py
SafeDyn-VLN Guard: Cold-start initialization.

Initializes:
  - Static map placeholder
  - Track cache
  - B_plan and B_cert
  - First planner call
  - First cached plan

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ColdStartConfig:
    """Configuration for cold-start initialization."""
    warmup_steps: int = 5            # steps before planner is trusted
    initial_robot_radius: float = 0.25
    initial_horizon: int = 20
    initial_dt: float = 0.05
    initial_planning_margin: float = 0.20
    initial_certified_margin: float = 0.45


@dataclass
class ColdStartState:
    """State produced by cold-start initialization."""
    initialized: bool = False
    warmup_step: int = 0
    first_plan: Optional[Dict[str, Any]] = None
    first_b_plan: List[Any] = field(default_factory=list)
    first_b_cert: List[Any] = field(default_factory=list)
    first_tracks: List[Any] = field(default_factory=list)
    static_map_loaded: bool = False
    tracker_initialized: bool = False
    planner_called: bool = False


def cold_start_initialize(
    robot_pos: np.ndarray,
    robot_yaw: float,
    goal_pos: np.ndarray,
    entities: List[Dict[str, Any]],
    config: ColdStartConfig,
) -> ColdStartState:
    """
    Perform cold-start initialization.

    1. Load static map placeholder
    2. Initialize tracker with first observations
    3. Build initial B_plan and B_cert
    4. Run first planner call
    5. Cache first plan

    Returns ColdStartState with initialization results.
    """
    state = ColdStartState()

    # Step 1: Static map placeholder
    state.static_map_loaded = True

    # Step 2: Initialize tracker (would use KalmanTracker in real runtime)
    state.tracker_initialized = True

    # Step 3: Initial B_plan and B_cert are empty (no observations yet)
    state.first_b_plan = []
    state.first_b_cert = []
    state.first_tracks = []

    # Step 4: First planner call — nominal forward action
    state.first_plan = {
        "action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
        "source": "cold_start",
        "mode": "forward",
        "step": 0,
    }
    state.planner_called = True

    state.warmup_step = 0
    state.initialized = True

    return state


def is_warmup_complete(state: ColdStartState, config: ColdStartConfig) -> bool:
    """Check if warmup period is complete."""
    return state.initialized and state.warmup_step >= config.warmup_steps


def advance_warmup(state: ColdStartState) -> None:
    """Advance warmup step counter."""
    state.warmup_step += 1
