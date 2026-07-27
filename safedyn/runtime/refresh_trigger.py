"""
runtime/refresh_trigger.py
SafeDyn-VLN Guard: Full refresh trigger.

Triggers full planner re-plan when:
  - Plan is stale
  - Entity configuration changed significantly
  - Robot deviated from planned path
  - New entity appeared

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RefreshTriggerConfig:
    """Configuration for refresh trigger."""
    position_deviation_threshold: float = 0.5  # meters
    entity_change_threshold: float = 0.3       # meters
    new_entity_trigger: bool = True
    stale_plan_trigger: bool = True
    max_consecutive_refreshes: int = 3


@dataclass
class RefreshTriggerState:
    """State of refresh trigger."""
    last_refresh_step: int = -1
    consecutive_refreshes: int = 0
    refresh_count: int = 0
    last_robot_pos: Optional[np.ndarray] = None
    last_entity_positions: Dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.last_entity_positions is None:
            self.last_entity_positions = {}


def check_refresh_needed(
    robot_pos: np.ndarray,
    planned_pos: np.ndarray,
    entities: List[Dict[str, Any]],
    state: RefreshTriggerState,
    config: RefreshTriggerConfig,
    current_step: int,
) -> Dict[str, Any]:
    """
    Check if a full refresh is needed.

    Returns dict with refresh_needed, reason, and triggers.
    """
    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    planned_pos = np.asarray(planned_pos, dtype=np.float64)

    triggers = []

    # Position deviation
    pos_dev = float(np.linalg.norm(robot_pos - planned_pos))
    if pos_dev > config.position_deviation_threshold:
        triggers.append(f"position_deviation={pos_dev:.3f}")

    # Entity change
    for e in entities:
        eid = e.get("entity_id", "unknown")
        epos = np.asarray(e.get("position", [0.0, 0.0]), dtype=np.float64)
        if eid in state.last_entity_positions:
            change = float(np.linalg.norm(epos - state.last_entity_positions[eid]))
            if change > config.entity_change_threshold:
                triggers.append(f"entity_{eid}_moved={change:.3f}")
        elif config.new_entity_trigger:
            triggers.append(f"new_entity_{eid}")

    # Rate limiting
    refresh_allowed = state.consecutive_refreshes < config.max_consecutive_refreshes

    refresh_needed = len(triggers) > 0 and refresh_allowed

    if refresh_needed:
        state.last_refresh_step = current_step
        state.consecutive_refreshes += 1
        state.refresh_count += 1
    else:
        state.consecutive_refreshes = 0

    # Update state
    state.last_robot_pos = robot_pos.copy()
    for e in entities:
        eid = e.get("entity_id", "unknown")
        state.last_entity_positions[eid] = np.asarray(
            e.get("position", [0.0, 0.0]), dtype=np.float64
        )

    return {
        "refresh_needed": refresh_needed,
        "triggers": triggers,
        "consecutive_refreshes": state.consecutive_refreshes,
        "total_refreshes": state.refresh_count,
    }
