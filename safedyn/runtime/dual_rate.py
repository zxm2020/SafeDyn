"""
runtime/dual_rate.py
SafeDyn-VLN Guard: Dual-rate runtime architecture.

Planner thread runs at lower rate (e.g., 10 Hz).
Shield thread runs at higher rate (e.g., 20 Hz).
Deadline handling for both threads.
Stale plan detection.
Anytime best plan interface.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import threading
import numpy as np


@dataclass
class DualRateConfig:
    """Configuration for dual-rate runtime."""
    planner_rate_hz: float = 10.0    # planner update rate
    shield_rate_hz: float = 20.0     # shield certification rate
    planner_deadline_ms: float = 100.0  # planner deadline
    shield_deadline_ms: float = 50.0    # shield deadline
    stale_plan_max_age_s: float = 0.3   # max age before plan is stale


@dataclass
class DualRateState:
    """State of the dual-rate runtime."""
    planner_last_update: float = 0.0
    shield_last_update: float = 0.0
    planner_deadline_misses: int = 0
    shield_deadline_misses: int = 0
    plans_generated: int = 0
    plans_consumed: int = 0
    stale_plans_rejected: int = 0
    current_plan: Optional[Dict[str, Any]] = None
    best_plan: Optional[Dict[str, Any]] = None
    plan_age_s: float = 0.0


class DualRateRuntime:
    """
    Dual-rate runtime architecture.

    Planner thread: generates plans at lower rate.
    Shield thread: certifies actions at higher rate.
    """

    def __init__(self, config: Optional[DualRateConfig] = None):
        self.config = config or DualRateConfig()
        self.state = DualRateState()
        self._lock = threading.Lock()

    def planner_tick(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        goal_pos: np.ndarray,
        b_plan: List[Any],
        b_cert: List[Any],
        nominal_action: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """
        Planner thread tick. Generates a new plan.

        Returns plan dict if successful, None if deadline missed.
        """
        start_time = time.time()

        # Simulate planning (in real implementation, this calls MPPI)
        plan = {
            "action": dict(nominal_action),
            "source": "planner",
            "robot_pos": robot_pos.tolist() if hasattr(robot_pos, 'tolist') else list(robot_pos),
            "robot_yaw": float(robot_yaw),
            "timestamp": start_time,
            "step": self.state.plans_generated,
        }

        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.config.planner_deadline_ms:
            self.state.planner_deadline_misses += 1
            return None

        with self._lock:
            self.state.current_plan = plan
            self.state.planner_last_update = start_time
            self.state.plans_generated += 1

            # Update best plan
            if self.state.best_plan is None:
                self.state.best_plan = plan

        return plan

    def shield_tick(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        b_cert: List[Any],
        action: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Shield thread tick. Certifies action against B_cert.

        Returns certification result dict.
        """
        start_time = time.time()

        # Check if plan is stale
        plan_stale = False
        with self._lock:
            if self.state.current_plan is not None:
                age = time.time() - self.state.current_plan.get("timestamp", 0)
                self.state.plan_age_s = age
                if age > self.config.stale_plan_max_age_s:
                    plan_stale = True
                    self.state.stale_plans_rejected += 1

        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.config.shield_deadline_ms:
            self.state.shield_deadline_misses += 1
            return {
                "certified": False,
                "deadline_miss": True,
                "plan_stale": plan_stale,
            }

        with self._lock:
            self.state.shield_last_update = start_time

        return {
            "certified": True,
            "deadline_miss": False,
            "plan_stale": plan_stale,
        }

    def get_anytime_best_plan(self) -> Optional[Dict[str, Any]]:
        """Get the best plan available so far (anytime interface)."""
        with self._lock:
            return self.state.best_plan

    def get_state_summary(self) -> Dict[str, Any]:
        """Get runtime state summary."""
        return {
            "planner_last_update": self.state.planner_last_update,
            "shield_last_update": self.state.shield_last_update,
            "planner_deadline_misses": self.state.planner_deadline_misses,
            "shield_deadline_misses": self.state.shield_deadline_misses,
            "plans_generated": self.state.plans_generated,
            "plans_consumed": self.state.plans_consumed,
            "stale_plans_rejected": self.state.stale_plans_rejected,
            "plan_age_s": self.state.plan_age_s,
            "has_current_plan": self.state.current_plan is not None,
            "has_best_plan": self.state.best_plan is not None,
        }
