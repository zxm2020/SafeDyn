"""
runtime/stale_plan.py
SafeDyn-VLN Guard: Stale plan detection and rejection.

Rejects stale cached plans and triggers refresh or cautious mode.
Cached plan must never bypass CertifiedAccept.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time


@dataclass
class StalePlanConfig:
    """Configuration for stale plan detection."""
    max_plan_age_s: float = 0.3       # max age before plan is stale
    max_steps_since_plan: int = 5     # max steps since plan generation
    trigger_cautious_on_stale: bool = True
    trigger_refresh_on_stale: bool = True


@dataclass
class StalePlanResult:
    """Result of stale plan check."""
    is_stale: bool
    plan_age_s: float
    steps_since_plan: int
    action: str   # "use_plan", "refresh", "cautious"
    reason: str


def check_plan_staleness(
    plan_timestamp: float,
    current_timestamp: float,
    steps_since_plan: int,
    config: StalePlanConfig,
) -> StalePlanResult:
    """
    Check if a cached plan is stale.

    A plan is stale if:
      1. Its age exceeds max_plan_age_s, OR
      2. More than max_steps_since_plan have elapsed
    """
    age = current_timestamp - plan_timestamp
    is_stale = age > config.max_plan_age_s or steps_since_plan > config.max_steps_since_plan

    if is_stale:
        if config.trigger_refresh_on_stale:
            action = "refresh"
            reason = f"plan_age={age:.3f}s > max={config.max_plan_age_s}s"
        elif config.trigger_cautious_on_stale:
            action = "cautious"
            reason = f"plan_stale_age={age:.3f}s"
        else:
            action = "use_plan"
            reason = "stale_but_forced"
    else:
        action = "use_plan"
        reason = "plan_fresh"

    return StalePlanResult(
        is_stale=is_stale,
        plan_age_s=age,
        steps_since_plan=steps_since_plan,
        action=action,
        reason=reason,
    )


def reject_stale_plan(
    plan: Dict[str, Any],
    current_timestamp: float,
    config: StalePlanConfig,
) -> Dict[str, Any]:
    """
    Reject stale plan and return replacement.

    Replacement is either a refresh signal or cautious mode action.
    Never bypasses CertifiedAccept.
    """
    plan_ts = plan.get("timestamp", 0.0)
    result = check_plan_staleness(plan_ts, current_timestamp, 0, config)

    if result.is_stale:
        if result.action == "refresh":
            return {
                "action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                "source": "stale_plan_refresh",
                "stale_plan_rejected": True,
                "stale_plan_age": result.plan_age_s,
            }
        else:
            return {
                "action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                "source": "stale_plan_cautious",
                "stale_plan_rejected": True,
                "stale_plan_age": result.plan_age_s,
            }

    return dict(plan)
