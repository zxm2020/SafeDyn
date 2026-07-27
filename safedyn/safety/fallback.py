"""
safety/fallback.py
Phase 3A: Fallback hierarchy for emergency action selection.
Coordinate system: x-z-yaw.

When the nominal VLN action fails exact rollout validation,
this module generates a priority-ordered list of fallback candidates
and selects the first one that passes rollout validation.

Fallback types (in priority order):
  1. stop           — zero velocity, zero angular velocity
  2. yield_left     — slow down + turn left (angular velocity > 0)
  3. yield_right    — slow down + turn right (angular velocity < 0)
  4. slow_reverse   — small backward velocity (if dynamics allow)

If no candidate passes, emit emergency_stop.

IMPORTANT: These are NOT safety certificates. All actions are validated
geometrically via exact_rollout. certified=False.
"""

from typing import List, Dict, Any, Tuple


# ── Fallback candidate builders ────────────────────────────────────────────

def make_stop_action() -> Dict[str, Any]:
    return {"linear_velocity": 0.0, "angular_velocity": 0.0}


def make_yield_left_action(
    max_linear: float = 0.8,
    max_angular: float = 1.2,
    yield_linear: float = 0.3,
    yield_angular: float = 0.8,
) -> Dict[str, Any]:
    """Yield left: reduce speed + turn left (positive omega)."""
    return {
        "linear_velocity": float(min(yield_linear, max_linear)),
        "angular_velocity": float(min(yield_angular, max_angular)),
    }


def make_yield_right_action(
    max_linear: float = 0.8,
    max_angular: float = 1.2,
    yield_linear: float = 0.3,
    yield_angular: float = 0.8,
) -> Dict[str, Any]:
    """Yield right: reduce speed + turn right (negative omega)."""
    return {
        "linear_velocity": float(min(yield_linear, max_linear)),
        "angular_velocity": float(-min(yield_angular, max_angular)),
    }


def make_slow_reverse_action(
    max_linear: float = 0.8,
    reverse_speed: float = 0.15,
) -> Dict[str, Any]:
    """Slow reverse: small backward velocity, no turning."""
    return {
        "linear_velocity": float(-min(reverse_speed, max_linear)),
        "angular_velocity": 0.0,
    }


# ── Fallback priority list ─────────────────────────────────────────────────

def get_fallback_candidates(
    max_linear: float = 0.8,
    max_angular: float = 1.2,
    include_reverse: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return ordered list of (fallback_name, action_dict) candidates.

    Priority: stop → yield_left → yield_right → slow_reverse
    """
    candidates = [
        ("stop", make_stop_action()),
        ("yield_left", make_yield_left_action(max_linear, max_angular)),
        ("yield_right", make_yield_right_action(max_linear, max_angular)),
    ]
    if include_reverse:
        candidates.append(
            ("slow_reverse", make_slow_reverse_action(max_linear))
        )
    return candidates


# ── Selection ─────────────────────────────────────────────────────────────

def select_fallback(
    candidate_actions: List[Tuple[str, Dict[str, Any]]],
    safe_actions: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Select the highest-priority fallback that passed rollout validation.

    Args:
      candidate_actions: ordered list of (name, action_dict)
      safe_actions: list of action_dicts that passed rollout

    Returns:
      (fallback_name, action_dict, emergency_stop)
      emergency_stop=True iff no candidate passed rollout.
    """
    safe_set = set()
    for a in safe_actions:
        key = (round(a.get("linear_velocity", 0.0), 4),
               round(a.get("angular_velocity", 0.0), 4))
        safe_set.add(key)

    for name, action in candidate_actions:
        key = (round(action.get("linear_velocity", 0.0), 4),
               round(action.get("angular_velocity", 0.0), 4))
        if key in safe_set:
            return name, action, False

    # Emergency stop: no candidate passed
    return "emergency_stop", make_stop_action(), True
