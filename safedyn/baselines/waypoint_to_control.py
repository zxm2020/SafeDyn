"""
policies/waypoint_to_control.py
Waypoint to Control Action Converter

Converts ETPNav waypoint proposals (position + heading + stop) to SafeDyn continuous control actions (v, w).
"""

import math
from typing import Dict, Any, Optional, Tuple


def normalize_heading_error(error: float) -> float:
    """Normalize heading error to [-pi, pi]."""
    while error > math.pi:
        error -= 2 * math.pi
    while error < -math.pi:
        error += 2 * math.pi
    return error


def waypoint_to_control(
    current_position: Tuple[float, float, float],
    current_heading: float,
    target_position: Tuple[float, float, float],
    target_heading: float,
    stop: bool,
    max_v: float = 0.5,
    max_w: float = 1.0,
    position_tolerance: float = 0.25,
    heading_gain: float = 1.0,
    distance_gain: float = 0.5,
) -> Dict[str, Any]:
    """
    Convert waypoint proposal to continuous control action.

    Args:
        current_position: (x, y, z) current position
        current_heading: current heading in radians
        target_position: (x, y, z) target waypoint position
        target_heading: target heading in radians
        stop: if True, stop at this waypoint
        max_v: maximum linear velocity
        max_w: maximum angular velocity
        position_tolerance: distance tolerance for considering waypoint reached
        heading_gain: gain for angular velocity control
        distance_gain: gain for linear velocity control

    Returns:
        Dict with:
            - v: linear velocity
            - w: angular velocity
            - stop: stop flag
            - conversion: "waypoint_to_unicycle"
            - target_position: original target
            - target_heading: original target heading
            - distance: distance to target
            - heading_error: heading error
    """
    # If stop is True, return zero velocity
    if stop:
        return {
            "v": 0.0,
            "w": 0.0,
            "stop": True,
            "conversion": "waypoint_to_unicycle",
            "target_position": list(target_position),
            "target_heading": target_heading,
            "distance": 0.0,
            "heading_error": 0.0,
        }

    # Calculate distance in xz plane (ignore y/height for navigation)
    dx = target_position[0] - current_position[0]
    dz = target_position[2] - current_position[2]
    distance = math.sqrt(dx * dx + dz * dz)

    # Calculate target angle (angle from current position to target)
    if distance > 0:
        target_angle = math.atan2(dz, dx)
    else:
        target_angle = current_heading

    # Calculate heading error (difference between current heading and target angle)
    heading_error = normalize_heading_error(target_angle - current_heading)

    # Compute control commands
    if distance < position_tolerance:
        # Close to target, focus on heading alignment
        v = 0.0
        w = clip(heading_gain * heading_error, -max_w, max_w)
    else:
        # Normal control
        v = clip(distance_gain * distance, 0.0, max_v)
        w = clip(heading_gain * heading_error, -max_w, max_w)

    return {
        "v": v,
        "w": w,
        "stop": False,
        "conversion": "waypoint_to_unicycle",
        "target_position": list(target_position),
        "target_heading": target_heading,
        "distance": distance,
        "heading_error": heading_error,
    }


def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def waypoint_record_to_control(
    record: Dict[str, Any],
    current_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a waypoint proposal record to control action.

    Args:
        record: Proposal record with waypoint field
        current_state: Current robot state (if None, uses waypoint as relative target)

    Returns:
        Control action dict compatible with SafeDyn
    """
    waypoint = record.get("waypoint", {})

    if not waypoint:
        return {
            "v": 0.0,
            "w": 0.0,
            "stop": True,
            "conversion": "waypoint_to_unicycle",
            "error": "no_waypoint_in_record",
        }

    target_position = waypoint.get("position", [0.0, 0.0, 0.0])
    target_heading = waypoint.get("heading", 0.0)
    stop = waypoint.get("stop", False)

    if current_state is None:
        # No current state, assume we're at origin and target is relative
        # This is a fallback - normally current_state should be provided
        current_position = (0.0, 0.0, 0.0)
        current_heading = 0.0
    else:
        current_position = current_state.get("position", (0.0, 0.0, 0.0))
        current_heading = current_state.get("heading", 0.0)

    return waypoint_to_control(
        current_position=current_position,
        current_heading=current_heading,
        target_position=tuple(target_position),
        target_heading=target_heading,
        stop=stop,
    )
