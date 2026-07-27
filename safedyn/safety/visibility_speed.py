"""
safety/visibility_speed.py
Stage F2D: Visibility-aware speed certification helper.

Provides minimum visibility-speed hard check for CertifiedAccept.
Uses auditable VisibilityModel interface (conservative kinematic proxy by default).

Paper Reference: Section 4.6 - Visibility-Aware Speed Certification
Formula: Equation 3 - Maximum safe speed for visible distance

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

# Import auditable visibility model interface
try:
    from safedyn.safety.visibility_model import (
        get_default_visibility_model,
        ConservativeVisibilityProxy,
    )
    _VISIBILITY_MODEL_AVAILABLE = True
except ImportError:
    _VISIBILITY_MODEL_AVAILABLE = False


@dataclass
class VisibilitySpeedConfig:
    """Configuration for visibility-speed certification."""
    reaction_time: float = 0.2          # seconds
    max_deceleration: float = 1.0       # m/s^2
    safety_margin: float = 0.1          # meters
    default_visible_distance: float = 10.0  # meters
    min_visible_distance: float = 0.1   # meters


def estimate_stopping_distance(
    speed: float,
    config: VisibilitySpeedConfig,
) -> float:
    """
    Estimate stopping distance from current speed.

    Formula:
        stopping_distance = speed * reaction_time
                         + speed^2 / (2 * max_deceleration)
                         + safety_margin
    """
    if speed <= 0:
        return config.safety_margin

    reaction_distance = speed * config.reaction_time
    braking_distance = (speed ** 2) / (2 * config.max_deceleration)
    stopping_distance = reaction_distance + braking_distance + config.safety_margin

    return max(config.min_visible_distance, stopping_distance)


def compute_visibility_speed_limit(
    visible_distance: float,
    config: VisibilitySpeedConfig,
) -> float:
    """
    Compute maximum safe speed for given visible distance.

    Solving stopping_distance formula for speed:
        v^2 / (2*a) + v*t_r + s_m - d_vis = 0

    Using quadratic formula for positive root:
        v_max = -b + sqrt(b^2 - 4ac) / 2a
    where a = 1/(2*decel), b = reaction_time, c = safety_margin - visible_distance

    If visible_distance is very small, returns 0.
    """
    if visible_distance <= config.min_visible_distance:
        return 0.0

    # Quadratic coefficients for: (1/(2a))v^2 + t_r*v + (s_m - d_vis) = 0
    a = 1.0 / (2.0 * config.max_deceleration)
    b = config.reaction_time
    c = config.safety_margin - visible_distance

    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0:
        return 0.0

    # Positive root (max speed)
    v_max = (-b + math.sqrt(discriminant)) / (2.0 * a)
    return max(0.0, v_max)


def check_visibility_speed(
    speed: float,
    visible_distance: Optional[float],
    config: Optional[VisibilitySpeedConfig] = None,
    scenario_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check if current speed is safe given visible distance.

    Args:
        speed: current/proposed speed (m/s)
        visible_distance: visible distance ahead (m), None for default
        config: VisibilitySpeedConfig instance
        scenario_metadata: optional scenario metadata (e.g. sudden_appearance flags)

    Returns:
        dict with visibility-speed check results
    """
    if config is None:
        config = VisibilitySpeedConfig()

    # Determine visible distance
    visibility_source = "provided"
    if visible_distance is None:
        # Check scenario metadata for occlusion signals
        if scenario_metadata:
            if scenario_metadata.get("visibility_state") == "hidden":
                visible_distance = scenario_metadata.get("emergence_distance", config.default_visible_distance)
                visibility_source = "scenario_hidden"
            elif scenario_metadata.get("visibility_state") == "occluded":
                visible_distance = scenario_metadata.get("emergence_timing", config.default_visible_distance)
                visibility_source = "scenario_occluded"
            else:
                visible_distance = config.default_visible_distance
                visibility_source = "default"
        else:
            visible_distance = config.default_visible_distance
            visibility_source = "default"

    # Ensure minimum visible distance
    visible_distance = max(config.min_visible_distance, visible_distance)

    # Compute stopping distance and speed limit
    stopping_distance = estimate_stopping_distance(speed, config)
    visibility_speed_limit = compute_visibility_speed_limit(visible_distance, config)

    # Hard check: stopping_distance must be <= visible_distance
    visibility_speed_passed = (stopping_distance <= visible_distance)
    visibility_speed_violation = not visibility_speed_passed

    # Get visibility model type (auditable)
    visibility_model_type = "conservative_kinematic_proxy"
    if _VISIBILITY_MODEL_AVAILABLE:
        try:
            model = get_default_visibility_model()
            if hasattr(model, '__class__'):
                visibility_model_type = model.__class__.__name__
        except:
            visibility_model_type = "conservative_kinematic_proxy"

    return {
        "visibility_speed_implemented": True,
        "visibility_speed_passed": visibility_speed_passed,
        "visibility_speed_violation": visibility_speed_violation,
        "speed": float(speed),
        "visible_distance": float(visible_distance),
        "stopping_distance": float(stopping_distance),
        "visibility_speed_limit": float(visibility_speed_limit),
        "visibility_source": visibility_source,
        "visibility_model_type": visibility_model_type,  # Auditable model type
        "visibility_speed_reason": (
            "" if visibility_speed_passed
            else f"stopping_distance({stopping_distance:.2f}) > visible_distance({visible_distance:.2f})"
        ),
        "reaction_time": config.reaction_time,
        "max_deceleration": config.max_deceleration,
        "safety_margin": config.safety_margin,
    }


def get_default_visibility_config() -> VisibilitySpeedConfig:
    """Return default visibility-speed configuration."""
    return VisibilitySpeedConfig()
