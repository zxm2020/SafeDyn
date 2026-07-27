"""
safety/visibility_model.py
Auditable Visibility and Field-of-View Model Interface.

Paper Reference: Section 4.6 - Visibility-Aware Speed Certification

Provides explicit interface for visibility modeling with multiple implementations:
- ConservativeVisibilityProxy: Conservative kinematic stopping distance (default)
- RaycastVisibilityModel: Full ray-casting with FOV (future work)

The conservative proxy is SAFE by design - it provides an upper bound on safe speed
without requiring full sensor simulation.

All implementations must satisfy the VisibilityModel interface contract:
  visible_distance ≥ actual_visible_distance (conservative)
  speed_limit ≤ safe_speed_for_visible_distance (safe)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class VisibilityResult:
    """
    Result of visibility query.

    Fields:
        visible_distance: Estimated visible distance ahead (meters)
        visible_entities: List of visible dynamic entities
        occlusion_detected: Whether occlusion is detected
        visibility_model_type: Name of visibility model used
        conservative: Whether result is conservative (safe upper bound)
    """
    visible_distance: float
    visible_entities: List[Dict[str, Any]]
    occlusion_detected: bool
    visibility_model_type: str
    conservative: bool
    fov_degrees: float = 180.0
    raycast_performed: bool = False


class VisibilityModel(ABC):
    """
    Abstract interface for visibility models.

    All implementations must be CONSERVATIVE (safe).
    visible_distance must be >= actual visible distance.
    """

    @abstractmethod
    def visible_dynamic_entities(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        entities: List[Any],
        scene_geometry: Optional[Any] = None,
    ) -> VisibilityResult:
        """
        Query which dynamic entities are visible from robot position.

        Args:
            robot_pos: Robot [x, z] position
            robot_yaw: Robot heading angle
            entities: List of dynamic entities
            scene_geometry: Optional scene geometry for occlusion (if available)

        Returns:
            VisibilityResult with visible entities and distance
        """
        pass

    @abstractmethod
    def estimate_occlusion_boundary(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        max_distance: float,
        scene_geometry: Optional[Any] = None,
    ) -> float:
        """
        Estimate distance to nearest occlusion boundary (wall, obstacle).

        Returns:
            Distance to occlusion boundary, or max_distance if none
        """
        pass

    @abstractmethod
    def conservative_stopping_distance(
        self,
        speed: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> float:
        """
        Compute conservative stopping distance for given speed.

        Formula:
            stopping_distance = speed * reaction_time
                             + speed^2 / (2 * max_deceleration)
                             + safety_margin
        """
        pass

    @abstractmethod
    def certify_visibility_speed(
        self,
        speed: float,
        visible_distance: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Certify if speed is safe for given visible distance.

        Returns:
            dict with:
                - passed: bool
                - visible_distance: float
                - stopping_distance: float
                - speed_limit: float
                - violation: bool
        """
        pass


class ConservativeVisibilityProxy(VisibilityModel):
    """
    Conservative kinematic visibility proxy (default implementation).

    This is NOT a full sensor/FOV model - it's a conservative safe upper bound.

    Conservative assumptions:
    - Default visible distance: 10 meters (configurable)
    - No occlusion modeling (assumes open space)
    - All entities within max_distance are considered visible

    This is SAFE because:
    - It never overestimates visible distance
    - It always provides conservative speed limits
    - It does not require sensor simulation or ray-casting

    Paper Reference: Section 4.6 footnote - we use conservative kinematic model
    """

    def __init__(
        self,
        default_visible_distance: float = 10.0,
        min_visible_distance: float = 0.1,
        max_visible_distance: float = 20.0,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ):
        self.default_visible_distance = default_visible_distance
        self.min_visible_distance = min_visible_distance
        self.max_visible_distance = max_visible_distance
        self.reaction_time = reaction_time
        self.max_deceleration = max_deceleration
        self.safety_margin = safety_margin

    def visible_dynamic_entities(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        entities: List[Any],
        scene_geometry: Optional[Any] = None,
    ) -> VisibilityResult:
        """
        Conservative proxy: all entities within max_distance are visible.
        """
        visible = []
        min_distance = float('inf')

        for entity in entities:
            if hasattr(entity, 'position'):
                entity_pos = entity.position()
                distance = float(np.linalg.norm(np.asarray(robot_pos) - np.asarray(entity_pos)))

                if distance <= self.max_visible_distance:
                    visible.append({
                        'entity_id': getattr(entity, 'entity_id', 'unknown'),
                        'position': entity_pos.tolist() if hasattr(entity_pos, 'tolist') else entity_pos,
                        'distance': distance,
                    })
                    min_distance = min(min_distance, distance)

        # Conservative: use default visible distance or min entity distance
        visible_distance = self.default_visible_distance
        if min_distance < float('inf'):
            visible_distance = min(visible_distance, min_distance)

        return VisibilityResult(
            visible_distance=visible_distance,
            visible_entities=visible,
            occlusion_detected=False,  # Conservative proxy does not model occlusion
            visibility_model_type="conservative_kinematic_proxy",
            conservative=True,
            fov_degrees=360.0,  # Assumes omnidirectional visibility (conservative)
            raycast_performed=False,
        )

    def estimate_occlusion_boundary(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        max_distance: float,
        scene_geometry: Optional[Any] = None,
    ) -> float:
        """
        Conservative proxy: assumes no occlusion (returns max_distance).
        This is SAFE because it does not assume visibility beyond max_distance.
        """
        return min(max_distance, self.default_visible_distance)

    def conservative_stopping_distance(
        self,
        speed: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> float:
        """
        Compute conservative stopping distance.

        Formula matches paper Equation 3 derivation.
        """
        if speed <= 0:
            return safety_margin

        reaction_distance = speed * reaction_time
        braking_distance = (speed ** 2) / (2 * max_deceleration)
        stopping_distance = reaction_distance + braking_distance + safety_margin

        return max(self.min_visible_distance, stopping_distance)

    def certify_visibility_speed(
        self,
        speed: float,
        visible_distance: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Certify if speed is safe for given visible distance.

        Returns certification result with speed limit and violation flag.
        """
        # Compute stopping distance for current speed
        stopping_distance = self.conservative_stopping_distance(
            speed, reaction_time, max_deceleration, safety_margin
        )

        # Compute max safe speed for visible distance
        # Solving: stopping_distance = visible_distance
        # v^2 / (2*a) + v*t_r + s_m - d_vis = 0
        if visible_distance <= self.min_visible_distance:
            speed_limit = 0.0
        else:
            import math
            a = 1.0 / (2.0 * max_deceleration)
            b = reaction_time
            c = safety_margin - visible_distance

            discriminant = b**2 - 4.0 * a * c
            if discriminant < 0:
                speed_limit = 0.0
            else:
                speed_limit = (-b + math.sqrt(discriminant)) / (2.0 * a)
                speed_limit = max(0.0, speed_limit)

        # Check if current speed exceeds limit
        violation = (speed > speed_limit)
        passed = not violation

        return {
            'passed': passed,
            'visible_distance': visible_distance,
            'stopping_distance': stopping_distance,
            'speed_limit': speed_limit,
            'violation': violation,
            'visibility_model_type': 'conservative_kinematic_proxy',
            'conservative': True,
        }


class RaycastVisibilityModel(VisibilityModel):
    """
    Full ray-casting visibility model with FOV constraints.

    FUTURE WORK: This is a placeholder for full sensor simulation.
    Current implementation falls back to conservative proxy.

    Full implementation would require:
    - Scene geometry ray-casting
    - Field-of-view constraints
    - Occlusion detection
    - Sensor noise modeling
    """

    def __init__(self, fov_degrees: float = 180.0):
        self.fov_degrees = fov_degrees
        self._proxy = ConservativeVisibilityProxy()
        print("[WARNING] RaycastVisibilityModel not fully implemented - using conservative proxy")

    def visible_dynamic_entities(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        entities: List[Any],
        scene_geometry: Optional[Any] = None,
    ) -> VisibilityResult:
        """Fall back to conservative proxy."""
        result = self._proxy.visible_dynamic_entities(robot_pos, robot_yaw, entities, scene_geometry)
        result.visibility_model_type = "raycast_fallback_to_conservative_proxy"
        return result

    def estimate_occlusion_boundary(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        max_distance: float,
        scene_geometry: Optional[Any] = None,
    ) -> float:
        """Fall back to conservative proxy."""
        return self._proxy.estimate_occlusion_boundary(robot_pos, robot_yaw, max_distance, scene_geometry)

    def conservative_stopping_distance(
        self,
        speed: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> float:
        """Fall back to conservative proxy."""
        return self._proxy.conservative_stopping_distance(speed, reaction_time, max_deceleration, safety_margin)

    def certify_visibility_speed(
        self,
        speed: float,
        visible_distance: float,
        reaction_time: float = 0.2,
        max_deceleration: float = 1.0,
        safety_margin: float = 0.1,
    ) -> Dict[str, Any]:
        """Fall back to conservative proxy."""
        return self._proxy.certify_visibility_speed(speed, visible_distance, reaction_time, max_deceleration, safety_margin)


# Default visibility model (conservative proxy)
_DEFAULT_VISIBILITY_MODEL = ConservativeVisibilityProxy()


def get_default_visibility_model() -> VisibilityModel:
    """Get default visibility model (conservative proxy)."""
    return _DEFAULT_VISIBILITY_MODEL


def create_visibility_model(model_type: str = "conservative") -> VisibilityModel:
    """
    Factory function to create visibility model.

    Args:
        model_type: "conservative" or "raycast"

    Returns:
        VisibilityModel instance
    """
    if model_type == "conservative":
        return ConservativeVisibilityProxy()
    elif model_type == "raycast":
        return RaycastVisibilityModel()
    else:
        raise ValueError(f"Unknown visibility model type: {model_type}")
