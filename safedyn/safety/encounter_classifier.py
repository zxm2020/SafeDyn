"""
safety/encounter_classifier.py
Phase 3C: Encounter-type-aware classification for backup scoring.
Coordinate system: x-z-yaw.

Classifies robot-entity encounters into types:
- head_on: entity approaching from front
- crossing_left_to_right: entity crossing from left to right
- crossing_right_to_left: entity crossing from right to left
- opposite_crossing: entities moving in opposite directions, paths cross
- overtaking: entity approaching from behind
- static_or_slow_blocking: entity stationary or slow in path
- dense_uncertain: multiple entities or uncertain classification
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EncounterClassification:
    """Result of encounter classification."""
    encounter_type: str
    relative_bearing: float  # radians, angle to entity in robot frame
    relative_velocity: float  # m/s, relative speed
    closing_speed: float  # m/s, positive = closing
    crossing_direction: str  # "left_to_right", "right_to_left", "none"
    confidence: float  # 0-1
    distance: float  # m, current distance to entity
    time_to_collision: float  # s, estimated TTC


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def classify_encounter(
    robot_pos: np.ndarray,
    robot_yaw: float,
    robot_vel: np.ndarray,
    entity_pos: np.ndarray,
    entity_vel: np.ndarray,
    robot_radius: float = 0.25,
    entity_radius: float = 0.25,
) -> EncounterClassification:
    """
    Classify the encounter type between robot and entity.

    Returns EncounterClassification with type and geometric parameters.
    """
    # Vector from robot to entity
    to_entity = np.asarray(entity_pos) - np.asarray(robot_pos)
    distance = float(np.linalg.norm(to_entity))

    if distance < 1e-6:
        return EncounterClassification(
            encounter_type="dense_uncertain",
            relative_bearing=0.0,
            relative_velocity=0.0,
            closing_speed=0.0,
            crossing_direction="none",
            confidence=0.0,
            distance=0.0,
            time_to_collision=float("inf"),
        )

    # Unit direction to entity
    direction = to_entity / distance

    # Relative bearing in robot frame (yaw=0 is -z direction)
    # angle_to_entity: 0 = front, pi/2 = left, -pi/2 = right, pi = behind
    angle_to_entity = np.arctan2(direction[0], -direction[1])
    relative_bearing = normalize_angle(angle_to_entity - robot_yaw)

    # Relative velocity
    rel_vel = np.asarray(entity_vel) - np.asarray(robot_vel)
    relative_velocity = float(np.linalg.norm(rel_vel))

    # Closing speed (positive = closing)
    closing_speed = -np.dot(rel_vel, direction)

    # Time to collision (if closing)
    if closing_speed > 0.1:
        time_to_collision = (distance - robot_radius - entity_radius) / closing_speed
    else:
        time_to_collision = float("inf")

    # Crossing direction detection
    # Project relative velocity onto perpendicular direction
    perp_direction = np.array([direction[1], -direction[0]])  # perpendicular
    lateral_velocity = np.dot(rel_vel, perp_direction)

    if lateral_velocity > 0.3:
        crossing_direction = "left_to_right"
    elif lateral_velocity < -0.3:
        crossing_direction = "right_to_left"
    else:
        crossing_direction = "none"

    # Classify encounter type
    abs_bearing = abs(relative_bearing)
    entity_speed = float(np.linalg.norm(entity_vel))
    robot_speed = float(np.linalg.norm(robot_vel))

    # Head-on: entity in front (±60 deg), closing, significant speed
    if abs_bearing < np.pi / 3 and closing_speed > 0.3 and entity_speed > 0.3:
        encounter_type = "head_on"
        confidence = 0.9

    # Overtaking: entity behind (±60 deg), closing from behind
    elif abs_bearing > 2 * np.pi / 3 and closing_speed > 0.1:
        encounter_type = "overtaking"
        confidence = 0.8

    # Crossing: entity to side (60-120 deg), lateral motion
    elif np.pi / 3 < abs_bearing < 2 * np.pi / 3 and crossing_direction != "none":
        if crossing_direction == "left_to_right":
            encounter_type = "crossing_left_to_right"
        else:
            encounter_type = "crossing_right_to_left"
        confidence = 0.85

    # Opposite crossing: paths cross with opposite direction motion
    elif abs_bearing < np.pi / 2 and closing_speed > 0.2:
        # Check if velocities are roughly opposite
        robot_dir = robot_vel / (robot_speed + 1e-6)
        entity_dir = entity_vel / (entity_speed + 1e-6)
        dot_product = np.dot(robot_dir, entity_dir)
        if dot_product < -0.5:  # roughly opposite
            encounter_type = "opposite_crossing"
            confidence = 0.8
        else:
            encounter_type = "crossing"
            confidence = 0.7

    # Static blocking: entity slow/stationary in path
    elif entity_speed < 0.2 and abs_bearing < np.pi / 2:
        encounter_type = "static_or_slow_blocking"
        confidence = 0.75

    # Default
    else:
        encounter_type = "crossing"
        confidence = 0.6

    return EncounterClassification(
        encounter_type=encounter_type,
        relative_bearing=relative_bearing,
        relative_velocity=relative_velocity,
        closing_speed=closing_speed,
        crossing_direction=crossing_direction,
        confidence=confidence,
        distance=distance,
        time_to_collision=time_to_collision,
    )


def get_encounter_scoring_weights(encounter_type: str) -> Dict[str, float]:
    """
    Get scoring weights for different encounter types.

    Returns dict of weight adjustments for backup scoring.
    """
    weights = {
        "head_on": {
            "forward_penalty": 8.0,
            "reverse_bonus": 2.0,
            "turn_bonus": 1.0,
            "stop_penalty": 3.0,  # stop may be hit
            "clearance_weight": 12.0,
            "ttc_weight": 3.0,
        },
        "crossing_left_to_right": {
            "forward_penalty": 2.0,
            "reverse_penalty": 2.0,  # don't over-prefer reverse
            "turn_left_bonus": 1.5,  # yield to left
            "turn_right_penalty": 1.0,
            "stop_bonus": 1.0,  # stop may be OK
            "clearance_weight": 10.0,
            "ttc_weight": 2.0,
        },
        "crossing_right_to_left": {
            "forward_penalty": 2.0,
            "reverse_penalty": 2.0,
            "turn_right_bonus": 1.5,  # yield to right
            "turn_left_penalty": 1.0,
            "stop_bonus": 1.0,
            "clearance_weight": 10.0,
            "ttc_weight": 2.0,
        },
        "opposite_crossing": {
            "forward_penalty": 4.0,
            "reverse_bonus": 1.0,
            "turn_bonus": 2.0,
            "stop_bonus": 0.5,
            "side_consistency_weight": 3.0,  # avoid oscillation
            "clearance_weight": 11.0,
            "ttc_weight": 2.5,
        },
        "overtaking": {
            "forward_bonus": 1.0,  # accelerate away
            "reverse_penalty": 3.0,  # don't reverse when overtaken
            "clearance_weight": 8.0,
            "ttc_weight": 1.5,
        },
        "static_or_slow_blocking": {
            "forward_penalty": 1.0,
            "reverse_bonus": 1.0,
            "turn_bonus": 2.0,
            "stop_bonus": 2.0,  # stop is fine
            "clearance_weight": 10.0,
            "ttc_weight": 2.0,
        },
        "dense_uncertain": {
            "forward_penalty": 3.0,
            "reverse_bonus": 1.5,
            "turn_bonus": 1.0,
            "stop_bonus": 1.0,
            "clearance_weight": 15.0,  # prioritize clearance
            "ttc_weight": 3.0,
        },
        "crossing": {
            "forward_penalty": 2.0,
            "reverse_penalty": 1.0,
            "turn_bonus": 1.0,
            "stop_bonus": 1.0,
            "clearance_weight": 10.0,
            "ttc_weight": 2.0,
        },
    }

    return weights.get(encounter_type, weights["crossing"])
