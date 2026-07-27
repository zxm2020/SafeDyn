#!/usr/bin/env python3
"""
minimal_runtime_authorization.py

Demonstrates the SafeDyn runtime authorization chain
without requiring Habitat scenes or trained model weights.

This example exercises the certificate gate end-to-end:
1. Build a CertifiedAccept gate
2. Propose actions (some safe, some unsafe)
3. Observe which actions pass/fail certification
4. Observe the fallback chain for rejected actions
"""

import numpy as np
import sys
import os

# Allow running from the repo root or release root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safedyn.safety.certified_accept import (
    CertifiedAccept,
    CertifiedAcceptInput,
    CertifiedAcceptDecision,
)
from safedyn.safety.fallback import FallbackPolicy
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig
from safedyn.tracking.kalman import KalmanTracker


def demo():
    print("=== SafeDyn Minimal Runtime Authorization Demo ===\n")

    # 1. Set up the components
    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.20,
        certified_margin=0.45,
        planning_scale=1.0,
        certified_scale=2.0,
    )

    certified_accept = CertifiedAccept(tube_cfg=tube_cfg)
    fallback = FallbackPolicy()
    tracker = KalmanTracker()

    print("Components initialized:")
    print(f"  CertifiedTubeRadiusConfig: B_plan={tube_cfg.planning_margin}m, B_cert={tube_cfg.certified_margin}m")
    print()

    # 2. Run a short simulated episode
    robot_pos = np.array([0.0, 0.0])  # x-z plane
    goal_pos = np.array([2.0, 1.0])

    # Propose safe and unsafe actions
    proposed_actions = [
        {"linear_velocity": 1.0, "angular_velocity": 0.0},   # safe: moving toward goal
        {"linear_velocity": 1.5, "angular_velocity": 0.0},   # unsafe: too fast
        {"linear_velocity": 0.0,  "angular_velocity": 0.0},   # safe: stationary
    ]

    for i, action in enumerate(proposed_actions):
        input_data = CertifiedAcceptInput(
            robot_position=robot_pos,
            robot_yaw=0.0,
            linear_velocity=action["linear_velocity"],
            angular_velocity=action["angular_velocity"],
            goal_position=goal_pos,
            tracked_entities=[],
            config_override=None,
        )

        decision = certified_accept.evaluate(input_data)

        print(f"  Action {i+1}: v={action['linear_velocity']}, w={action['angular_velocity']}")
        print(f"    certified={decision.certified}, "
              f"preliminary={decision.preliminary}, "
              f"fallback_reason={decision.fallback_reason}")

        if not decision.certified:
            fallback_action = fallback.select_fallback(input_data, decision)
            print(f"    --> Fallback: {fallback_action}")
        else:
            robot_pos += np.array([action["linear_velocity"] * 0.1, 0.0])
            print(f"    --> Action executed; new robot pos: {robot_pos.round(3)}")
        print()

    print("=== Demo complete ===")


if __name__ == "__main__":
    demo()
