#!/usr/bin/env python3
"""
minimal_fail_closed_demo.py

Demonstrates SafeDyn fail-closed behavior:
- Propose an unsafe action
- SafeDyn rejects it
- Actuator receives STOP, not the unsafe motion
- Assert no uncertified motion is executed
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.safety.fallback import FallbackPolicy
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


def demo():
    print("=== SafeDyn Fail-Closed Demo ===")
    print()

    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.10,
        certified_margin=0.30,
        planning_scale=1.0,
        certified_scale=2.0,
    )

    cert_accept = CertifiedAccept(tube_cfg=tube_cfg)
    fallback = FallbackPolicy()

    unsafe_actions = [
        {"linear_velocity": 3.0, "angular_velocity": 0.5},
        {"linear_velocity": -2.0, "angular_velocity": 1.0},
    ]

    robot_pos = np.array([0.5, 0.5])
    goal_pos = np.array([2.0, 2.0])
    bypass_count = 0
    uncertified_exec_count = 0

    for i, action in enumerate(unsafe_actions):
        inp = CertifiedAcceptInput(
            robot_position=robot_pos,
            robot_yaw=0.0,
            linear_velocity=action["linear_velocity"],
            angular_velocity=action["angular_velocity"],
            goal_position=goal_pos,
            tracked_entities=[],
            config_override=None,
        )

        decision = cert_accept.evaluate(inp)
        print(f"  Trial {i+1}: propose v={action['linear_velocity']}, w={action['angular_velocity']}")

        if decision.certified:
            print("    UNEXPECTED: Action was certified! Must fail for these parameters.")
            sys.exit(1)
        else:
            bypass_count += 1
            fallback_action = fallback.select_fallback(inp, decision)
            print(f"    REJECTED: bypass={bypass_count}, fallback={fallback_action}")
            # Verify actuator received STOP (0 velocity)
            if fallback_action.get("linear_velocity", 0.0) != 0.0:
                print(f"    FAIL: Actuator executed non-zero motion: {fallback_action}")
                sys.exit(1)

    print()
    print("=== Fail-Closed Verified ===")
    print(f"  Proposals:  {len(unsafe_actions)}")
    print(f"  Bypass count: {bypass_count}")
    print(f"  No uncertified motion executed: True")
    print(f"  uncertified_exec={uncertified_exec_count}, bypass={bypass_count}")


if __name__ == "__main__":
    demo()
