#!/usr/bin/env python3
"""
run_method_demo.py

Runs a SafeDyn method demo from a YAML config.
Demonstrates the runtime authorization chain end-to-end.

Usage:
    python scripts/run_method_demo.py --config configs/safedyn_default.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
try:
    import yaml
except ImportError:
    yaml = None

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.safety.fallback import FallbackPolicy
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


def main():
    parser = argparse.ArgumentParser(description="SafeDyn method demo")
    parser.add_argument("--config", default="configs/safedyn_default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of demo steps")
    args = parser.parse_args()

    # Load config
    config = {}
    if yaml and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {args.config}")

    sd = config.get("safe_dyn", {})
    tube = sd.get("tube", {})

    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=tube.get("planning_margin", 0.20),
        certified_margin=tube.get("certified_margin", 0.45),
        planning_scale=tube.get("planning_scale", 1.0),
        certified_scale=tube.get("certified_scale", 2.0),
    )

    cert_accept = CertifiedAccept(tube_cfg=tube_cfg)
    fallback = FallbackPolicy()

    print(f"\n=== SafeDyn Method Demo ({args.steps} steps) ===\n")
    print(f"  B_plan = {tube_cfg.planning_margin}m, B_cert = {tube_cfg.certified_margin}m")
    print(f"  Fail-closed: {sd.get('fail_closed', True)}")
    print()

    robot_pos = np.array([0.0, 0.0])
    goal_pos = np.array([5.0, 2.0])
    certified_count = 0
    rejected_count = 0

    for step in range(args.steps):
        # Propose action: move toward goal (some noise after step 15)
        direction = goal_pos - robot_pos
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            direction = direction / dist

        if step >= 15:
            # Propose potentially unsafe action
            vel = 2.0
        else:
            vel = 0.5

        action = {"linear_velocity": vel, "angular_velocity": 0.0}

        inp = CertifiedAcceptInput(
            robot_position=robot_pos.copy(),
            robot_yaw=0.0,
            linear_velocity=action["linear_velocity"],
            angular_velocity=action["angular_velocity"],
            goal_position=goal_pos,
            tracked_entities=[],
            config_override=None,
        )

        decision = cert_accept.evaluate(inp)

        if decision.certified:
            robot_pos += direction * vel * 0.1
            certified_count += 1
            status = "CERTIFIED"
        else:
            rejected_count += 1
            fallback_action = fallback.select_fallback(inp, decision)
            status = f"REJECTED -> {fallback_action}"

        print(f"  Step {step:2d}: v={vel:.2f} | {status}")

    print(f"\n=== Demo Complete ===")
    print(f"  Certified: {certified_count} / Rejected: {rejected_count}")
    print(f"  Robot final position: {robot_pos.round(3)}")
    print(f"  bypass={rejected_count}, uncertified_exec=0")


if __name__ == "__main__":
    main()