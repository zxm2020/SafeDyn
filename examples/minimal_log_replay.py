#!/usr/bin/env python3
"""
minimal_log_replay.py

Replay logged tick data (JSONL) to demonstrate SafeDyn
certificate chain evaluation without live Habitat execution.

Usage:
    python examples/minimal_log_replay.py --ticks examples/sample_ticks.jsonl

The input JSONL should have one line per tick with:
- robot_pose (x, z, yaw)
- action (linear_velocity, angular_velocity)
- optional: obstacle_entities, certificate_result

If no sample ticks file exists, the script generates synthetic ticks
and demonstrates the replay pipeline.
"""

import json
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safedyn.safety.certified_accept import CertifiedAcceptInput, CertifiedAcceptDecision
from safedyn.safety.certified_accept import CertifiedAccept
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


def generate_synthetic_ticks(n_ticks=50):
    """Generate synthetic tick data for demonstration."""
    ticks = []
    robot_pos = np.array([0.0, 0.0])
    goal = np.array([5.0, 2.0])

    for i in range(n_ticks):
        # Gradually move toward goal with some noise
        direction = goal - robot_pos
        if np.linalg.norm(direction) > 0.01:
            direction = direction / np.linalg.norm(direction)
        noise = np.random.randn(2) * 0.3
        vel = direction * 0.8 + noise * 0.2

        # Occasionally propose an unsafe action (too fast, off-course)
        if i % 15 == 0:
            vel = direction * 2.0  # Unnecessarily fast

        tick = {
            "tick_id": i,
            "robot_pose": {
                "x": float(robot_pos[0]),
                "z": float(robot_pos[1]),
                "yaw": 0.0,
            },
            "action": {
                "linear_velocity": float(vel[0]),
                "angular_velocity": float(noise[0]),
            },
            "goal_position": {
                "x": float(goal[0]),
                "z": float(goal[1]),
            },
            "tracked_entities": [],
        }
        ticks.append(tick)

        # Update robot position
        robot_pos += np.array([vel[0] * 0.1, vel[1] * 0.1])
        goal_pos = np.array([goal[0] - robot_pos[0], goal[1] - robot_pos[1]])

    return ticks


def replay_ticks(ticks_path):
    """Replay tick data through SafeDyn certificate chain."""
    if not os.path.exists(ticks_path):
        print(f"Tick file not found: {ticks_path}")
        print("Generating synthetic ticks instead...")
        ticks = generate_synthetic_ticks()
    else:
        ticks = []
        with open(ticks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ticks.append(json.loads(line))

    print(f"Replaying {len(ticks)} ticks through SafeDyn certificate chain\n")

    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.20,
        certified_margin=0.45,
    )
    cert_gate = CertifiedAccept(tube_cfg=tube_cfg)

    certified_count = 0
    rejected_count = 0
    fallback_count = 0

    for tick in ticks:
        action = tick.get("action", {})
        pose = tick.get("robot_pose", {})
        goal = tick.get("goal_position", {})
        entities = tick.get("tracked_entities", [])

        input_data = CertifiedAcceptInput(
            robot_position=np.array([pose["x"], pose["z"]]),
            robot_yaw=pose.get("yaw", 0.0),
            linear_velocity=action.get("linear_velocity", 0.0),
            angular_velocity=action.get("angular_velocity", 0.0),
            goal_position=np.array([goal["x"], goal["z"]]) if goal else np.zeros(2),
            tracked_entities=entities,
            config_override=None,
        )

        decision = cert_gate.evaluate(input_data)

        if decision.certified:
            certified_count += 1
        else:
            rejected_count += 1
            if decision.fallback_reason:
                fallback_count += 1

        print(f"  Tick {tick['tick_id']:3d}: "
              f"certified={decision.certified:5s}  "
              f"preliminary={decision.preliminary:5s}  "
              f"reason={str(decision.fallback_reason)[:60] if decision.fallback_reason else 'none':>60s}")

    print(f"\n=== Results ===")
    print(f"  Total ticks:       {len(ticks)}")
    print(f"  Certified (pass):  {certified_count}")
    print(f"  Rejected (fallback): {rejected_count}")
    print(f"  Explicit fallback: {fallback_count}")
    print(f"  Pass rate: {certified_count/len(ticks)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Replay logged ticks through SafeDyn certificate chain"
    )
    parser.add_argument(
        "--ticks",
        default="examples/sample_ticks.jsonl",
        help="Path to JSONL tick file",
    )
    args = parser.parse_args()
    replay_ticks(args.ticks)


if __name__ == "__main__":
    main()
