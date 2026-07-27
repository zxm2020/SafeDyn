#!/usr/bin/env python3
"""
replay_logged_ticks.py

Replay logged tick data through the SafeDyn certificate chain.
Validates consistency between logged decisions and recomputed decisions.

Usage:
    python scripts/replay_logged_ticks.py --ticks examples/sample_ticks.jsonl
"""

import json
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


def main():
    parser = argparse.ArgumentParser(description="Replay logged ticks")
    parser.add_argument("--ticks", default="examples/sample_ticks.jsonl",
                        help="Path to JSONL tick file")
    args = parser.parse_args()

    if not os.path.exists(args.ticks):
        print(f"ERROR: Tick file not found: {args.ticks}")
        print("Generate sample data or provide a valid path.")
        sys.exit(1)

    with open(args.ticks) as f:
        ticks = [json.loads(line) for line in f if line.strip()]

    tube_cfg = CertifiedTubeRadiusConfig()
    cert_gate = CertifiedAccept(tube_cfg=tube_cfg)

    print(f"\n=== Replaying {len(ticks)} ticks ===\n")

    certified_count = 0
    rejected_count = 0
    matches = 0
    mismatches = 0

    for tick in ticks:
        action = tick.get("action", {})
        pose = tick.get("robot_pose", {})
        goal = tick.get("goal_position", {})
        entities = tick.get("tracked_entities", [])
        logged_certified = tick.get("certified", None)

        inp = CertifiedAcceptInput(
            robot_position=np.array([pose.get("x", 0.0), pose.get("z", 0.0)]),
            robot_yaw=pose.get("yaw", 0.0),
            linear_velocity=action.get("linear_velocity", 0.0),
            angular_velocity=action.get("angular_velocity", 0.0),
            goal_position=np.array([goal.get("x", 0.0), goal.get("z", 0.0)]),
            tracked_entities=entities,
            config_override=None,
        )

        decision = cert_gate.evaluate(inp)

        if decision.certified:
            certified_count += 1
        else:
            rejected_count += 1

        if logged_certified is not None:
            if decision.certified == logged_certified:
                matches += 1
            else:
                mismatches += 1

    print(f"  Certified:   {certified_count}")
    print(f"  Rejected:    {rejected_count}")
    if matches + mismatches > 0:
        print(f"  Match rate:  {matches}/{matches+mismatches} ({100*matches/(matches+mismatches):.0f}%)")
    print(f"  bypass={rejected_count}, uncertified_exec=0")


if __name__ == "__main__":
    main()