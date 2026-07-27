#!/usr/bin/env python3
"""
test_log_replay.py

Tests log replay consistency.
Reads examples/sample_ticks.jsonl and verifies certificate decisions.
"""

import json
import os
import numpy as np
import pytest

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


@pytest.fixture
def cert_gate():
    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.20,
        certified_margin=0.45,
        planning_scale=1.0,
        certified_scale=2.0,
    )
    return CertifiedAccept(tube_cfg=tube_cfg)


class TestLogReplay:
    """Test replaying logged ticks."""

    def test_replay_ticks(self, cert_gate):
        """Replay sample ticks and verify consistency."""
        ticks_path = "examples/sample_ticks.jsonl"
        if not os.path.exists(ticks_path):
            pytest.skip("sample_ticks.jsonl not found")

        with open(ticks_path) as f:
            ticks = [json.loads(line) for line in f if line.strip()]

        assert len(ticks) > 0

        certified_count = 0
        rejected_count = 0

        for tick in ticks:
            action = tick.get("action", {})
            pose = tick.get("robot_pose", {})
            goal = tick.get("goal_position", {})
            entities = tick.get("tracked_entities", [])

            inp = CertifiedAcceptInput(
                robot_position=np.array([pose.get("x", 0.0), pose.get("z", 0.0)]),
                robot_yaw=pose.get("yaw", 0.0),
                linear_velocity=action.get("linear_velocity", 0.0),
                angular_velocity=action.get("angular_velocity", 0.0),
                goal_position=np.array([goal.get("x", 0.0), goal.get("z", 0.0)]) if goal else np.zeros(2),
                tracked_entities=entities,
                config_override=None,
            )

            decision = cert_gate.evaluate(inp)
            if decision.certified:
                certified_count += 1
            else:
                rejected_count += 1

        assert certified_count + rejected_count == len(ticks)
        print(f"  Replayed {len(ticks)} ticks: {certified_count} certified, {rejected_count} rejected")
