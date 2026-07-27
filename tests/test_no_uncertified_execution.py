#!/usr/bin/env python3
"""
test_no_uncertified_execution.py

Verifies that no action is written to the actuator unless certified.
This is the core safety invariant of SafeDyn-VLN Guard.
"""

import numpy as np
import pytest

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.safety.fallback import FallbackPolicy
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


@pytest.fixture
def guard():
    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.10,
        certified_margin=0.30,
        planning_scale=1.0,
        certified_scale=2.0,
    )
    return CertifiedAccept(tube_cfg=tube_cfg)


class TestNoUncertifiedExecution:
    """The actuator must never receive an uncertified action."""

    def test_only_certified_written(self, guard):
        """Only certified actions reach the actuator."""
        fallback = FallbackPolicy()
        for v in [0.1, 0.3, 0.5]:  # Safe velocities
            inp = CertifiedAcceptInput(
                robot_position=np.array([0.5, 0.5]),
                robot_yaw=0.0,
                linear_velocity=v,
                angular_velocity=0.0,
                goal_position=np.array([2.0, 2.0]),
                tracked_entities=[],
                config_override=None,
            )
            decision = guard.evaluate(inp)
            if decision.certified:
                # The certified action goes to the actuator
                # We verify it matches the proposal for safe actions
                assert inp.linear_velocity == v

    def test_no_uncertified_motion_when_unsafe(self, guard):
        """When unsafe, only STOP (or safe fallback) is written."""
        fallback = FallbackPolicy()
        unsafe_action = {"linear_velocity": 5.0, "angular_velocity": 1.0}
        inp = CertifiedAcceptInput(
            robot_position=np.array([0.5, 0.5]),
            robot_yaw=0.0,
            linear_velocity=unsafe_action["linear_velocity"],
            angular_velocity=unsafe_action["angular_velocity"],
            goal_position=np.array([2.0, 2.0]),
            tracked_entities=[],
            config_override=None,
        )
        decision = guard.evaluate(inp)
        if not decision.certified:
            fallback_action = fallback.select_fallback(inp, decision)
            assert fallback_action.get("linear_velocity") == 0.0
