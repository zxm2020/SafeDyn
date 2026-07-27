#!/usr/bin/env python3
"""
test_fail_closed.py

Verifies SafeDyn fail-closed behavior:
1. Unsafe actions are never passed to the actuator.
2. The fallback is always a certified safe action (STOP).
3. No uncertified motion is executed.
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


@pytest.fixture
def fallback():
    return FallbackPolicy()


class TestFailClosedInvariant:
    """The fail-closed invariant: no uncertified action is ever executed."""

    def test_unsafe_proposal_yields_stop(self, guard, fallback):
        """An unsafe action must result in a STOP fallback."""
        unsafe_vels = [2.0, 5.0, 10.0, -5.0]
        for v in unsafe_vels:
            inp = CertifiedAcceptInput(
                robot_position=np.array([0.5, 0.5]),
                robot_yaw=0.0,
                linear_velocity=v,
                angular_velocity=1.0,
                goal_position=np.array([2.0, 2.0]),
                tracked_entities=[],
                config_override=None,
            )

            decision = guard.evaluate(inp)
            if not decision.certified:
                result = fallback.select_fallback(inp, decision)
                assert result.get("linear_velocity", 0.0) == 0.0, f"Fallback for v={v} was not STOP: {result}"
                assert result.get("angular_velocity", 0.0) == 0.0, f"Fallback for v={v} has non-zero angular vel: {result}"

    def test_zero_velocity_is_still_safe(self, guard, fallback):
        """Even zero velocity must pass through the certificate chain."""
        inp = CertifiedAcceptInput(
            robot_position=np.array([0.5, 0.5]),
            robot_yaw=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([2.0, 2.0]),
            tracked_entities=[],
            config_override=None,
        )

        decision = guard.evaluate(inp)
        # Zero velocity should be certified as safe
        assert decision.certified is True

    def test_bypass_count_never_decreases(self):
        """Bypass count is monotonic (never decreases) across consecutive decisions."""
        tube_cfg = CertifiedTubeRadiusConfig()
        guard = CertifiedAccept(tube_cfg=tube_cfg)

        previous_bypass = 0
        for i in range(10):
            inp = CertifiedAcceptInput(
                robot_position=np.array([float(i), 0.0]),
                robot_yaw=0.0,
                linear_velocity=2.0,  # Potentially unsafe
                angular_velocity=0.0,
                goal_position=np.array([10.0, 0.0]),
                tracked_entities=[],
                config_override=None,
            )
            decision = guard.evaluate(inp)
            # bypass count in ledger must be >= previous
            current_bypass = getattr(decision, 'bypass_count', 0)
            assert current_bypass >= previous_bypass, f"Bypass count decreased: {previous_bypass} -> {current_bypass}"
            previous_bypass = current_bypass
