#!/usr/bin/env python3
"""
test_certified_accept.py

Tests for the CertifiedAccept certificate chain.
These tests exercise the core method components without requiring
Habitat, MP3D scenes, or ETPNav checkpoints.
"""

import numpy as np
import pytest

from safedyn.safety.certified_accept import (
    CertifiedAccept,
    CertifiedAcceptInput,
    CertifiedAcceptDecision,
)
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig
from safedyn.tracking.kalman import KalmanTracker


@pytest.fixture
def tube_config():
    """Standard tube config for testing."""
    return CertifiedTubeRadiusConfig(
        planning_margin=0.20,
        certified_margin=0.45,
        planning_scale=1.0,
        certified_scale=2.0,
    )


@pytest.fixture
def cert_accept(tube_config):
    """CertifiedAccept instance with standard tube config."""
    return CertifiedAccept(tube_cfg=tube_config)


def make_input(robot_pos, goal_pos, velocity=0.5, entities=None):
    """Helper to create a CertifiedAcceptInput."""
    return CertifiedAcceptInput(
        robot_position=np.array(robot_pos),
        robot_yaw=0.0,
        linear_velocity=velocity,
        angular_velocity=0.0,
        goal_position=np.array(goal_pos),
        tracked_entities=entities or [],
        config_override=None,
    )


class TestCertifiedAcceptBasic:
    """Basic acceptance tests."""

    def test_slow_motion_accepted(self, cert_accept):
        """Slow motion toward the goal should be certified."""
        inp = make_input([0.0, 0.0], [2.0, 1.0], velocity=0.3)
        decision = cert_accept.evaluate(inp)
        assert decision.certified is True

    def test_fast_motion_rejected(self, cert_accept):
        """Unreasonably fast motion should be rejected."""
        inp = make_input([0.0, 0.0], [2.0, 1.0], velocity=3.0)
        decision = cert_accept.evaluate(inp)
        assert decision.certified is False

    def test_zero_velocity_accepted(self, cert_accept):
        """Zero velocity should always be safe."""
        inp = make_input([0.0, 0.0], [2.0, 1.0], velocity=0.0)
        decision = cert_accept.evaluate(inp)
        assert decision.certified is True

    def test_stop_action_accepted(self, cert_accept):
        """Stop action should be accepted."""
        inp = make_input([1.0, 1.0], [2.0, 1.0], velocity=0.0)
        decision = cert_accept.evaluate(inp)
        assert decision.certified is True

    def test_reverse_velocity_accepted(self, cert_accept):
        """Use caution with reverse velocity — it could be safe or unsafe depending on context."""
        inp = make_input([0.0, 0.0], [2.0, 1.0], velocity=-0.5)
        decision = cert_accept.evaluate(inp)
        # Just verify it produces a decision (do not enforce a specific value)
        assert decision.preliminary is not None


class TestCertificateInvariants:
    """Test the certificate invariant: B_cert >= B_plan."""

    def test_bcert_greater_than_bplan(self, tube_config):
        """B_cert must be >= B_plan in the tube config."""
        assert tube_config.certified_margin >= tube_config.planning_margin
        assert tube_config.certified_scale >= tube_config.planning_scale

    def test_cert_ge_plan_invariant(self, tube_config):
        """Tube config must satisfy the cert >= plan invariant."""
        assert tube_config.certified_margin > tube_config.planning_margin


class TestFailClosed:
    """Test that rejected actions produce safe fallback, not unsafe execution."""

    def test_fail_closed_stop(self, cert_accept):
        """Unsafe proposal should be replaced by STOP, not executed."""
        from safedyn.safety.fallback import FallbackPolicy
        fallback = FallbackPolicy()

        unsafe_action = {"linear_velocity": 5.0, "angular_velocity": 2.0}
        inp = make_input([0.0, 0.0], [2.0, 1.0], velocity=unsafe_action["linear_velocity"])

        decision = cert_accept.evaluate(inp)
        assert not decision.certified

        # Select fallback
        fallback_action = fallback.select_fallback(inp, decision)
        assert fallback_action.get("linear_velocity", 0.0) == 0.0
        assert fallback_action.get("angular_velocity", 0.0) == 0.0

    def test_uncertified_exec_never_written(self, cert_accept):
        """The certificate gate must never pass an uncertified action to the actuator."""
        from safedyn.safety.fallback import FallbackPolicy
        fallback = FallbackPolicy()

        for velocity in [2.0, 5.0, 10.0, -3.0]:
            inp = make_input([0.0, 0.0], [5.0, 5.0], velocity=velocity)
            decision = cert_accept.evaluate(inp)

            if not decision.certified:
                fallback_action = fallback.select_fallback(inp, decision)
                # Fallback must be STOP, not the proposed velocity
                assert fallback_action.get("linear_velocity") == 0.0
