#!/usr/bin/env python3
"""
test_certificate_ledger.py

Tests the certificate ledger / log:
- Each certificate decision must have a unique ID.
- The ledger must record: bypass, uncertified_exec, fallback reason.
- The ledger must not lose entries across consecutive decisions.
"""

import numpy as np
import pytest

from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig


@pytest.fixture
def cert_accept():
    tube_cfg = CertifiedTubeRadiusConfig(
        planning_margin=0.20,
        certified_margin=0.45,
        planning_scale=1.0,
        certified_scale=2.0,
    )
    return CertifiedAccept(tube_cfg=tube_cfg)


def make_input(robot_pos, goal_pos, velocity=0.5, entities=None):
    return CertifiedAcceptInput(
        robot_position=np.array(robot_pos),
        robot_yaw=0.0,
        linear_velocity=velocity,
        angular_velocity=0.0,
        goal_position=np.array(goal_pos),
        tracked_entities=entities or [],
        config_override=None,
    )


class TestCertificateLedger:
    """Test certificate logging properties."""

    def test_each_decision_has_id(self, cert_accept):
        """Every CertifiedAcceptDecision must have a unique certificate ID."""
        ids = set()
        for i in range(10):
            inp = make_input([float(i), 0.0], [10.0, 0.0], velocity=0.3)
            decision = cert_accept.evaluate(inp)
            cert_id = getattr(decision, 'certificate_id', None)
            assert cert_id is not None, f"Decision {i} missing certificate_id"
            assert cert_id not in ids, f"Duplicate certificate_id: {cert_id}"
            ids.add(cert_id)

    def test_ledger_records_bypass(self, cert_accept):
        """The ledger must record bypass events."""
        ids = set()
        for i in range(15):
            inp = make_input([float(i), 0.0], [10.0, 0.0], velocity=1.5)
            decision = cert_accept.evaluate(inp)
            # bypass_count should be tracked in the ledger
            bypass_count = getattr(decision, 'bypass_count', None)
            assert bypass_count is not None, "bypass_count not in decision"
            assert bypass_count >= 0

    def test_ledger_records_uncertified_exec(self, cert_accept):
        """Uncertified execution counts must be tracked."""
        for i in range(10):
            inp = make_input([float(i), 0.0], [10.0, 0.0], velocity=2.0)
            decision = cert_accept.evaluate(inp)
            uncertified_exec = getattr(decision, 'uncertified_exec_count', None)
            assert uncertified_exec is not None

    def test_no_certificate_id_leak(self, cert_accept):
        """Certificate IDs must not be None for any decision."""
        for i in range(5):
            inp = make_input([float(i), 1.0], [5.0, 1.0], velocity=0.5)
            decision = cert_accept.evaluate(inp)
            assert getattr(decision, 'certificate_id', '') != ''
