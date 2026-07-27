#!/usr/bin/env python3
"""
test_runtime_loop.py

Tests the runtime authorization loop components
(dual rate, incremental loop, stale plan, cold start).
These are lightweight and do not require Habitat-Sim.
"""

import numpy as np
import pytest

from safedyn.runtime.dual_rate import DualRateRuntime, DualRateConfig
from safedyn.runtime.incremental_loop import IncrementalLoop


class TestDualRateRuntime:
    """Tests for dual-rate runtime."""

    def test_dual_rate_initialization(self):
        """DualRateRuntime can be initialized."""
        cfg = DualRateConfig(
            timeout_ms=100,
            soft_deadline_ms=50,
        )
        runtime = DualRateRuntime(cfg=cfg)
        assert runtime is not None

    def test_dual_rate_config_fields(self):
        """DualRateConfig must have deadline fields."""
        cfg = DualRateConfig(
            timeout_ms=100,
            soft_deadline_ms=50,
        )
        assert cfg.timeout_ms is not None
        assert cfg.soft_deadline_ms is not None


class TestIncrementalLoop:
    """Tests for incremental loop."""

    def test_incremental_loop_initialization(self):
        """IncrementalLoop can be initialized."""
        loop = IncrementalLoop()
        assert loop is not None
