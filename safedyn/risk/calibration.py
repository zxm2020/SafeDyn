"""
risk/calibration.py
SafeDyn-VLN Guard: Tube calibration module.

Collects residuals from calibration data, estimates beta_cert from
quantile, freezes beta_cert after calibration, reports coverage.
Ensures learned risk cannot shrink B_cert.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationConfig:
    """Configuration for tube calibration."""
    quantile_level: float = 0.95     # coverage quantile target
    min_samples: int = 30            # minimum samples for calibration
    beta_min: float = 0.5            # minimum beta_cert
    beta_max: float = 5.0            # maximum beta_cert
    default_beta: float = 1.0        # default before calibration
    freeze_after_calibration: bool = True


@dataclass
class CalibrationState:
    """State of calibration process."""
    calibrated: bool = False
    frozen: bool = False
    beta_cert: float = 1.0
    residuals: List[float] = field(default_factory=list)
    coverage_achieved: float = 0.0
    estimated_margin: float = 0.0
    sample_count: int = 0


def collect_residual(
    state: CalibrationState,
    predicted_radius: float,
    observed_distance: float,
    entity_radius: float,
) -> None:
    """
    Collect a residual sample for calibration.

    Residual = predicted_radius - (observed_distance - entity_radius)
    Positive residual means prediction was conservative (good).
    """
    residual = predicted_radius - (observed_distance - entity_radius)
    state.residuals.append(float(residual))
    state.sample_count = len(state.residuals)


def estimate_beta_cert(
    state: CalibrationState,
    config: CalibrationConfig,
) -> float:
    """
    Estimate beta_cert from residual quantile.

    beta_cert = max(beta_min, quantile(residuals, quantile_level))

    This ensures the certified tube covers the specified quantile
    of observed residuals.
    """
    if state.sample_count < config.min_samples:
        return config.default_beta

    residuals = np.array(state.residuals)
    quantile_val = float(np.quantile(residuals, config.quantile_level))
    beta_cert = float(np.clip(quantile_val, config.beta_min, config.beta_max))

    # Compute coverage
    coverage = float(np.mean(residuals <= quantile_val))
    state.coverage_achieved = coverage
    state.estimated_margin = quantile_val

    return beta_cert


def freeze_calibration(
    state: CalibrationState,
    config: CalibrationConfig,
) -> float:
    """
    Freeze beta_cert after calibration.

    Once frozen, beta_cert cannot be changed (prevents learned risk
    from shrinking B_cert).
    """
    if state.sample_count >= config.min_samples:
        state.beta_cert = estimate_beta_cert(state, config)
        state.calibrated = True
        if config.freeze_after_calibration:
            state.frozen = True

    return state.beta_cert


def validate_beta_cert(
    state: CalibrationState,
    proposed_beta: float,
) -> Tuple[bool, str]:
    """
    Validate that proposed beta_cert does not shrink B_cert.

    If calibrated and frozen, proposed beta must be >= current beta_cert.
    """
    if state.frozen:
        if proposed_beta < state.beta_cert:
            return False, (
                f"proposed_beta={proposed_beta:.3f} < "
                f"frozen_beta_cert={state.beta_cert:.3f}. "
                "Cannot shrink B_cert."
            )
        return True, "ok"

    if state.calibrated:
        if proposed_beta < state.beta_cert:
            return False, (
                f"proposed_beta={proposed_beta:.3f} < "
                f"calibrated_beta_cert={state.beta_cert:.3f}. "
                "Cannot shrink B_cert."
            )
        return True, "ok"

    return True, "not_yet_calibrated"


def get_calibration_report(state: CalibrationState) -> Dict[str, Any]:
    """Get calibration status report."""
    return {
        "calibrated": state.calibrated,
        "frozen": state.frozen,
        "beta_cert": state.beta_cert,
        "sample_count": state.sample_count,
        "coverage_achieved": state.coverage_achieved,
        "estimated_margin": state.estimated_margin,
        "residual_mean": float(np.mean(state.residuals)) if state.residuals else 0.0,
        "residual_std": float(np.std(state.residuals)) if state.residuals else 0.0,
        "residual_min": float(np.min(state.residuals)) if state.residuals else 0.0,
        "residual_max": float(np.max(state.residuals)) if state.residuals else 0.0,
    }
