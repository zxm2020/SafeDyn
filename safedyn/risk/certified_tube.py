"""
risk/certified_tube.py
Phase 2: Certified tube B_cert construction.
Coordinate system: x-z-yaw.

B_cert is used for HARD SAFETY CERTIFICATION — it does NOT enter planner cost.
Rules:
  - margin MUST be >= planning_margin.
  - Covariance scale MUST be >= planning covariance_scale.
  - CANNOT be shrunk by learned risk, planner preference, or confidence optimism.
  - Missed observations MUST inflate the tube (not shrink it).
  - Low confidence nearby tracks MUST inflate the tube (not shrink it).
  - CANNOT be modified by any learned module.
  - Is NOT a safety guarantee — just a conservative geometric input.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Dict


@dataclass
class CertifiedTubeRadiusConfig:
    """
    Production configuration for B_cert radius computation.
    All parameters are explicit production tube parameters, not diagnostic.
    """
    beta: float = 1.0
    covariance_scale: float = 2.0
    delay_margin: float = 0.0
    dynamics_margin: float = 0.0
    min_margin: float = 0.0  # production minimum margin (replaces diagnostic_min_margin)
    low_confidence_inflation: float = 0.1
    missed_observation_inflation: float = 0.05
    certified_margin: float = 0.45
    mode: str = "production"  # "production" or "posthoc_preliminary"


def compute_production_bcert_radius(
    entity_radius: float,
    covariance_radius: float,
    config: CertifiedTubeRadiusConfig,
    extra_inflation: float = 0.0,
    missed_steps: int = 0,
    low_confidence: bool = False,
) -> Dict[str, Any]:
    """
    Compute production B_cert radius with explicit parameter accounting.

    Formula:
        radius = entity_radius
                 + beta * covariance_scale * covariance_radius
                 + delay_margin
                 + dynamics_margin
                 + min_margin
                 + low_confidence_inflation (if low_confidence)
                 + missed_observation_inflation * missed_steps
                 + extra_inflation

    Returns dict with:
        radius, covariance_radius, beta_applied, covariance_scale_applied,
        delay_margin_applied, dynamics_margin_applied, min_margin_applied,
        low_confidence_inflation, missed_observation_inflation,
        extra_inflation, radius_inflation_mode
    """
    # Base certified covariance inflation
    cert_cov = covariance_radius * config.beta * config.covariance_scale

    # Additional inflation sources
    low_conf_inflation = config.low_confidence_inflation if low_confidence else 0.0
    missed_inflation = config.missed_observation_inflation * missed_steps

    radius = (
        entity_radius
        + cert_cov
        + config.certified_margin
        + config.delay_margin
        + config.dynamics_margin
        + config.min_margin
        + low_conf_inflation
        + missed_inflation
        + extra_inflation
    )

    return {
        "radius": float(radius),
        "covariance_radius": float(cert_cov),
        "beta_applied": float(config.beta),
        "covariance_scale_applied": float(config.covariance_scale),
        "delay_margin_applied": float(config.delay_margin),
        "dynamics_margin_applied": float(config.dynamics_margin),
        "min_margin_applied": float(config.min_margin),
        "low_confidence_inflation": float(low_conf_inflation),
        "missed_observation_inflation": float(missed_inflation),
        "extra_inflation": float(extra_inflation),
        "radius_inflation_mode": str(config.mode),
        "production_radius_integration": (config.mode == "production"),
    }


def validate_tube_config(
    planning_margin: float,
    certified_margin: float,
    covariance_scale_plan: float,
    covariance_scale_cert: float,
) -> None:
    """
    Validate tube config constraints.
    Raises ValueError if constraints are violated.
    """
    if certified_margin < planning_margin:
        raise ValueError(
            f"certified_margin ({certified_margin}) must be >= planning_margin ({planning_margin}). "
            "B_cert cannot be smaller than B_plan."
        )
    if covariance_scale_cert < covariance_scale_plan:
        raise ValueError(
            f"covariance_scale_cert ({covariance_scale_cert}) must be >= "
            f"covariance_scale_plan ({covariance_scale_plan}). "
            "B_cert uncertainty cannot be smaller than B_plan uncertainty."
        )


@dataclass
class CertifiedTubeElement:
    """
    One element of the certified tube.
    Conservative circle in x-z plane for safety check.
    """
    step: int              # time step index
    center: np.ndarray     # [x, z]
    radius: float          # certified radius (conservative)
    covariance_radius: float  # uncertainty inflation
    margin: float          # certified margin applied
    source: str = "certified"
    track_id: str = ""      # track/entity identifier
    entity_id: str = ""    # source entity identifier
    beta_applied: float = 0.0  # beta multiplier used
    delay_margin_applied: float = 0.0  # delay margin applied
    dynamics_margin_applied: float = 0.0  # dynamics margin applied
    low_confidence_inflation: float = 0.0  # inflation from low confidence
    missed_observation_inflation: float = 0.0  # inflation from missed observations
    # Deprecated: diagnostic_min_margin is replaced by min_margin_applied
    diagnostic_min_margin: float = 0.0
    # Production fields (F2B+)
    min_margin_applied: float = 0.0
    covariance_scale_applied: float = 0.0
    extra_inflation: float = 0.0
    radius_inflation_mode: str = "posthoc_preliminary"


def build_certified_tube(
    tracks: List[Any],        # List[TemporalEntityState]
    robot_pos: np.ndarray,
    robot_radius: float,
    dt: float,
    horizon_steps: int = 20,
    planning_margin: float = 0.20,
    certified_margin: float = 0.45,
    covariance_scale_plan: float = 1.0,
    covariance_scale_cert: float = 2.0,
    missed_observation_inflation: float = 0.05,
    beta: float = 1.0,
    delay_margin: float = 0.0,
    dynamics_margin: float = 0.0,
    min_margin: float = 0.0,   # Production minimum margin (replaces diagnostic_min_margin)
    low_confidence_inflation: float = 0.1,
    radius_inflation_mode: str = "production",
) -> Tuple[List[CertifiedTubeElement], Dict[str, Any]]:
    """
    Build certified tubes around tracked entities.
    Certified tube is MORE conservative than planning tube:
      - larger margin
      - larger covariance scale
      - additional inflation for missed observations or low confidence
      - beta * covariance_radius
      - delay_margin
      - dynamics_margin
      - min_margin (production minimum margin)

    Args:
        tracks: current tracked entities
        robot_pos: robot [x, z]
        robot_radius: collision radius
        dt: time step (s)
        horizon_steps: future steps
        planning_margin: planning tube margin (m)
        certified_margin: certified tube margin (m) — must be >= planning_margin
        covariance_scale_plan: planning tube covariance multiplier
        covariance_scale_cert: certified tube covariance multiplier — must be >= plan
        missed_observation_inflation: extra radius inflation per missed step
        beta: additional covariance multiplier for certified tube
        delay_margin: additional margin for delay uncertainty
        dynamics_margin: additional margin for dynamics model uncertainty
        min_margin: production minimum safety margin (replaces diagnostic_min_margin)
        low_confidence_inflation: extra margin for low confidence tracks
        radius_inflation_mode: "production" or "posthoc_preliminary"

    Returns:
        Tuple of (tube_elements, metadata_dict). Raises ValueError if config violates constraints.
    """
    # Production config
    radius_config = CertifiedTubeRadiusConfig(
        beta=beta,
        covariance_scale=covariance_scale_cert,
        delay_margin=delay_margin,
        dynamics_margin=dynamics_margin,
        min_margin=min_margin,
        low_confidence_inflation=low_confidence_inflation,
        missed_observation_inflation=missed_observation_inflation,
        certified_margin=certified_margin,
        mode=radius_inflation_mode,
    )

    # Validate before building
    validate_tube_config(
        planning_margin, certified_margin,
        covariance_scale_plan, covariance_scale_cert,
    )

    robot_pos = np.asarray(robot_pos, dtype=np.float64)
    tube_elements: List[CertifiedTubeElement] = []

    # Track cert_ge_plan violations
    cert_ge_plan_violations = 0
    total_elements = 0
    min_cert_margin = float('inf')
    mean_cert_radius_accum = 0.0

    for track in tracks:
        pos = np.asarray(track.position, dtype=np.float64)  # [x, z]
        vel = np.asarray(track.velocity, dtype=np.float64)   # [vx, vz]
        entity_radius = float(track.radius)
        uncertainty_inflation = float(track.uncertainty_radius) - entity_radius

        # Extra inflation for missed steps or low confidence
        missed_steps = int(getattr(track, 'missed_steps', 0))
        low_confidence = float(getattr(track, 'confidence', 1.0)) < 0.5
        extra_inflation = 0.0  # additional extra inflation beyond config

        # Compute production B_cert radius
        radius_result = compute_production_bcert_radius(
            entity_radius=entity_radius,
            covariance_radius=uncertainty_inflation,
            config=radius_config,
            extra_inflation=extra_inflation,
            missed_steps=missed_steps,
            low_confidence=low_confidence,
        )
        base_radius = radius_result["radius"]
        cert_cov = radius_result["covariance_radius"]

        # Compare with planning radius to detect violations
        plan_cov = uncertainty_inflation * float(covariance_scale_plan)
        plan_radius = (
            entity_radius
            + plan_cov
            + float(planning_margin)
        )

        if base_radius < plan_radius:
            cert_ge_plan_violations += horizon_steps

        total_elements += horizon_steps
        min_cert_margin = min(min_cert_margin, float(certified_margin) + radius_result["low_confidence_inflation"] + radius_result["missed_observation_inflation"])
        mean_cert_radius_accum += base_radius * horizon_steps

        for step_idx in range(1, horizon_steps + 1):
            t = float(step_idx) * float(dt)
            center = pos + vel * t

            tube_elements.append(CertifiedTubeElement(
                step=step_idx,
                center=center.copy(),
                radius=float(base_radius),
                covariance_radius=float(cert_cov),
                margin=float(certified_margin) + radius_result["low_confidence_inflation"] + radius_result["missed_observation_inflation"] + float(delay_margin) + float(dynamics_margin) + float(min_margin),
                source="certified",
                track_id=getattr(track, 'track_id', ""),
                entity_id=getattr(track, 'entity_id', ""),
                beta_applied=float(radius_result["beta_applied"]),
                delay_margin_applied=float(radius_result["delay_margin_applied"]),
                dynamics_margin_applied=float(radius_result["dynamics_margin_applied"]),
                low_confidence_inflation=float(radius_result["low_confidence_inflation"]),
                missed_observation_inflation=float(radius_result["missed_observation_inflation"]),
                min_margin_applied=float(radius_result["min_margin_applied"]),
                covariance_scale_applied=float(radius_result["covariance_scale_applied"]),
                extra_inflation=float(radius_result["extra_inflation"]),
                radius_inflation_mode=str(radius_result["radius_inflation_mode"]),
            ))

    metadata = {
        "cert_ge_plan_violations": cert_ge_plan_violations,
        "total_elements": total_elements,
        "min_cert_margin": min_cert_margin if tracks else 0.0,
        "mean_cert_radius": mean_cert_radius_accum / total_elements if total_elements > 0 else 0.0,
        "radius_p95": base_radius if tracks else 0.0,  # Simplified, could compute actual p95
        "beta": beta,
        "delay_margin": delay_margin,
        "dynamics_margin": dynamics_margin,
        "min_margin": min_margin,
        "radius_inflation_mode": radius_inflation_mode,
        "production_radius_integration": (radius_inflation_mode == "production"),
    }

    return tube_elements, metadata


def cert_overlaps_robot(
    robot_pos: np.ndarray,
    robot_radius: float,
    tube_elem: CertifiedTubeElement,
) -> bool:
    """Check if robot circle overlaps a certified tube element."""
    dist = float(np.linalg.norm(np.asarray(robot_pos) - tube_elem.center))
    combined = float(robot_radius) + tube_elem.radius
    return bool(dist < combined)


def cert_to_dict(elem: CertifiedTubeElement) -> dict:
    return {
        "step": elem.step,
        "center": elem.center.tolist(),
        "radius": elem.radius,
        "covariance_radius": elem.covariance_radius,
        "margin": elem.margin,
        "source": elem.source,
        "track_id": elem.track_id,
        "entity_id": elem.entity_id,
        "beta_applied": elem.beta_applied,
        "delay_margin_applied": elem.delay_margin_applied,
        "dynamics_margin_applied": elem.dynamics_margin_applied,
        "low_confidence_inflation": elem.low_confidence_inflation,
        "missed_observation_inflation": elem.missed_observation_inflation,
        "min_margin_applied": elem.min_margin_applied,
        "covariance_scale_applied": elem.covariance_scale_applied,
        "extra_inflation": elem.extra_inflation,
        "radius_inflation_mode": elem.radius_inflation_mode,
    }
