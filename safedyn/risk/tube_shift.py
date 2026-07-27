"""
risk/tube_shift.py
Phase 2: Tube shift validation and certified-versus-planning radius check.
Coordinate system: x-z-yaw.

B_cert MUST always have radius >= B_plan radius.
This file provides the validation logic used during runtime.
"""

from typing import List


def cert_ge_plan_violation(plan_radii: List[float], cert_radii: List[float]) -> int:
    """
    Count how many tube elements have cert_radius < plan_radius.
    B_cert must never be smaller than B_plan.
    Returns 0 when invariant holds.
    """
    if not plan_radii or not cert_radii:
        return 0
    return sum(1 for cp, pp in zip(cert_radii, plan_radii) if float(cp) < float(pp))


def validate_cert_ge_plan(plan_radii: List[float], cert_radii: List[float]) -> None:
    """
    Raise ValueError if any cert_radius < plan_radius.
    Call this during tube construction to catch bugs early.
    """
    violations = cert_ge_plan_violation(plan_radii, cert_radii)
    if violations > 0:
        raise ValueError(
            f"B_cert < B_plan violation: {violations} tube elements have "
            f"cert_radius < plan_radius. B_cert must never be smaller."
        )
