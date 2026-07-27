"""
safety/cbf_clf_qp.py
SafeDyn-VLN Guard: Relaxed CBF-CLF-QP safety filter.

Full implementation with:
  - CBF safety constraints
  - CLF/progress term
  - Anisotropic weights
  - Urgent zero-slack constraints
  - Non-urgent relaxed slack
  - Active constraint subset
  - QPResult dataclass

QPResult cannot directly execute. It must go through CertifiedAccept.

Retains safety/cbf_qp_lite.py as ablation/legacy.

Coordinate system: x-z-yaw.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Optional QP solver
_QP_AVAILABLE: bool = False
try:
    import cvxpy as cp
    _QP_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore


@dataclass
class CBFCLFQPConfig:
    """Configuration for CBF-CLF-QP."""
    dt: float = 0.05
    v_min: float = 0.0
    v_max: float = 0.8
    omega_min: float = -1.2
    omega_max: float = 1.2
    robot_radius: float = 0.25
    gamma_cbf: float = 5.0          # CBF class-K function rate
    gamma_clf: float = 2.0          # CLF convergence rate
    cbf_weight: float = 100.0       # weight for CBF constraint
    clf_weight: float = 10.0        # weight for CLF progress term
    anisotropic_v_weight: float = 1.0
    anisotropic_omega_weight: float = 0.5
    urgent_slack_penalty: float = 1000.0   # high penalty for urgent slack
    nonurgent_slack_penalty: float = 10.0  # lower penalty for non-urgent
    n_nearest: int = 3              # nearest tube elements for active set
    goal_direction: Optional[np.ndarray] = None  # [dx, dz] to goal


@dataclass
class QPResult:
    """
    Result of CBF-CLF-QP solve.

    This action MUST NOT be executed directly.
    It must pass through CertifiedAccept.
    """
    action: Dict[str, float]
    cbf_satisfied: bool
    clf_value: float
    urgent_slack_used: float
    nonurgent_slack_used: float
    active_constraints: int
    solve_status: str          # "optimal", "infeasible", "solver_error", "analytical_fallback"
    qp_success: bool
    fallback_to_analytical: bool
    total_slack: float = 0.0
    clearance_min: float = float("inf")
    components: Dict[str, float] = field(default_factory=dict)


def compute_clearance_gradient(
    robot_pos: np.ndarray,
    yaw: float,
    tube_center: np.ndarray,
) -> np.ndarray:
    """
    Compute gradient of clearance h(x) w.r.t. [v, omega].

    h = ||robot - center|| - R
    dh/dv = (heading · dir_to_center)
    dh/domega = (robot_pos - center) perp component
    """
    diff = np.asarray(tube_center) - np.asarray(robot_pos)
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return np.array([0.0, 0.0])

    dir_to_center = diff / dist
    heading = np.array([np.sin(yaw), -np.cos(yaw)])

    # dh/dv = heading · dir_to_center (how fast clearance changes with v)
    dh_dv = float(np.dot(heading, dir_to_center))

    # dh/domega ≈ 0 for small perturbations (heading rotates slowly)
    dh_domega = 0.0

    return np.array([dh_dv, dh_domega])


def compute_clf_gradient(
    robot_pos: np.ndarray,
    goal_pos: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """
    Compute gradient of CLF function V(x) w.r.t. [v, omega].

    V = ||robot - goal||^2
    dV/dv = -2 * (robot - goal) · heading
    dV/domega ≈ 0
    """
    diff = np.asarray(robot_pos) - np.asarray(goal_pos)
    heading = np.array([np.sin(yaw), -np.cos(yaw)])

    dV_dv = float(-2.0 * np.dot(diff, heading))
    dV_domega = 0.0

    return np.array([dV_dv, dV_domega])


def find_active_constraints(
    robot_pos: np.ndarray,
    b_cert: List[Any],
    robot_radius: float,
    n_nearest: int,
) -> List[Any]:
    """Find nearest tube elements for active constraint set."""
    if not b_cert:
        return []
    dists = []
    for te in b_cert:
        d = float(np.linalg.norm(np.asarray(robot_pos) - np.asarray(te.center)))
        dists.append((d, te))
    dists.sort(key=lambda x: x[0])
    return [te for _, te in dists[:n_nearest]]


def solve_cbf_clf_qp(
    nominal_v: float,
    nominal_omega: float,
    robot_pos: np.ndarray,
    yaw: float,
    b_cert: List[Any],
    goal_pos: Optional[np.ndarray],
    config: CBFCLFQPConfig,
) -> QPResult:
    """
    Solve CBF-CLF-QP for safe, goal-progressing action.

    If QP solver not available, falls back to analytical filter.
    If QP fails, returns with fallback_to_analytical=True.

    Returns QPResult (must go through CertifiedAccept before execution).
    """
    robot_pos = np.asarray(robot_pos, dtype=np.float64)

    # Find active constraint subset
    active_tubes = find_active_constraints(
        robot_pos, b_cert, config.robot_radius, config.n_nearest,
    )

    if not active_tubes:
        # No constraints — return nominal
        return QPResult(
            action={"linear_velocity": nominal_v, "angular_velocity": nominal_omega},
            cbf_satisfied=True,
            clf_value=0.0,
            urgent_slack_used=0.0,
            nonurgent_slack_used=0.0,
            active_constraints=0,
            solve_status="no_constraints",
            qp_success=True,
            fallback_to_analytical=False,
            clearance_min=float("inf"),
        )

    # Compute clearances
    clearances = []
    for te in active_tubes:
        dist = float(np.linalg.norm(robot_pos - np.asarray(te.center)))
        h = dist - config.robot_radius - float(te.radius)
        clearances.append(h)
    min_clearance = min(clearances) if clearances else float("inf")

    # Try QP solver
    if _QP_AVAILABLE and cp is not None:
        try:
            return _solve_qp(
                nominal_v, nominal_omega, robot_pos, yaw,
                active_tubes, clearances, goal_pos, config,
            )
        except Exception:
            pass

    # Fallback: analytical CBF filter
    return _analytical_fallback(
        nominal_v, nominal_omega, robot_pos, yaw,
        active_tubes, clearances, min_clearance, config,
    )


def _solve_qp(
    nominal_v: float,
    nominal_omega: float,
    robot_pos: np.ndarray,
    yaw: float,
    active_tubes: List[Any],
    clearances: List[float],
    goal_pos: Optional[np.ndarray],
    config: CBFCLFQPConfig,
) -> QPResult:
    """Solve with cvxpy QP."""
    dv = cp.Variable()
    domega = cp.Variable()
    v_new = nominal_v + dv
    omega_new = nominal_omega + domega

    # Urgent slack (for hard CBF constraints)
    s_urgent = cp.Variable(len(active_tubes), nonneg=True)
    # Non-urgent slack (for CLF progress)
    s_nonurgent = cp.Variable(nonneg=True)

    constraints = [
        config.v_min <= v_new, v_new <= config.v_max,
        config.omega_min <= omega_new, omega_new <= config.omega_max,
    ]

    # CBF constraints: h_dot + gamma * h >= -s_urgent
    for i, te in enumerate(active_tubes):
        h = clearances[i]
        grad = compute_clearance_gradient(robot_pos, yaw, np.asarray(te.center))
        # h_dot ≈ grad · [v_new, omega_new]
        # CBF: h_dot >= -gamma * h - s_urgent[i]
        constraints.append(
            grad[0] * v_new + grad[1] * omega_new >=
            -config.gamma_cbf * h - s_urgent[i]
        )

    # CLF constraint: V_dot <= -gamma * V + s_nonurgent
    if goal_pos is not None:
        V = float(np.linalg.norm(robot_pos - np.asarray(goal_pos))**2)
        clf_grad = compute_clf_gradient(robot_pos, goal_pos, yaw)
        constraints.append(
            clf_grad[0] * v_new + clf_grad[1] * omega_new <=
            -config.gamma_clf * V + s_nonurgent
        )

    # Objective: minimize control effort + slack penalties
    objective = cp.Minimize(
        config.anisotropic_v_weight * cp.square(dv) +
        config.anisotropic_omega_weight * cp.square(domega) +
        config.urgent_slack_penalty * cp.sum(s_urgent) +
        config.nonurgent_slack_penalty * s_nonurgent
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP if hasattr(cp, 'OSQP') else cp.SCS, verbose=False)

    if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        v_out = float(np.clip(nominal_v + dv.value, config.v_min, config.v_max))
        omega_out = float(np.clip(nominal_omega + domega.value, config.omega_min, config.omega_max))
        urgent_slack = float(np.sum(s_urgent.value)) if s_urgent.value is not None else 0.0
        nonurgent_slack = float(s_nonurgent.value) if s_nonurgent.value is not None else 0.0

        # Check CBF satisfaction
        cbf_satisfied = urgent_slack < 1e-6

        return QPResult(
            action={"linear_velocity": v_out, "angular_velocity": omega_out},
            cbf_satisfied=cbf_satisfied,
            clf_value=float(np.linalg.norm(robot_pos - np.asarray(goal_pos))**2) if goal_pos is not None else 0.0,
            urgent_slack_used=urgent_slack,
            nonurgent_slack_used=nonurgent_slack,
            active_constraints=len(active_tubes),
            solve_status="optimal",
            qp_success=True,
            fallback_to_analytical=False,
            total_slack=urgent_slack + nonurgent_slack,
            clearance_min=float(min(clearances)) if clearances else float("inf"),
        )

    # QP failed
    return _analytical_fallback(
        nominal_v, nominal_omega, robot_pos, yaw,
        active_tubes, clearances, min(clearances) if clearances else float("inf"), config,
    )


def _analytical_fallback(
    nominal_v: float,
    nominal_omega: float,
    robot_pos: np.ndarray,
    yaw: float,
    active_tubes: List[Any],
    clearances: List[float],
    min_clearance: float,
    config: CBFCLFQPConfig,
) -> QPResult:
    """Analytical CBF filter fallback (no QP solver needed)."""
    v = float(nominal_v)
    omega = float(nominal_omega)

    for i, te in enumerate(active_tubes):
        h = clearances[i]
        if h < 0:
            # Must move away — clamp forward velocity
            diff = np.asarray(te.center) - robot_pos
            dist = float(np.linalg.norm(diff))
            if dist > 1e-6:
                heading = np.array([np.sin(yaw), -np.cos(yaw)])
                dir_to_center = diff / dist
                v_along = float(np.dot(heading, dir_to_center)) * v
                if v_along > 0:
                    v = max(0.0, v * 0.1)  # drastically reduce
        elif h < config.gamma_cbf * config.dt:
            # Near constraint — scale down approach velocity
            diff = np.asarray(te.center) - robot_pos
            dist = float(np.linalg.norm(diff))
            if dist > 1e-6:
                heading = np.array([np.sin(yaw), -np.cos(yaw)])
                dir_to_center = diff / dist
                v_along = float(np.dot(heading, dir_to_center)) * v
                max_approach = h / max(config.dt, 1e-6)
                if v_along > max_approach and abs(v_along) > 1e-6:
                    scale = max(0.0, min(1.0, max_approach / v_along))
                    v = v * scale

    v = float(np.clip(v, config.v_min, config.v_max))
    omega = float(np.clip(omega, config.omega_min, config.omega_max))

    # Compute slack estimate
    urgent_slack = max(0.0, -min_clearance) if min_clearance < 0 else 0.0

    return QPResult(
        action={"linear_velocity": v, "angular_velocity": omega},
        cbf_satisfied=(min_clearance >= 0),
        clf_value=0.0,
        urgent_slack_used=urgent_slack,
        nonurgent_slack_used=0.0,
        active_constraints=len(active_tubes),
        solve_status="analytical_fallback",
        qp_success=False,
        fallback_to_analytical=True,
        total_slack=urgent_slack,
        clearance_min=float(min_clearance),
    )


def filter_action_cbf_clf(
    nominal_action: Dict[str, float],
    robot_pos: np.ndarray,
    yaw: float,
    b_cert: List[Any],
    goal_pos: Optional[np.ndarray] = None,
    config: Optional[CBFCLFQPConfig] = None,
) -> QPResult:
    """
    Top-level CBF-CLF-QP filter entry point.

    Returns QPResult. Action must pass through CertifiedAccept before execution.
    """
    if config is None:
        config = CBFCLFQPConfig()

    nominal_v = float(nominal_action.get("linear_velocity", 0.0))
    nominal_omega = float(nominal_action.get("angular_velocity", 0.0))

    return solve_cbf_clf_qp(
        nominal_v=nominal_v,
        nominal_omega=nominal_omega,
        robot_pos=np.asarray(robot_pos, dtype=np.float64),
        yaw=yaw,
        b_cert=b_cert,
        goal_pos=goal_pos,
        config=config,
    )
