# METHOD.md — SafeDyn-VLN Guard Architecture

## Overview

SafeDyn is a runtime actuator-authorization layer. It wraps an untrusted navigation policy
and enforces that **only certified actions reach the actuator**. The certificate chain
evaluates each proposed action through multiple safety gates and issues a certificate only
when all gates pass. If any gate fails, the guard executes a certified fallback (STOP,
yield, or reverse) instead.

## Architecture

```
Untrusted Policy (ETPNav, etc.)
        │  proposes action (v, ω)
        ▼
┌─────────────────────────────────────┐
│         CertifiedAccept Gate        │
│  ┌───────────┬──────────┬────────┐  │
│  │ Tube      │ Exact    │ Backup │  │
│  │ Overlap   │ Rollout  │ Valid. │  │
│  └───────────┴──────────┴────────┘  │
│  ┌───────────┬──────────┬────────┐  │
│  │ Visibility│ CBF-QP   │ Recov. │  │
│  │ Speed     │ Lite     │ Mach.  │  │
│  └───────────┴──────────┴────────┘  │
└─────────────────────────────────────┘
        │ cert=True or cert=False
        ▼
┌──────────────────┐    ┌──────────────────┐
│ CERTIFIED        │    │ REJECTED         │
│ → Write action   │    │ → Fail closed    │
│   to actuator    │    │   (STOP/fallback)│
└──────────────────┘    └──────────────────┘
```

## Untrusted Candidate Sources

The guard accepts candidate actions from any untrusted source:

- **ETPNav / VLN policy** — via `baselines/etpnav_wrapper.py`
- **User-defined policy** — via `baselines/policy_interface.py`
- **Random/dummy policy** — for testing (see `examples/minimal_runtime_authorization.py`)

## CertifiedAccept

`CertifiedAccept` is the single entry point. It receives:

- Robot state (position `x,z`, yaw)
- Proposed action `(linear_velocity, angular_velocity)`
- Goal position
- Tracked entity states (positions, velocities, covariances)

It returns a `CertifiedAcceptDecision` with:

| Field | Type | Description |
|-------|------|-------------|
| `certified` | bool | True if the action passes all checks |
| `preliminary` | bool | True if any check was skipped (bypass) |
| `fallback_reason` | str or None | Why the action was rejected |
| `certificate_id` | str | Unique ID for this decision |
| `bypass_count` | int | Cumulative bypass count |
| `uncertified_exec_count` | int | Cumulative uncertified execution count |

## Certificate Contract

The certificate contract is:

1. **Every action** proposed to the guard must be evaluated by `CertifiedAccept`.
2. **Only certified actions** are written to the actuator.
3. **Uncertified actions** are replaced by a certified fallback (STOP).
4. **bypass=0 and uncertified_exec=0** means every action was properly certified before execution.
5. The **certificate LEDGER** records every decision with a unique ID.

## Actuator Authorization Invariant

**Definition:** The actuator must never receive an action that has not passed
`CertifiedAccept` with `certified=True`. If `certified=False`, the actuator
receives only the certified fallback action (STOP).

This is enforced by the runtime authorization loop (`DualRateRuntime`,
`IncrementalLoop`).

## Fail-Closed Behavior

When certification fails:

1. The proposed action is **not written** to the actuator.
2. The fallback policy (`fallback.py`) selects a certified safe action:
   - **Priority:** STOP → yield left → yield right → slow reverse
3. The actuator receives only the fallback action.
4. The LEDGER records the bypass with reason.

This guarantees that even if the policy proposes dangerous actions, the robot
never moves unsafely.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| B_plan margin | 0.20 m | Planning tube margin |
| B_cert margin | 0.45 m | Certified tube margin (≥ B_plan) |
| B_plan scale | 1.0 σ | Planning tube covariance scale |
| B_cert scale | 2.0 σ | Certified tube covariance scale (≥ B_plan) |

## Invariant: B_cert ≥ B_plan

The certified tube must always be larger (more conservative) than the planning
tube. This ensures that the certifier is always more cautious than the planner.
This invariant is validated in `tube_shift.py` and tested in `tests/test_certified_accept.py`.
