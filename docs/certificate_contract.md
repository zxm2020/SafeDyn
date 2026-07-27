# Certificate Contract

## Purpose

The certificate contract defines the obligations of the SafeDyn runtime guard.
It is a formal specification of the safety guarantee that SafeDyn provides.

## Terms

### 1. Certification Requirement

Every action written to the robot actuator must have been certified by
`CertifiedAccept` (or its equivalent certificate function).

### 2. No Bypass

The runtime must never bypass the certificate check. Every proposed action
must go through `CertifiedAccept.evaluate()` before reaching the actuator.

### 3. Fail-Closed (No Uncertified Execution)

If `CertifiedAccept` returns `certified=False`, the runtime must not write the
proposed action to the actuator. Instead, it must write a certified fallback
action (STOP with v=0, ω=0).

### 4. Certificate LEDGER

All certificate decisions must be recorded in a LEDGER with:
- Unique certificate ID
- Timestamp
- Proposed action
- Certified/unccertified
- Fallback reason (if any)
- Bypass count
- Uncertified execution count

### 5. Monotonic Counters

Bypass and uncertified-execution counters are monotonic: they must never
decrease across consecutive decisions.

### 6. B_cert ≥ B_plan Invariant

The certified tube parameters (margin, scale) must always be ≥ the planning
tube parameters. This ensures the certifier is always more conservative than
the planner.

## Enforcement

The invariants are enforced by:

- `tests/test_certified_accept.py` — tests certificate behavior
- `tests/test_no_uncertified_execution.py` — tests no bypass invariant
- `tests/test_fail_closed.py` — tests fail-closed behavior
- `tests/test_certificate_ledger.py` — tests LEDGER properties

## Violation Response

If any invariant is violated:
1. The guard is not certified.
2. The robot must stop immediately.
3. A diagnostic message must be logged.
