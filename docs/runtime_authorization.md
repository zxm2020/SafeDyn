# RUNTIME AUTHORIZATION — How It Works

## Overview

The runtime authorization loop is the central executive of SafeDyn. It runs at a
configurable frequency (dual-rate: soft deadline + hard timeout) and orchestrates
the full certificate chain.

## Flow

```
1. Receive proposed action from untrusted policy
2. Build CertifiedAcceptInput from robot state + proposal
3. Call CertifiedAccept.evaluate(input)
4. Examine CertifiedAcceptDecision:
   a. If certified=True: write action to actuator
   b. If certified=False: select fallback, write fallback to actuator
5. Log decision to certificate LEDGER
6. Return to step 1
```

## Dual-Rate Runtime

The runtime operates at two rates:

- **Soft deadline:** Time budget for the full certificate chain. If exceeded,
  the guard transitions to a degraded mode.
- **Hard timeout:** If the certificate chain does not complete within this
  limit, the guard fails closed (STOP).

## Fail-Closed Path

- If `CertifiedAccept` returns `certified=False`, the runtime calls
  `FallbackPolicy.select_fallback()`.
- The fallback action is always a certified-safe command (STOP).
- The runtime **never** bypasses the certificate decision.
- Even in degraded mode (missed soft deadline), the hard timeout ensures
  safe behavior.

## State Machine

The `RecoveryStateMachine` tracks the guard's operational state:

- **NORMAL:** Full certification active
- **DEGRADED:** Soft deadline missed; conservative fallback
- **FAILED:** Hard timeout; immediate STOP
- **RECOVERY:** Attempting to resume normal operation

## Logging

Each decision produces a log entry with:

- Timestamp
- Certificate ID
- Proposed action
- Certified? (yes/no)
- Fallback reason (if rejected)
- Bypass count
- Uncertified execution count
- Actuator write status
