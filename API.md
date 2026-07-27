# API.md — SafeDyn Core API

## CertifiedAccept

```python
from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput, CertifiedAcceptDecision
```

### CertifiedAcceptInput

| Field | Type | Description |
|-------|------|-------------|
| `robot_position` | np.ndarray (2,) | Robot x-z position (meters) |
| `robot_yaw` | float | Heading angle (radians) |
| `linear_velocity` | float | Proposed forward speed (m/s) |
| `angular_velocity` | float | Proposed turn rate (rad/s) |
| `goal_position` | np.ndarray (2,) | Navigation goal x-z position |
| `tracked_entities` | List[dict] | Tracked obstacle states |
| `config_override` | dict or None | Optional per-call config override |

### CertifiedAcceptDecision

| Field | Type | Description |
|-------|------|-------------|
| `certified` | bool | True = action passes all checks |
| `preliminary` | bool | True = any check was skipped (bypass) |
| `fallback_reason` | str or None | Description of failure, if any |
| `certificate_id` | str | Unique identifier for this certificate |
| `bypass_count` | int | Cumulative bypass count across decisions |
| `uncertified_exec_count` | int | Cumulative uncertified action count |

### CertifiedAccept

```python
cert_accept = CertifiedAccept(tube_cfg=CertifiedTubeRadiusConfig(...))
decision = cert_accept.evaluate(input)
```

## CertifiedTubeRadiusConfig

```python
from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig
```

| Field | Default | Description |
|-------|---------|-------------|
| `planning_margin` | 0.20 | B_plan radius (m) |
| `certified_margin` | 0.45 | B_cert radius (m) |
| `planning_scale` | 1.0 | B_plan covariance std multiplier |
| `certified_scale` | 2.0 | B_cert covariance std multiplier |

## FallbackPolicy

```python
from safedyn.safety.fallback import FallbackPolicy
fallback = FallbackPolicy()
fallback_action = fallback.select_fallback(input, decision)
```

Returns: dict with keys `linear_velocity`, `angular_velocity`.
Default: STOP = `{"linear_velocity": 0.0, "angular_velocity": 0.0}`.

## DualRateRuntime

```python
from safedyn.runtime.dual_rate import DualRateRuntime, DualRateConfig
from safedyn.runtime.incremental_loop import IncrementalLoop
```

Configurable runtime loop with soft deadline and hard timeout.

## PolicyInterface

```python
from safedyn.baselines.policy_interface import PolicyInterface
```

Abstract base class for untrusted candidate policies.

## Log Format (JSONL)

Each tick in a JSONL log file:

```json
{
  "tick_id": 0,
  "robot_pose": {"x": 0.0, "z": 0.0, "yaw": 0.0},
  "action": {"linear_velocity": 0.5, "angular_velocity": 0.0},
  "goal_position": {"x": 5.0, "z": 2.0},
  "tracked_entities": [],
  "certified": true,
  "bypass": 0,
  "uncertified_exec": 0
}
```

Fields:
- `tick_id`: integer step counter
- `robot_pose`: robot position in Habitat x-z coordinate system
- `action`: proposed control command
- `goal_position`: navigation target
- `tracked_entities`: list of detected obstacle states
- `certified`: whether the certificate chain passed
- `bypass`: cumulative number of bypassed certificate checks
- `uncertified_exec`: cumulative number of uncertified executions
