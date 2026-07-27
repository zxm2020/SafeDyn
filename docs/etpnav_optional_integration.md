# ETPNav Optional Integration

SafeDyn can be used with the ETPNav VLN policy. This is **optional** and
requires external code, data, and weights.

## Setup

1. Clone the official ETPNav repository:

```bash
git clone https://github.com/username/ETPNav.git external/ETPNav
```

2. Download ETPNav checkpoint:

```bash
pip install modelscope
modelscope download --model admagic/ETPNav --local_dir ./data
```

3. Set up MP3D scenes and R2R-CE data (see DATASETS.md).

4. Use the ETPNav wrapper (`safedyn/baselines/etpnav_wrapper.py`) to connect
   SafeDyn to ETPNav:

```python
from safedyn.baselines.etpnav_wrapper import ETPNavWrapper

policy = ETPNavWrapper(checkpoint_path="data/ETPNav.pt")
action = policy.act(observation)
```

## What is NOT Included

The ETPNav source code, checkpoints, data, and scenes are **not** included
in this repository. They must be obtained from their respective sources.

## Integration Adapter

The `safedyn/baselines/` directory provides adapter interfaces:

- `etpnav_wrapper.py` — Wraps ETPNav as a `PolicyInterface`
- `policy_interface.py` — Generic abstract policy interface
- `waypoint_to_control.py` — Converts waypoint predictions to control commands

These adapters assume ETPNav is installed and importable. They will fail
with an `ImportError` if ETPNav is not available.
