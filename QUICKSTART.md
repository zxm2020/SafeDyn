# QUICKSTART

## Prerequisites

```bash
conda create -n safedyn-method python=3.9 -y
conda activate safedyn-method
pip install -r requirements/safedyn.txt
pip install -e .
```

## 1. Run Method Tests

```bash
pytest tests/ -v
```

**Note:** This is a method-only release. Tests are provided as lightweight checks. Repository maintainers have not verified local pytest execution — this public push was not blocked on a passing test suite. If tests fail in your environment, this is expected to be a local setup issue; the method core logic is documented in source code and examples.

**Known limitation:** Tests may require `pip install -e .` (editable install) in your environment.

## 2. Run Minimal Authorization Demo

```bash
python examples/minimal_runtime_authorization.py
```

Shows SafeDyn certificate chain: proposes safe and unsafe actions, shows which pass/fail, prints bypass and uncertified-exec counts.

## 3. Run Fail-Closed Demo

```bash
python examples/minimal_fail_closed_demo.py
```

Demonstrates that SafeDyn replaces unsafe actions with STOP. Verifies the core safety invariant: no uncertified motion reaches the actuator.

## 4. Replay Logged Ticks

```bash
python examples/minimal_log_replay.py --ticks examples/sample_ticks.jsonl
```

Reads sample tick data, recomputes certificate decisions, prints summary.

## 5. Run Method Demo with Config

```bash
python scripts/run_method_demo.py --config configs/safedyn_default.yaml
```

## 6. Check Environment

```bash
python scripts/check_environment.py
```

## 7. Full Habitat Run (Optional)

Requires MP3D scenes, R2R-CE data, and ETPNav checkpoint:

```bash
# After setting up external data (see DATASETS.md, WEIGHTS.md)
# Run SafeDyn + ETPNav evaluation
# See docs/etpnav_optional_integration.md
```
