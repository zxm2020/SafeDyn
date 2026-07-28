# SafeDyn-VLN Guard (Method Repository)

**A runtime actuator-authorization guard for vision-language navigation in dynamic environments.**

SafeDyn is a **runtime safety/authorization layer**, not a new VLN policy. It sits between an untrusted navigation policy (e.g., ETPNav) and the robot actuator. Every proposed action must pass through a multi-stage certificate chain (`CertifiedAccept`) before reaching the actuator. If any check fails, the guard issues a certified fallback (STOP, yield, or reverse) rather than the proposed action.

---

## This Repository: Method-Focused

This repository reproduces the **SafeDyn-VLN Guard method**:

- Runtime authorization loop with `CertifiedAccept`
- Fail-closed execution (no uncertified action reaches the actuator)
- Certificate LEDGER, bypass, and uncertified-exec tracking
- Lightweight demos and tests (no Habitat scenes, no ETPNav weights required)



## Relationship to the Paper

Paper-scale experiments required:
- MP3D scene data (proprietary, must be obtained separately)
- R2R-CE preprocessed datasets
- ETPNav fine-tuned checkpoint (downloaded via ModelScope)
- Large raw tick logs (not included)

These are **not** included. The method itself — the certificate chain, fail-closed logic, tracking, risk tubes — is fully reproducible from this repository with no external data.

## Quick Start

```bash
# Create environment
conda create -n safedyn-method python=3.9 -y
conda activate safedyn-method
pip install -r requirements/safedyn.txt

# Run method demos (no external data needed)
python examples/minimal_runtime_authorization.py
python examples/minimal_fail_closed_demo.py

# Replay logged ticks
python examples/minimal_log_replay.py --ticks examples/sample_ticks.jsonl
```

**Note:** This is a method-only release. Paper-scale verification (Habitat/MP3D rollouts, Table 1–3, all RQ experiments) and generated paper figures are not included. Tests are provided as lightweight checks, but this public push was not blocked on local pytest execution. See [QUICKSTART.md](QUICKSTART.md) for details.

## Repository Contents

| Directory | Contents |
|-----------|----------|
| `safedyn/` | Core method modules: runtime, safety, risk, tracking, planning, baselines, metrics |
| `examples/` | Self-contained method demos (no external dependencies) |
| `scripts/` | Method demo runner, tick replay, environment check, optional ETPNav download |
| `tests/` | Method-focused pytest suite |
| `configs/` | SafeDyn YAML configuration templates |
| `docs/` | Method overview, runtime authorization description, certificate contract |

## Requirements

- **Minimal**: The method core only needs `numpy`, `scipy`. Demos and tests work without Habitat-Sim.
- **Optional**: Habitat-Sim for live Habitat integration; PyTorch, modelscope, transformers for ETPNav integration.

See [INSTALL.md](INSTALL.md) and [WEIGHTS.md](WEIGHTS.md) for details.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{safedyn2025,
  title={SafeDyn-VLN Guard: Runtime Actuator Authorization for VLN},
  author={...},
  year={2025}
}
```
