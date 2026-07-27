# INSTALL

## Minimal Install (Method Demos + Tests)

SafeDyn method core requires only `numpy` and `scipy`. Demos and tests work without Habitat-Sim or GPU.

```bash
conda create -n safedyn-method python=3.9 -y
conda activate safedyn-method

# Core requirements
pip install -r requirements/safedyn.txt

# Install SafeDyn itself (editable mode)
pip install -e .
```

## Optional: Habitat Integration

If you want to run SafeDyn with live Habitat-Sim:

```bash
pip install -r requirements/habitat_optional.txt
```

See [habitat-sim](https://github.com/facebookresearch/habitat-sim) for Habitat-Sim build/install instructions.

## Optional: ETPNav Integration

For SafeDyn + ETPNav usage:

```bash
pip install -r requirements/habitat_optional.txt  # Habitat dependencies
pip install torch torchvision  # PyTorch
pip install modelscope         # For ETPNav weight download
python scripts/download_etpnav_weights_modelscope.sh
```

See [WEIGHTS.md](WEIGHTS.md) and [DATASETS.md](DATASETS.md) for data requirements.

## Optional: Visualization (for logged tick plotting)

```bash
pip install -r requirements/visualization_optional.txt
```

## Verify Install

```bash
# Check SafeDyn imports
python -c "from safedyn.safety.certified_accept import CertifiedAccept; print('OK')"
python -c "from safedyn.runtime.dual_rate import DualRateRuntime; print('OK')"

# Run tests (optional — see note)
pytest tests/ -v
```

**Note:** This is a method-only release. Tests are lightweight checks; this push was not blocked on local pytest execution. See [QUICKSTART.md](QUICKSTART.md) for details.
