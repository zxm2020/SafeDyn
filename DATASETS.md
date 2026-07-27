# DATASETS

## Minimal Demos

SafeDyn method demos (`examples/minimal_*.py`) do **not** require MP3D scenes,
R2R-CE data, or any external datasets. They generate synthetic robot positions
and obstacles, exercise the certificate chain, and print results to the console.

## Optional: Full Habitat Reproduction

Running SafeDyn with live Habitat-Sim requires:

1. **MP3D scene data** — Habitat-compatible MP3D GLB files and navmeshes.
   - Must be downloaded separately from Matterport.
   - Expected path: `data/scene_datasets/mp3d/<scene_id>/`
   - See https://github.com/facebookresearch/habitat-sim for setup.

2. **R2R-CE preprocessed data** — VLN-CE episode datasets.
   - Must be downloaded from the VLN-CE project.
   - Expected path: `data/datasets/R2R_VLNCE_v1-2_preprocessed/`
   - See https://github.com/jacobkrantz/VLN-CE for setup.

These datasets are **not included** due to licensing and size constraints.

## Verification

```bash
# Check if MP3D scenes are present
ls data/scene_datasets/mp3d/ 2>/dev/null | head -5

# Check if R2R-CE data is present
ls data/datasets/R2R_VLNCE_v1-2_preprocessed/val_seen/ 2>/dev/null | head -5
```

If these directories are missing, the method demos and tests will still work.
Only Habitat-integrated experiments require these datasets.
