#!/usr/bin/env python3
"""
check_environment.py

Verify SafeDyn method environment: imports, dependencies, configs, examples.
Does NOT require Habitat-Sim, MP3D scenes, or ETPNav checkpoints.
"""

import sys
import os
import platform


def check(passed, message, detail=""):
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    print(f"  {icon} [{status}] {message}" + (f"  -- {detail}" if detail else ""))


def main():
    print("=== SafeDyn Environment Check ===\n")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")
    print()

    # 1. Python version
    check(sys.version_info >= (3, 8), "Python >= 3.8", sys.version)

    # 2. Core imports
    try:
        import numpy
        check(True, "numpy", numpy.__version__)
    except ImportError:
        check(False, "numpy")

    try:
        import scipy
        check(True, "scipy", scipy.__version__)
    except ImportError:
        check(False, "scipy")

    try:
        import torch
        check(True, "PyTorch", torch.__version__)
    except ImportError:
        check(False, "PyTorch", "optional for demos")

    # 3. SafeDyn imports
    try:
        from safedyn.runtime import DualRateRuntime
        check(True, "safedyn.runtime")
    except ImportError as e:
        check(False, "safedyn.runtime", str(e))

    try:
        from safedyn.safety.certified_accept import CertifiedAccept, CertifiedAcceptInput
        from safedyn.safety.fallback import FallbackPolicy
        check(True, "safedyn.safety")
    except ImportError as e:
        check(False, "safedyn.safety", str(e))

    try:
        from safedyn.risk.certified_tube import CertifiedTubeRadiusConfig, build_certified_tube
        check(True, "safedyn.risk")
    except ImportError as e:
        check(False, "safedyn.risk", str(e))

    try:
        from safedyn.tracking.kalman import KalmanTracker
        check(True, "safedyn.tracking")
    except ImportError as e:
        check(False, "safedyn.tracking", str(e))

    try:
        from safedyn.baselines.policy_interface import PolicyInterface
        check(True, "safedyn.baselines")
    except ImportError as e:
        check(False, "safedyn.baselines", str(e))

    # 4. Optional Habitat-Sim
    try:
        import habitat_sim
        check(True, "habitat_sim", habitat_sim.__version__)
    except ImportError:
        check(False, "habitat_sim", "optional — demos and tests do not require it")

    # 5. YAML config
    try:
        import yaml
        check(True, "pyyaml")
    except ImportError:
        check(False, "pyyaml", "install with: pip install pyyaml")

    # 6. Config files
    for fname in ["safedyn_default.yaml", "runtime_authorization.yaml", "toy_demo.yaml"]:
        fpath = f"configs/{fname}"
        check(os.path.exists(fpath), f"config/{fname}")

    # 7. Example scripts
    for fname in ["minimal_runtime_authorization.py", "minimal_fail_closed_demo.py", "minimal_log_replay.py"]:
        fpath = f"examples/{fname}"
        check(os.path.exists(fpath), f"examples/{fname}")

    # 8. Sample ticks
    check(os.path.exists("examples/sample_ticks.jsonl"), "examples/sample_ticks.jsonl")

    print()
    print("=== Check complete ===")


if __name__ == "__main__":
    # Ensure running from repo root
    if not os.path.exists("safedyn") and os.path.exists("../safedyn"):
        os.chdir("..")
    main()