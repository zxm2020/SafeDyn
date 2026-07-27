#!/usr/bin/env bash
# download_etpnav_weights_modelscope.sh
# Optional: download ETPNav weights for users who want
# full Habitat integration (not required for SafeDyn method demos).
# Prerequisites: pip install modelscope

set -euo pipefail

echo "=== Optional: ETPNav Weight Download ==="
echo "SafeDyn method demos do NOT require ETPNav weights."
echo ""
echo "Running this script is only needed if you plan to use"
echo "SafeDyn with the ETPNav navigation policy."
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipped. Method demos will work without this."
    exit 0
fi

echo "Installing modelscope..."
python -m pip install -U modelscope

echo "Downloading..."
modelscope download --model admagic/ETPNav --local_dir ./data

echo ""
echo "Downloaded to ./data/"
echo "ETPNav weights are now available for optional integration."