#!/usr/bin/env bash
# PUSH_TO_GITHUB_TEMPLATE.sh
# Template for publishing the SafeDyn method repository.
#
# Before running:
#   1. Replace <USER> and <REPO> below
#   2. Verify SSH access: ssh -T git@github.com
#   3. Create an empty repo on GitHub (no README, no LICENSE, no .gitignore)

set -euo pipefail

git remote add origin git@github.com:<USER>/<REPO>.git
git branch -M main
git push -u origin main

echo "=== Done ==="
echo "Visit: https://github.com/<USER>/<REPO>"
