#!/usr/bin/env bash
set -euo pipefail

msg=${1:-"chore: app tweaks/tests"}
git add -A
git commit -m "$msg" || true
git push origin HEAD
