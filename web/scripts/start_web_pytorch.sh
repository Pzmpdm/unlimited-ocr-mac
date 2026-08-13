#!/usr/bin/env bash
# Compatibility launcher for the original, slower PyTorch/MPS backend.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VENV_DIR="$REPO_DIR/.venv-ocr"
export OCR_BACKEND=pytorch
export PORT="${1:-8800}"

cd "$REPO_DIR/web"
source "$VENV_DIR/bin/activate"
python3 server.py
