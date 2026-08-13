#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
(sleep 3; open "http://localhost:8800/") &
exec bash "$PROJECT_DIR/web/scripts/start_web.sh" 8800
