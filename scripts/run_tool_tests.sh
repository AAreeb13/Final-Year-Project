#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "Running tool tests from $ROOT..."

if [ -f .venv/bin/activate ]; then
  echo "Activating .venv..."
  # shellcheck source=/dev/null
  . .venv/bin/activate
else
  echo "No .venv found — creating one and installing pytest..."
  python3 -m venv .venv
  # shellcheck source=/dev/null
  . .venv/bin/activate
  pip install --upgrade pip
  pip install pytest
fi

echo "Running pytest for python_code_execution tool..."
pytest -q tests/tools/python_code_execution
