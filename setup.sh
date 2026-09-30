#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,14), "Use standard CPython 3.11-3.14"'
if [ -L .venv ]; then echo "Refusing a symlinked virtual environment" >&2; exit 2; fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
echo "Setup finished. Model download, camera check, and publication are separate commands."
