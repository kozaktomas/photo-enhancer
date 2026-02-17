#!/usr/bin/env bash
set -euo pipefail

CPU=false
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU=true ;;
        *) echo "Usage: ./run.sh [--cpu]"; exit 1 ;;
    esac
done

# Activate venv (if it exists)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Set local weights directory
export WEIGHTS_DIR="${WEIGHTS_DIR:-$SCRIPT_DIR/weights}"

if [ "$CPU" = true ]; then
    export FORCE_CPU=true
    echo -e "\033[33mRunning in CPU mode (FORCE_CPU=true)\033[0m"
fi

# Run the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
