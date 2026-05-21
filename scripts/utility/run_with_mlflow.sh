#!/usr/bin/env bash
set -euo pipefail

export UV_ENV_FILE=/app/aqml4msc.env

if [ $# -eq 0 ]; then
    uv run main.py
else
    for script in "$@"; do
        echo "Running $script..."
        uv run "$script"
    done
fi

# Explicit exit
exit 0