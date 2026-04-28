#!/usr/bin/env bash
set -euo pipefail

export UV_ENV_FILE=/app/aqml4msc.env

echo "Starting MLflow server..."
uv run mlflow server \
    --backend-store-uri "${MLFLOW_TRACKING_URI}"\
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --default-artifact-root "${MLFLOW_ARTIFACTS_ROOT}" &
MLFLOW_PID=$!
echo !!

# Ensure MLflow is killed on script exit, error, or Ctrl+C
trap "echo 'Stopping all MLflow processes...'; kill $MLFLOW_PID 2>/dev/null; pkill -f 'uvicorn.*mlflow.server.fastapi_app' 2>/dev/null; wait" EXIT

sleep 5

uv run main.py

# Explicit exit
exit 0