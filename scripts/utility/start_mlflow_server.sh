
uv run --env-file aqml4msc.env -- bash -c '
mlflow server \
    --backend-store-uri "${MLFLOW_TRACKING_URI}"\
    --host 127.0.0.1 \
    --port 5000 \
    --workers 1 \
    --default-artifact-root "${MLFLOW_ARTIFACTS_ROOT}"
'