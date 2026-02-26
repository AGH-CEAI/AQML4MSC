import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

from aqml4msc.mlflow_utils import EXPERIMENT_NAME

# =============================
# Configuration
# =============================
TARGET_X_METRICS = ["qlr", "eee", "qmi"]
TARGET_Y_METRIC = "accuracy_avg"


# =============================
# MLflow setup
# =============================
client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=500,
)


# =============================
# Group runs (parent/children)
# =============================
parent_runs = {}
children_by_parent = defaultdict(list)

for run in runs:
    parent_id = run.data.tags.get("mlflow.parentRunId")

    if parent_id is None:
        parent_runs[run.info.run_id] = run
    else:
        children_by_parent[parent_id].append(run)


# =============================
# Aggregate metrics per parent
# =============================
aggregated_data = {metric: {"x": [], "y": []} for metric in TARGET_X_METRICS}

for parent_id, parent_run in parent_runs.items():
    # ---- Filter: parent must be FINISHED ----
    if parent_run.info.status != "FINISHED":
        continue

    child_runs = children_by_parent.get(parent_id, [])
    if not child_runs:
        continue

    # ---- Collect metrics from children ----
    metric_values = {TARGET_Y_METRIC: [], **{m: [] for m in TARGET_X_METRICS}}

    for child in child_runs:
        metrics = child.data.metrics

        # Ensure all required metrics exist in this child
        if not all(m in metrics for m in [TARGET_Y_METRIC] + TARGET_X_METRICS):
            continue

        metric_values[TARGET_Y_METRIC].append(metrics[TARGET_Y_METRIC])
        for m in TARGET_X_METRICS:
            metric_values[m].append(metrics[m])

    # If after filtering no valid folds remain → skip parent
    if len(metric_values[TARGET_Y_METRIC]) == 0:
        continue

    # ---- Compute means over folds ----
    y_mean = np.mean(metric_values[TARGET_Y_METRIC])

    for m in TARGET_X_METRICS:
        x_mean = np.mean(metric_values[m])
        aggregated_data[m]["x"].append(x_mean)
        aggregated_data[m]["y"].append(y_mean)

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

for metric in TARGET_X_METRICS:
    x = np.array(aggregated_data[metric]["x"]).reshape(-1, 1)
    y = np.array(aggregated_data[metric]["y"])

    if len(x) < 3:
        print(f"Not enough data for {metric}")
        continue

    # Pearson
    pearson_r, pearson_p = pearsonr(x.flatten(), y)

    # Spearman
    spearman_r, spearman_p = spearmanr(x.flatten(), y)

    # Linear regression
    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]
    r2 = model.score(x, y)

    print(f"\n=== {metric} ===")
    print(f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}")
    print(f"Slope = {slope:.4f}")
    print(f"R^2 = {r2:.3f}")

# =============================
# Plotting
# =============================
for metric in TARGET_X_METRICS:
    x = aggregated_data[metric]["x"]
    y = aggregated_data[metric]["y"]

    if len(x) == 0:
        print(f"No valid data for metric '{metric}', skipping plot.")
        continue

    plt.figure()
    plt.scatter(x, y)

    plt.xlabel(metric)
    plt.ylabel(TARGET_Y_METRIC)
    plt.title(f"{TARGET_Y_METRIC} vs {metric}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
