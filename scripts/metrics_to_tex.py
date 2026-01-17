import os
from collections import defaultdict

from mlflow.tracking import MlflowClient

from aqml4msc.logging.mlflow_utils import EXPERIMENT_NAME

SEARCHED_RUNS_IDS = {"8dd441d425f844cba9dad329b54c68e3"}


# -----------------------------
# Utilities
# -----------------------------
def latex_escape(s) -> str:
    return str(s).replace("_", r"\_")


# -----------------------------
# MLflow setup
# -----------------------------
client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=500,
)

# -----------------------------
# Group runs
# -----------------------------
parent_runs = {}
children_by_parent = defaultdict(list)

for run in runs:
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if parent_id is None:
        if not bool(SEARCHED_RUNS_IDS) or run.info.run_id in SEARCHED_RUNS_IDS:
            parent_runs[run.info.run_id] = run
    else:
        if not bool(SEARCHED_RUNS_IDS) or parent_id in SEARCHED_RUNS_IDS:
            children_by_parent[parent_id].append(run)


# -----------------------------
# Generate LaTeX
# -----------------------------
for parent_id, parent_run in parent_runs.items():
    child_runs = children_by_parent.get(parent_id, [])
    if not child_runs:
        continue

    run_name = parent_run.data.tags.get("mlflow.runName", parent_run.info.run_id)

    # ==================================================
    # PARAMETERS (PARENT)
    # ==================================================
    print(f"% ===== Hyperparameters: {latex_escape(run_name)} =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Hyperparameters for experiment " + latex_escape(run_name) + r"}")
    print(r"\begin{tabular}{ll}")
    print(r"\hline")
    print(r"Parameter & Mean Value \\")
    print(r"\hline")

    for k, v in sorted(parent_run.data.params.items()):
        print(
            f"{latex_escape(k)} & "
            r"\texttt{" + latex_escape(v) + r"} \\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ==================================================
    # METRICS (PARENT)
    # ==================================================
    print(f"% ===== Final Metrics: {latex_escape(run_name)} =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Final metrics for " + latex_escape(run_name) + r"}")
    print(r"\begin{tabular}{lc}")
    print(r"\hline")
    print(r"Metric & Value \\")
    print(r"\hline")

    for metric, value in sorted(parent_run.data.metrics.items()):
        if metric.endswith("_mean"):
            base_name = metric[:-5]
            std_metric = f"{base_name}_std"
            std_value = parent_run.data.metrics[std_metric]
            print(
                f"{latex_escape(base_name)} & ${value:.3f} \\pm {std_value:.3f}$ \\\\"
            )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    # ==================================================
    # METRICS (CHILD RUNS / FOLDS)
    # ==================================================
    metric_names = sorted({m for run in child_runs for m in run.data.metrics.keys()})

    fold_names = [
        run.data.tags.get("mlflow.runName", f"Fold {i + 1}")
        for i, run in enumerate(child_runs)
    ]

    print(f"% ===== CV Metrics per Fold: {latex_escape(run_name)} =====")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(
        r"\caption{Cross-validation metrics per fold for "
        + latex_escape(run_name)
        + r"}"
    )

    col_spec = "l" + "c" * len(child_runs)
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\hline")

    header = ["Metric"] + [latex_escape(n) for n in fold_names]
    print(" & ".join(header) + r" \\")
    print(r"\hline")

    for metric in metric_names:
        row = [latex_escape(metric)]
        for run in child_runs:
            value = run.data.metrics.get(metric)
            row.append(f"{value:.3f}" if value is not None else "--")
        print(" & ".join(row) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("\n")
