from collections import defaultdict

from mlflow.tracking import MlflowClient

from aqml4msc.logging.mlflow_utils import EXPERIMENT_NAME, MLFLOW_URI


def latex_escape(s):
    return str(s).replace("_", r"\_")


client = MlflowClient(tracking_uri=MLFLOW_URI)
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Fetch runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],  # type: ignore
    order_by=[],
    max_results=100,
)

# Group runs by parent run name
runs_by_parent = defaultdict(list)
for run in runs:
    parent_id = run.data.tags.get("mlflow.parentRunId", None)
    parent_run_name = "Root"
    if parent_id:
        parent_run = client.get_run(parent_id)
        parent_run_name = parent_run.data.tags.get("mlflow.runName", parent_id)
    runs_by_parent[parent_run_name].append(run)

# Transposed LaTeX tables
for parent_name, group_runs in runs_by_parent.items():
    # Collect all metrics keys in this group
    all_metrics = set()
    for run in group_runs:
        all_metrics.update(run.data.metrics.keys())
    all_metrics = sorted(all_metrics)

    # Include model as a "metric" to show in rows
    all_metrics = ["Model"] + all_metrics

    # Print table header (runs as columns)
    print(f"% Parent run: {parent_name}")
    col_spec = "l" + "c" * len(group_runs)
    print(r"\begin{tabular}{" + col_spec + "}")
    header = ["Metric / Run"] + [
        run.data.tags.get("mlflow.runName", run.info.run_id) for run in group_runs
    ]
    print(" & ".join(header) + r" \\")
    print(r"\hline")

    # Print rows (metrics as rows)
    for metric in all_metrics:
        row = [latex_escape(metric)]
        for run in group_runs:
            if metric == "Model":
                row.append(run.data.tags.get("model", "N/A"))
            else:
                row.append(f"{run.data.metrics.get(metric, float('nan')):.3f}")
        print(" & ".join(row) + r" \\")
    print(r"\end{tabular}")
    print("\n")
