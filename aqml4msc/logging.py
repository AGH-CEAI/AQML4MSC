import os
from statistics import mean, stdev
from typing import Any, Dict, TextIO, Tuple

import mlflow
import numpy as np
import pennylane as qml
from aqmlator.tuner import compute_qc_metrics
from metrics import compute_classification_metrics
from mlflow.models import infer_signature
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix

from aqml4msc.training.base_training import BaseTraining

EXPERIMENT_NAME = "MNIST_Multisource_Classification"
# MLFLOW_URI = "http://localhost:5001"


# ------------------------------------------------------------------------------
# FILE REPORTS
# ------------------------------------------------------------------------------


def print_report_to_file(file: TextIO, model_name: str, y, y_pred):
    file.write(f"\nClassification report for {model_name}:\n")
    file.write(str(classification_report(y, y_pred)))


def print_conf_matrix_to_file(file: TextIO, model_name: str, y, y_pred):
    file.write(f"\nConfusion matrix for {model_name}:\n")
    file.write(str(confusion_matrix(y, y_pred)))


def log_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    file_path = "classification_report.txt"

    with open(file_path, "w") as f:
        f.write(str(report))

    mlflow.log_artifact(file_path)
    os.remove(file_path)


def log_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    file_path = "confusion_matrix.txt"

    with open(file_path, "w") as f:
        f.write(np.array2string(matrix))

    mlflow.log_artifact(file_path)
    os.remove(file_path)


# ------------------------------------------------------------------------------
# MLflow Setup
# ------------------------------------------------------------------------------


def setup_mlflow():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    prepare_mlflow_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def prepare_mlflow_experiment(exp_name: str) -> None:
    """
    _summary_ TODO
    """
    if not mlflow.get_experiment_by_name(exp_name):
        create_mlflow_experiment(exp_name)


def create_mlflow_experiment(exp_name: str) -> None:
    """
    _summary_ TODO

    :param exp_name: _description_
    :type exp_name: str
    """
    tags: Dict[str, Any] = {
        "project_name": "AQML4MSC",
    }

    mlflow.create_experiment(
        name=exp_name, tags=tags, artifact_location=os.environ["MLFLOW_ARTIFACTS_ROOT"]
    )


def log_params(params):
    # Remove duplicate, if they happend to
    for k in mlflow.get_run(mlflow.active_run().info.run_id).data.params.keys():  # type: ignore
        if k in params.keys():
            params.pop(k)

    mlflow.log_params(params)


def start_parent_run(model_name: str) -> mlflow.ActiveRun:
    run = mlflow.start_run(run_name=model_name)
    mlflow.set_tag("model", model_name)
    return run


def start_child_hp_run(fold_name: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=fold_name, nested=True)


def log_metrics(metrics: dict):
    for metric_name, values in metrics.items():
        mlflow.log_metric(metric_name, values)


def log_aggregated_metrics(all_fold_metrics: dict):
    for metric_name, values in all_fold_metrics.items():
        mlflow.log_metric(f"{metric_name}_mean", mean(values))
        mlflow.log_metric(f"{metric_name}_std", stdev(values))


def log_model(
    trainer: BaseTraining,
    X_val: Tuple[np.ndarray, np.ndarray],
    model_name: str = "model",
):
    signature = infer_signature(X_val, trainer.predict(X_val))
    trainer.log_model(model_name=model_name, signature=signature)


def log_all_run_metrics(
    metrics: list,
    true_labels: np.ndarray,
    preds: np.ndarray,
    val_data: tuple,
    fold: int,
    classifier: BaseTraining,
    model_name: str,
    ansatz,
) -> list:
    metrics.append(compute_classification_metrics(y_true=true_labels, y_pred=preds))

    if ansatz is not None:
        metrics[-1].update(
            compute_qc_metrics(qml.QNode(ansatz, device=classifier.model.dev))
        )
    log_metrics(metrics[fold - 1])
    try:
        log_classification_report(y_true=true_labels, y_pred=preds)
        log_confusion_matrix(y_true=true_labels, y_pred=preds)
        log_model(
            trainer=classifier,
            X_val=val_data,
            model_name=model_name,
        )
    except Exception as e:
        print(f"Could not save artifacts. Error occured: {e}")

    return metrics


# ------------------------------------------------------------------------------
# Callback: Collect + Log Epoch Metrics
# ------------------------------------------------------------------------------


class EpochMetricsTracker(Callback):
    """
    Collects train/val epoch metrics and logs them ONCE per fold.
    This completely removes the need for external log_epochs_metrics().
    """

    def __init__(self):
        super().__init__()
        self.train_epoch_metrics = []
        self.val_epoch_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics: dict[str, float | int] = {"epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train_"):
                metrics[key] = float(value.item())
        self.train_epoch_metrics.append(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics: dict[str, float | int] = {"epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val_"):
                metrics[key] = float(value.item())
        self.val_epoch_metrics.append(metrics)

    def on_fit_end(self, trainer, pl_module):
        # Log all epoch metrics at the end of the fold
        for entry in self.train_epoch_metrics + self.val_epoch_metrics:
            epoch = entry["epoch"]
            for k, v in entry.items():
                if k != "epoch":
                    mlflow.log_metric(k, v, step=epoch)
