import logging
import os
from statistics import mean, stdev
from typing import Any, Dict, TextIO, Tuple

import mlflow
import numpy as np
import pennylane as qml
from aqmlator.tuner import compute_qc_metrics
from metrics import compute_classification_metrics
from mlflow.models import infer_signature
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import classification_report, confusion_matrix

from aqml4msc.training.base_training import BaseTraining

# ------------------------------------------------------------------------------
# FILE REPORTS
# ------------------------------------------------------------------------------


def print_report_to_file(
    file: TextIO, model_name: str, y: np.ndarray, y_pred: np.ndarray
) -> None:
    file.write(f"\nClassification report for {model_name}:\n")
    file.write(str(classification_report(y, y_pred)))


def print_conf_matrix_to_file(
    file: TextIO, model_name: str, y: np.ndarray, y_pred: np.ndarray
) -> None:
    file.write(f"\nConfusion matrix for {model_name}:\n")
    file.write(str(confusion_matrix(y, y_pred)))


def log_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred)
    file_path = "classification_report.txt"

    with open(file_path, "w") as f:
        f.write(str(report))

    mlflow.log_artifact(file_path)
    os.remove(file_path)


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    file_path = "confusion_matrix.txt"

    with open(file_path, "w") as f:
        f.write(np.array2string(matrix))

    mlflow.log_artifact(file_path)
    os.remove(file_path)


# ------------------------------------------------------------------------------
# MLflow Setup
# ------------------------------------------------------------------------------


def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    prepare_mlflow_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def prepare_mlflow_experiment(exp_name: str) -> None:
    if not mlflow.get_experiment_by_name(exp_name):
        create_mlflow_experiment(exp_name)


def create_mlflow_experiment(exp_name: str) -> None:
    tags: Dict[str, Any] = {
        "project_name": "AQML4MSC",
    }

    mlflow.create_experiment(
        name=exp_name, tags=tags, artifact_location=os.environ["MLFLOW_ARTIFACTS_ROOT"]
    )


# TODO(SD): Refactor to use MLFlowLogger directly in the pipeline and training, instead of these helper functions
def get_mlflow_logger() -> MLFlowLogger:
    logger = MLFlowLogger(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        run_id=mlflow.active_run().info.run_id,  # type: ignore
    )
    return logger


def log_params(params: Dict[str, Any]) -> None:
    for _, value in params.items():
        log_nested_params(value)


def log_nested_params(params: Dict[str, Any]) -> None:
    # Remove duplicate, if they happend to
    for k in mlflow.get_run(mlflow.active_run().info.run_id).data.params.keys():  # type: ignore
        if k in params.keys():
            logging.warning(f"Parameter {k} already logged, skipping duplicate.")
            params.pop(k)

    mlflow.log_params(params)


def start_parent_run(model_name: str) -> mlflow.ActiveRun:
    run = mlflow.start_run(run_name=model_name)
    mlflow.set_tag("model", model_name)
    return run


def start_child_hp_run(fold_name: str) -> mlflow.ActiveRun:
    return mlflow.start_run(run_name=fold_name, nested=True)


def log_metrics(metrics: Dict[str, Any]) -> None:
    for metric_name, values in metrics.items():
        mlflow.log_metric(metric_name, values)


def log_aggregated_metrics(all_fold_metrics: dict) -> None:
    for metric_name, values in all_fold_metrics.items():
        mlflow.log_metric(f"{metric_name}_mean", mean(values))
        mlflow.log_metric(f"{metric_name}_std", stdev(values))


def log_model(
    trainer: BaseTraining,
    X_val: Tuple[np.ndarray, np.ndarray],
    model_name: str = "model",
) -> None:
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
