from collections import defaultdict
from typing import Any, Dict, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:  # TODO(SD): Is there really anything other than float? Why Any?
    """
    Computes standard classification metrics in a consistent dictionary format.
    """
    metrics = {
        # Accuracy variants
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_avg": balanced_accuracy_score(y_true, y_pred),
        # F1-score variants
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # Precision variants
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        # Recall variants
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics


def aggregate_fold_metrics(
    fold_metrics: Iterable[Dict[str, float]],
) -> Dict[str, list[float]]:
    """
    Convert iterable of per-fold metric dicts into dict of metric -> list of values.
    """
    aggregated = defaultdict(list)

    for metrics in fold_metrics:
        for name, value in metrics.items():
            aggregated[name].append(float(value))

    return dict(aggregated)
