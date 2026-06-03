from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
from scipy.stats import t
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
) -> Dict[str, float]:
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
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        # Recall variants
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
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


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)

    # Calculate sample variance
    var = np.var(differences, ddof=1)

    # Correction factor for repeated cross validation
    correction = (1 / kr) + (n_test / n_train)

    # Corrected standard deviation
    corrected_std_val = np.sqrt(var * correction)

    return corrected_std_val


def corrected_ttest_two_tailed(res_1, res_2, n_train, n_test):
    """Computes a two-tailed corrected t-test for repeated cross-validation."""
    differences = np.array(res_1) - np.array(res_2)
    df = len(differences) - 1

    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)

    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df) * 2  # two-tailed
    return t_stat, p_val


def corrected_ttest_one_tailed(res_1, res_2, n_train, n_test, alternative="greater"):
    """
    Computes a one-tailed corrected t-test for repeated cross-validation.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.
    alternative : str, {'greater', 'less'}
        The alternative hypothesis.
        'greater' tests if the mean of differences is significantly > 0.
        'less' tests if the mean of differences is significantly < 0.

    Returns
    -------
    t_stat : float
        The corrected t-statistic.
    p_value : float
        The one-tailed p-value.
    """
    differences = np.array(res_1) - np.array(res_2)
    df = len(differences) - 1

    mean_diff = np.mean(differences)
    std_corr = corrected_std(differences, n_train, n_test)

    # Calculate the t-statistic
    t_stat = mean_diff / std_corr

    # Calculate the one-tailed p-value
    if alternative == "greater":
        p_value = t.sf(t_stat, df)  # equivalent to 1 - t.cdf
    elif alternative == "less":
        p_value = t.cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    return t_stat, p_value
