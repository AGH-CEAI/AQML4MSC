"""
This module implements AQML approach for ansatz finding using :mod:`aqmlator` package.

.. note::

    Bases heavily on ``src.aqml4msc.experiments``.

.. hint::

    TODO(SD): Consider refactoring the code, so that both are using single implementation of the
    common part of the code.
"""

from statistics import mean
from typing import Any, Callable

import optuna
from aqmlator.qml import AnsatzBuilder
from aqmlator.tuner import AnsatzFinder
from torch import nn

from aqml4msc.data import choose_digits, load_data
from aqml4msc.mlflow_utils import EpochMetricsTracker
from aqml4msc.models.vqa import QMLP_1
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def suggest_ansatz(trial: optuna.Trial) -> Callable[..., Any]:
    """
    Use AQML methods implemented in :mod:`aqmlator` to suggest, build and return an ansatz.

    .. important::
        Ansatze suggested by :mod:`aqmlator` are without measurements! Remember to add them
        prior to use!

    :param trial: _description_
    :type trial: optuna.Trial

    :return:
        Ansatz suggested by the :mod:`aqmlator`.
    :rtype: callable
    """
    ansatz_finder: AnsatzFinder = AnsatzFinder(
        n_wires=trial.params["n_qubits"],
        n_min_blocks=trial.params["n_layers"],
        n_max_blocks=trial.params["n_layers"],
    )

    ansatz_recipe: dict[str, Any] = ansatz_finder.suggest_ansatz(trial)
    return AnsatzBuilder.from_recipe(ansatz_recipe)


def optuna_aqml_objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna study that uses the AQML approach to suggest ansatze
    and trains a QMLP_1 model for classification.

    This function defines hyperparameters using Optuna's `suggest` methods,
    initializes training and data parameters, loads and preprocesses data,
    creates a classification pipeline, and evaluates the model's performance
    by computing the mean accuracy over multiple folds.

    :param trial: An Optuna trial object used for hyperparameter optimization.
    :type trial: optuna.Trial

    :return: The mean accuracy over all folds as the objective value.
    :rtype: float
    """

    # Define model parameters, including hyperparameters tuned by Optuna
    model_params: dict[str, Any] = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 14,
        "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
        "n_qubits": trial.suggest_int("n_qubits", low=4, high=12, step=2),
        "n_layers": trial.suggest_int("n_layers", low=1, high=5),
    }

    # Define trainer configuration parameters
    trainer_params: dict[str, Any] = {
        "max_epochs": 30,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "callbacks": [EpochMetricsTracker()],
        "logger": False,
        "accelerator": "auto",
        "devices": "auto",
    }

    import torch

    if torch.backends.mps.is_available():
        trainer_params["accelerator"] = "cpu"  # Pennylane HATES Macs

    # Define data loading and preprocessing parameters
    data_params: dict[str, Any] = {
        "batch_size": trial.suggest_int("batch_size", 32, 128),
        "num_workers": 8,
        "digits": [5, 6, 7],
    }

    # Define experiment configuration parameters
    experiment_params: dict[str, int | str] = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_AQML_test",
        "model_name": "QMLP_1",
    }

    # Initialize the trainer with model and training: MLPTraining parameters
    training = MLPTraining(
        model_cls=QMLP_1,
        model_kwargs=model_params,
        trainer_kwargs=trainer_params,
        batch_size=data_params["batch_size"],
    )

    # Load and preprocess the dataset
    X, y = load_data()
    X, y = choose_digits(X, y, data_params["digits"])

    # Initialize the classification pipeline: ClassificationPipeline
    pipeline = ClassificationPipeline()

    ansatz = suggest_ansatz(trial)

    # Execute the pipeline to process data, train, and evaluate the model
    metrics: dict[str, list[float]] = pipeline.process_data(
        X=X,
        y=y,
        classifier=training,
        experiment_params=experiment_params,
        data_params=data_params,
        model_params=model_params,
        trainer_params=trainer_params,
        optuna_params=trial.params,
        ansatz=ansatz,
    )

    # Return the mean accuracy across folds as the optimization objective
    return mean(metrics["accuracy"])


def main() -> None:
    """Calls the experiment."""
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(optuna_aqml_objective, n_trials=20)
    print(study.best_params)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
