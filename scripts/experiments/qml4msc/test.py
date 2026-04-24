from statistics import mean
from typing import Any, Callable

import optuna
from aqmlator.qml.models import AnsatzBuilder
from aqmlator.tuner import AnsatzFinder
from torch import nn

from aqml4msc.data import choose_digits, load_data
from aqml4msc.logging import EpochMetricsTracker
from aqml4msc.models.vqa import QMLP_1
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def hpo_quantum_test():
    print("\n\n***** hpo_quantum_test START *****\n\n")

    def objective(trial):
        model_params = {
            "lr": 1e-3,
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
            "n_qubits": trial.suggest_int("n_qubits", low=4, high=4, step=2),
            "n_layers": trial.suggest_int("n_layers", low=1, high=2),
        }

        trainer_params = {
            "max_epochs": 2,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
            "callbacks": [EpochMetricsTracker()],
            "logger": False,
            "accelerator": "auto",
            "devices": "auto",
        }
        data_params = {
            "batch_size": trial.suggest_int("batch_size", 32, 128),
            "num_workers": 8,
            "digits": [5, 6, 7],
        }

        experiment_params = {
            "seed": 42,
            "n_folds": 2,
            "parent_run_name": "TEST_QMSC",
            "model_name": "QMLP_1",
        }

        training = MLPTraining(
            model_cls=QMLP_1,
            model_kwargs=model_params,
            trainer_kwargs=trainer_params,
            batch_size=data_params["batch_size"],
        )

        X, y = load_data()
        X, y = choose_digits(X, y, data_params["digits"])
        pipeline = ClassificationPipeline()
        metrics = pipeline.process_data(
            X=X,
            y=y,
            classifier=training,
            params={
                "experiment_params": experiment_params,
                "data_params": data_params,
                "model_params": model_params,
                "trainer_params": trainer_params,
                "optuna_params": trial.params,
            },
        )

        return mean(metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    print(study.best_params)
    print("\n\n***** hpo_quantum_test END *****\n\n")


def test_suggest_ansatz(trial: optuna.Trial) -> Callable[..., Any]:

    ansatz_finder: AnsatzFinder = AnsatzFinder(
        n_wires=trial.params["n_qubits"],
        n_min_blocks=trial.params["n_layers"],
        n_max_blocks=trial.params["n_layers"],
    )

    ansatz_recipe: dict[str, Any] = ansatz_finder.suggest_ansatz(trial)
    return AnsatzBuilder.from_recipe(ansatz_recipe)


def test_optuna_aqml_objective(trial: optuna.Trial) -> float:
    # Define model parameters, including hyperparameters tuned by Optuna
    model_params: dict[str, Any] = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 14,
        "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
        "n_qubits": trial.suggest_int("n_qubits", low=4, high=6, step=2),
        "n_layers": trial.suggest_int("n_layers", low=1, high=2),
    }

    # Define trainer configuration parameters
    trainer_params: dict[str, Any] = {
        "max_epochs": 2,
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
        "n_folds": 2,
        "parent_run_name": "TEST_QMLP_AQML_Classical_Output",
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

    ansatz = test_suggest_ansatz(trial)

    # Execute the pipeline to process data, train, and evaluate the model
    metrics: dict[str, list[float]] = pipeline.process_data(
        X=X,
        y=y,
        classifier=training,
        params={
            "experiment_params": experiment_params,
            "data_params": data_params,
            "model_params": model_params,
            "trainer_params": trainer_params,
            "optuna_params": trial.params,
        },
        ansatz=ansatz,
    )

    # Return the mean accuracy across folds as the optimization objective
    return mean(metrics["accuracy"])


def test_ansatz_search_test() -> None:
    print("\n\n***** test_ansatz_search_test START *****\n\n")
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(test_optuna_aqml_objective, n_trials=2)
    print(study.best_params)
    print("\n\n***** test_ansatz_search_test END *****\n\n")


if __name__ == "__main__":
    print("Experiment start")
    hpo_quantum_test()
    test_ansatz_search_test()
    print("Experiment finished")
