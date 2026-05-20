from functools import partial
from statistics import mean
from typing import Any, Callable

import optuna
from aqmlator.qml import AnsatzBuilder
from aqmlator.tuner import AnsatzFinder
from datasets.mnist import MnistDataset
from torch import nn

from aqml4msc import logging
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.classical_mlp import ConcatMLPFusion, classical_2l_mlp
from aqml4msc.models.vqa import ConcatVQAFusion
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining

EXPERIMENT_NAME = "TEST_SD"


def hpo_quantum_test():
    print("\n\n***** hpo_quantum_test START *****\n\n")

    def objective(trial):
        # Set configurations
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
            # "callbacks": [EpochMetricsTracker()],
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

        model_params["output_dim_part"] = model_params["n_qubits"] // 2

        extractor_factories = [
            partial(
                classical_2l_mlp,
                model_params["input_dim"],
                model_params["hidden_dim_part"],
                model_params["output_dim_part"],
            ),
            partial(
                classical_2l_mlp,
                model_params["input_dim"],
                model_params["hidden_dim_part"],
                model_params["output_dim_part"],
            ),
        ]

        fusion_factory = partial(
            ConcatVQAFusion,
            model_params["n_qubits"],
            model_params["num_classes"],
        )

        main_model_factory = partial(
            BaseMLPModel,
            model_params=model_params,
            extractor_factories=extractor_factories,
            fusion_factory=fusion_factory,
        )

        # Initialize the trainer with model and training
        training = MLPTraining(trainer_kwargs=trainer_params)

        # Initialize the dataset with the specified data parameters
        dataset = MnistDataset(config=data_params)

        # Initialize the classification pipeline: ClassificationPipeline
        pipeline = ClassificationPipeline()
        metrics = pipeline.process_data(
            model_factory=main_model_factory,
            dataset=dataset,
            training=training,
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
        # "callbacks": [EpochMetricsTracker()],
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

    model_params["output_dim_part"] = model_params["n_qubits"] // 2

    ansatz = test_suggest_ansatz(trial)

    extractor_factories = [
        partial(
            classical_2l_mlp,
            model_params["input_dim"],
            model_params["hidden_dim_part"],
            model_params["output_dim_part"],
        ),
        partial(
            classical_2l_mlp,
            model_params["input_dim"],
            model_params["hidden_dim_part"],
            model_params["output_dim_part"],
        ),
    ]

    fusion_factory = partial(
        ConcatVQAFusion,
        model_params["n_qubits"],
        model_params["num_classes"],
        ansatz=ansatz,
    )

    main_model_factory = partial(
        BaseMLPModel,
        model_params=model_params,
        extractor_factories=extractor_factories,
        fusion_factory=fusion_factory,
    )

    # Initialize the trainer with model and training: MLPTraining parameters
    training = MLPTraining(trainer_kwargs=trainer_params)

    # Initialize the dataset with the specified data parameters
    dataset = MnistDataset(config=data_params)

    # Initialize the classification pipeline: ClassificationPipeline
    pipeline = ClassificationPipeline()

    # Execute the pipeline to process data, train, and evaluate the model
    metrics: dict[str, list[float]] = pipeline.process_data(
        model_factory=main_model_factory,
        dataset=dataset,
        training=training,
        params={
            "experiment_params": experiment_params,
            "data_params": data_params,
            "model_params": model_params,
            "trainer_params": trainer_params,
            "optuna_params": trial.params,
        },
    )

    # Return the mean accuracy across folds as the optimization objective
    return mean(metrics["accuracy"])


def test_ansatz_search_test() -> None:
    print("\n\n***** test_ansatz_search_test START *****\n\n")
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(test_optuna_aqml_objective, n_trials=2)
    print(study.best_params)
    print("\n\n***** test_ansatz_search_test END *****\n\n")


def test_hpo_baseline_1():
    def objective(trial):
        model_params = {
            "lr": trial.suggest_float("lr", 1e-3, 1e-2),
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [
                trial.suggest_categorical("hidden_dim_part", [64, 128, 256])
            ],
            "output_dim_part": trial.suggest_categorical(
                "output_dim_part", [64, 128, 256]
            ),
            "hidden_dim_class": [
                trial.suggest_categorical("hidden_dim_class", [64, 128, 256])
            ],
        }

        trainer_params = {
            "max_epochs": 2,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
            "accelerator": "auto",
            "devices": "auto",
        }
        data_params = {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_workers": 4,
            "digits": [5, 6, 7],
        }

        experiment_params = {
            "seed": 42,
            "n_folds": 5,
            "parent_run_name": "TEST_QMLP_AQML_Classical_Output",
            "model_name": "Classical_MLP_baseline",
        }

        extractor_factories = [
            partial(
                classical_2l_mlp,
                model_params["input_dim"],
                model_params["hidden_dim_part"],
                model_params["output_dim_part"],
            ),
            partial(
                classical_2l_mlp,
                model_params["input_dim"],
                model_params["hidden_dim_part"],
                model_params["output_dim_part"],
            ),
        ]

        fusion_factory = partial(
            ConcatMLPFusion,
            2 * model_params["output_dim_part"],
            model_params["hidden_dim_class"],
            model_params["num_classes"],
        )
        main_model_factory = partial(
            BaseMLPModel,
            model_params=model_params,
            extractor_factories=extractor_factories,
            fusion_factory=fusion_factory,
        )

        training = MLPTraining(trainer_kwargs=trainer_params)

        # Initialize the dataset with the specified data parameters
        dataset = MnistDataset(config=data_params)

        # Initialize the classification pipeline: ClassificationPipeline
        pipeline = ClassificationPipeline()
        metrics = pipeline.process_data(
            model_factory=main_model_factory,
            dataset=dataset,
            training=training,
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


if __name__ == "__main__":
    print("Experiment start")
    logging.setup_mlflow(EXPERIMENT_NAME)
    test_hpo_baseline_1()
    # hpo_quantum_test()
    # test_ansatz_search_test()
    print("Experiment finished")
