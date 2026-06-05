from functools import partial
from statistics import mean

import optuna
import pennylane as qml
from torch import nn

from aqml4msc.datasets.mnist import MnistDataset
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.classical_mlp import classical_2l_mlp
from aqml4msc.models.vqa import VQAAnsatz, ansatz_angle_basic
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def hpo_quantum_1():
    def objective(trial):
        model_params = {
            "lr": 1e-3,
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
            "n_qubits": trial.suggest_int("n_qubits", low=4, high=12, step=2),
            "n_layers": trial.suggest_int("n_layers", low=1, high=5),
        }
        # The two extractors each output half of the needed n_qubits
        model_params["output_dim_part"] = model_params["n_qubits"] // 2

        trainer_params = {
            "max_epochs": 30,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
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
            "n_folds": 5,
            "parent_run_name": "QMLP_HPO_1",
            "model_name": "QMLP_1",
        }
        quantum_device = qml.device("default.qubit", model_params["n_qubits"])

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
            VQAAnsatz,
            dev=quantum_device,
            num_classes=model_params["num_classes"],
            ansatz=ansatz_angle_basic(model_params["n_qubits"]),
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
    study.optimize(objective, n_trials=20)
    print(study.best_params)
