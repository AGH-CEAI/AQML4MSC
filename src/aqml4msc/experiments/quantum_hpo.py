from statistics import mean

import optuna
from torch import nn

from aqml4msc.data.loading import choose_digits, load_data
from aqml4msc.logging.mlflow_utils import EpochMetricsTracker
from aqml4msc.models.vqa import QMLP_1
from aqml4msc.pipeline.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def hpo_quantum_1():
    def objective(trial):
        model_params = {
            "lr": 1e-3,
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
            "n_qubits": trial.suggest_int("n_qubits", low=4, high=16, step=2),
            "n_layers": trial.suggest_int("n_layers", low=1, high=5),
        }

        trainer_params = {
            "max_epochs": 30,
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
            "n_folds": 5,
            "parent_run_name": "QMLP_HPO_1",
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
            experiment_params=experiment_params,
            data_params=data_params,
            model_params=model_params,
            trainer_params=trainer_params,
        )

        return mean(metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(study.best_params)
