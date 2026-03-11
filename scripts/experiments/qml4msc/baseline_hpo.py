from statistics import mean

import optuna
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn

from aqml4msc.data import choose_digits, load_data
from aqml4msc.logging import EpochMetricsTracker
from aqml4msc.models.classical_mlp import CMLP_1
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def hpo_baseline_1():
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
            "max_epochs": 30,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
            "callbacks": [EpochMetricsTracker()],
            "logger": False,
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
            "parent_run_name": "HPO_classical_MLP",
            "model_name": "Classical_MLP_baseline",
        }

        training = MLPTraining(
            model_cls=CMLP_1,
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
            optuna_params=trial.params,
        )

        return mean(metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(study.best_params)


def hpo_baseline_2():
    def objective(trial):
        model_params = {
            "lr": trial.suggest_float("lr", 1e-3, 1e-2),
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
            "output_dim_part": trial.suggest_int("output_dim_part", 64, 256),
            "hidden_dim_class": [trial.suggest_int("hidden_dim_class", 64, 256)],
        }

        trainer_params = {
            "max_epochs": 50,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
            "callbacks": [
                EpochMetricsTracker(),
                EarlyStopping(monitor="val_loss", mode="min"),
            ],
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
            "parent_run_name": "HPO_classical_MLP_2",
            "model_name": "Classical_MLP_baseline_2",
        }

        training = MLPTraining(
            model_cls=CMLP_1,
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
    study.optimize(objective, n_trials=1000)
    print(study.best_params)


def hpo_baseline_3():
    def objective(trial):
        model_params = {
            "lr": trial.suggest_float("lr", 1e-3, 1e-2),
            "loss_fn": nn.CrossEntropyLoss(),
            "num_classes": 3,
            "input_dim": 14,
            "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
            "output_dim_part": trial.suggest_int("output_dim_part", 64, 256),
            "hidden_dim_class": [trial.suggest_int("hidden_dim_class", 64, 256)],
        }

        trainer_params = {
            "max_epochs": 30,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "num_sanity_val_steps": 0,
            "callbacks": [
                EpochMetricsTracker(),
                # EarlyStopping(monitor="val_loss", mode="min"),
            ],
            "logger": False,
            "accelerator": "auto",
            "devices": "auto",
        }
        data_params = {
            "batch_size": trial.suggest_int("batch_size", 32, 128),
            "num_workers": 14,
            "digits": [5, 6, 7],
        }

        experiment_params = {
            "seed": 42,
            "n_folds": 5,
            "parent_run_name": "HPO_classical_MLP_3",
            "model_name": "Classical_MLP_baseline_3",
        }

        training = MLPTraining(
            model_cls=CMLP_1,
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
    study.optimize(objective, n_trials=100)
    print(study.best_params)
