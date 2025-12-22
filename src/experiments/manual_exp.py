from torch import nn

from src.data_loading import choose_digits, load_data
from src.logging_utils import EpochMetricsTracker
from src.models.vqa import QMLP_1
from src.pipeline import ClassificationPipeline
from src.training.mlp_training import MLPTraining


def manual_exp_1():
    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 14,
        "hidden_dim": [128],
        "n_qubits": 6,
        "n_layers": 3,
    }

    trainer_params = {
        "max_epochs": 100,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "callbacks": [EpochMetricsTracker()],
        "logger": False,
        "accelerator": "auto",
        "devices": "auto",
    }
    data_params = {
        "batch_size": 32,
        "num_workers": 4,
        "digits": [5, 6, 7],
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_initial_run",
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
    # ustawić srednie metryki
    # logowanie parametrów i moze moedlu
