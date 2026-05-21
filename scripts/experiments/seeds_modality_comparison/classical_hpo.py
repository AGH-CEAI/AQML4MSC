from functools import partial
from statistics import mean

import optuna
from datasets.seeds import SeedsTabDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from aqml4msc.logging import setup_mlflow
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.classical_mlp import ConcatMLPFusion, classical_2l_mlp
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def hpo_classical_tab_objective(trial):

    model_params = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-2),
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 12,
        "hidden_dim_part": [
            trial.suggest_categorical("hidden_dim_part", [64, 128, 256])
        ],
        "output_dim_part": trial.suggest_categorical("output_dim_part", [64, 128, 256]),
        "hidden_dim_class": [
            trial.suggest_categorical("hidden_dim_class", [64, 128, 256])
        ],
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer_params = {
        "max_epochs": 50,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": [checkpoint_callback],
    }
    data_params = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_workers": 8,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "Classical_MLP_tabular",
        "model_name": "Classical_MLP_baseline",
    }

    # Initialize the dataset with the specified data parameters
    dataset = SeedsTabDataset(config=data_params)

    extractor_factories = [
        partial(
            classical_2l_mlp,
            model_params["input_dim"],
            model_params["hidden_dim_part"],
            model_params["output_dim_part"],
        ),
    ]

    fusion_factory = partial(
        ConcatMLPFusion,
        model_params["output_dim_part"],
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


def main() -> None:
    """Calls the experiment."""
    setup_mlflow(experiment_name="Seeds_Multimodal_Classification")
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(hpo_classical_tab_objective, n_trials=2)
    print(study.best_params)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
