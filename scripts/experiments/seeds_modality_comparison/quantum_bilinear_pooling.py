from functools import partial
from statistics import mean

import pennylane as qml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from aqml4msc.datasets.seeds import SeedsDataset
from aqml4msc.logging import setup_mlflow
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.vqa import (
    VQAAmplStrongProbs_Bilinear_Pool,
)
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def get_default_configs():
    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "n_qubits": [4, 4],
    }

    trainer_params = {
        "max_epochs": 300,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
    }

    data_params = {
        "batch_size": 32,
        "num_workers": 8,
        "pca_img_components": 12,
        "pca_tab_components": 12,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_Multimodal_Bilinear_Pooling",
        "model_name": "BilinearPoolingVQAFusionLinear",
    }

    return model_params, trainer_params, data_params, experiment_params


def run_experiment(model_params, trainer_params, data_params, experiment_params):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # Use a copy or modify dict safely
    trainer_params_run = trainer_params.copy()
    trainer_params_run["callbacks"] = [checkpoint_callback]

    dataset = SeedsDataset(config=data_params)

    extractor_factories = [
        nn.Identity,
        nn.Identity,
    ]
    fusion_factory = partial(
        VQAAmplStrongProbs_Bilinear_Pool,
        dev=qml.device("default.qubit", wires=sum(model_params["n_qubits"])),
        n_qubits=model_params["n_qubits"],
        feat_sizes=[
            data_params["pca_tab_components"],
            data_params["pca_img_components"],
        ],
        num_classes=model_params["num_classes"],
    )
    main_model_factory = partial(
        BaseMLPModel,
        model_params=model_params,
        extractor_factories=extractor_factories,
        fusion_factory=fusion_factory,
    )

    training = MLPTraining(trainer_kwargs=trainer_params_run)

    pipeline = ClassificationPipeline()
    metrics = pipeline.process_data(
        model_factory=main_model_factory,
        dataset=dataset,
        training=training,
        params={
            "experiment_params": experiment_params,
            "data_params": data_params,
            "model_params": model_params,
            "trainer_params": trainer_params_run,
        },
    )

    return mean(metrics["accuracy"])


def manual_quantum_multimodal_bilinear_pooling_1():
    model_params, trainer_params, data_params, experiment_params = get_default_configs()
    return run_experiment(model_params, trainer_params, data_params, experiment_params)


def manual_quantum_multimodal_bilinear_pooling_2():
    model_params, trainer_params, data_params, experiment_params = get_default_configs()

    # Override specific configs for experiment 2
    model_params["n_qubits"] = [3, 3]
    data_params["pca_img_components"] = 6
    data_params["pca_tab_components"] = 6
    experiment_params["parent_run_name"] = "QMLP_Multimodal_6_6_bilinear_pooling"

    return run_experiment(model_params, trainer_params, data_params, experiment_params)


def main() -> None:
    """Calls the experiment."""
    setup_mlflow(experiment_name="Seeds_Multimodal_Classification")
    metrics = manual_quantum_multimodal_bilinear_pooling_1()
    print(metrics)
    metrics = manual_quantum_multimodal_bilinear_pooling_2()
    print(metrics)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
