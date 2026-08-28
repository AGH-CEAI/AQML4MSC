from functools import partial
from statistics import mean

import pennylane as qml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from aqml4msc.datasets.seeds import SeedsDataset, SeedsImageDataset, SeedsTabDataset
from aqml4msc.logging import setup_mlflow
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.vqa import (
    VQAAnsatzLinear,
    ansatz_angle_basic,
)
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def manual_quantum_tab_1():

    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "n_qubits": 6,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer_params = {
        "max_epochs": 300,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": [checkpoint_callback],
    }
    data_params = {
        "batch_size": 32,
        "num_workers": 8,
        "pca_tab_components": 6,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_tabular",
        "model_name": "VQAAnsatzLinear",
    }

    # Initialize the dataset with the specified data parameters
    dataset = SeedsTabDataset(config=data_params)

    extractor_factories = [
        nn.Identity,
    ]
    fusion_factory = partial(
        VQAAnsatzLinear,
        dev=qml.device("default.qubit", wires=model_params["n_qubits"]),
        ansatz=ansatz_angle_basic(model_params["n_qubits"]),
        n_qubits_measured=model_params["n_qubits"],
        num_classes=model_params["num_classes"],
        weight_shapes={"weights": (1, model_params["n_qubits"])},
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
        },
    )

    return mean(metrics["accuracy"])


def manual_quantum_images_1():

    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "n_qubits": 6,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer_params = {
        "max_epochs": 300,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": [checkpoint_callback],
    }
    data_params = {
        "batch_size": 32,
        "num_workers": 8,
        "pca_img_components": 6,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_Images",
        "model_name": "VQAAnsatzLinear",
    }

    # Initialize the dataset with the specified data parameters
    dataset = SeedsImageDataset(config=data_params)

    extractor_factories = [
        nn.Identity,
    ]
    fusion_factory = partial(
        VQAAnsatzLinear,
        dev=qml.device("default.qubit", wires=model_params["n_qubits"]),
        ansatz=ansatz_angle_basic(model_params["n_qubits"]),
        n_qubits_measured=model_params["n_qubits"],
        num_classes=model_params["num_classes"],
        weight_shapes={"weights": (1, model_params["n_qubits"])},
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
        },
    )

    return mean(metrics["accuracy"])


def manual_quantum_multimodal_1():

    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "n_qubits": 12,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer_params = {
        "max_epochs": 300,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": [checkpoint_callback],
    }
    data_params = {
        "batch_size": 32,
        "num_workers": 8,
        "pca_img_components": 6,
        "pca_tab_components": 6,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_Multimodal",
        "model_name": "VQAAnsatzLinear",
    }

    # Initialize the dataset with the specified data parameters
    dataset = SeedsDataset(config=data_params)

    extractor_factories = [
        nn.Identity,
        nn.Identity,
    ]
    fusion_factory = partial(
        VQAAnsatzLinear,
        dev=qml.device("default.qubit", wires=model_params["n_qubits"]),
        ansatz=ansatz_angle_basic(model_params["n_qubits"]),
        n_qubits_measured=model_params["n_qubits"],
        num_classes=model_params["num_classes"],
        weight_shapes={"weights": (1, model_params["n_qubits"])},
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
        },
    )

    return mean(metrics["accuracy"])


def manual_quantum_multimodal_2():

    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "n_qubits": 6,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/seeds",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer_params = {
        "max_epochs": 300,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "accelerator": "auto",
        "devices": "auto",
        "callbacks": [checkpoint_callback],
    }
    data_params = {
        "batch_size": 32,
        "num_workers": 8,
        "pca_img_components": 3,
        "pca_tab_components": 3,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_Multimodal_3_3",
        "model_name": "VQAAnsatzLinear",
    }

    # Initialize the dataset with the specified data parameters
    dataset = SeedsDataset(config=data_params)

    extractor_factories = [
        nn.Identity,
        nn.Identity,
    ]
    fusion_factory = partial(
        VQAAnsatzLinear,
        dev=qml.device("default.qubit", wires=model_params["n_qubits"]),
        ansatz=ansatz_angle_basic(model_params["n_qubits"]),
        n_qubits_measured=model_params["n_qubits"],
        num_classes=model_params["num_classes"],
        weight_shapes={"weights": (1, model_params["n_qubits"])},
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
        },
    )

    return mean(metrics["accuracy"])


def main() -> None:
    """Calls the experiment."""
    setup_mlflow(experiment_name="Seeds_Multimodal_Classification")
    metrics = manual_quantum_tab_1()
    print(metrics)
    metrics = manual_quantum_images_1()
    print(metrics)
    metrics = manual_quantum_multimodal_1()
    print(metrics)
    metrics = manual_quantum_multimodal_2()
    print(metrics)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
