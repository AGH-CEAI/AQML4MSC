from functools import partial
from statistics import mean

from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from aqml4msc.datasets.seeds import SeedsDataset
from aqml4msc.logging import setup_mlflow
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.models.classical_mlp import TNormMLPFusion, classical_1l_sigmoid_mlp
from aqml4msc.models.t_norms import godel_t_norm, lukasiewicz_t_norm, product_t_norm
from aqml4msc.pipeline import ClassificationPipeline
from aqml4msc.training.mlp_training import MLPTraining


def get_default_configs():
    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 12,
        "hidden_dim_part": [0],
        "output_dim_part": 12,
        "hidden_dim_class": [100],
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
        "pca_img_components": 12,
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "Classical_t_norm_fusion",
        "model_name": "TNormMLPFusion",
    }

    return model_params, trainer_params, data_params, experiment_params


def run_t_norm_experiment(t_norm: callable, parent_run_name: str):
    model_params, trainer_params, data_params, experiment_params = get_default_configs()
    experiment_params["parent_run_name"] = parent_run_name

    # Initialize the dataset with the specified data parameters
    dataset = SeedsDataset(config=data_params)

    extractor_factories = [
        partial(
            classical_1l_sigmoid_mlp,
            model_params["input_dim"],
            model_params["output_dim_part"],
        ),
        partial(
            classical_1l_sigmoid_mlp,
            model_params["input_dim"],
            model_params["output_dim_part"],
        ),
    ]

    fusion_factory = partial(
        TNormMLPFusion,
        model_params["output_dim_part"],
        model_params["hidden_dim_class"],
        model_params["num_classes"],
        t_norm=t_norm,
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
    setup_mlflow(experiment_name="Seeds_t_norm_fusion")
    metrics = run_t_norm_experiment(product_t_norm, "Classical_t_norm_product")
    print(metrics)
    metrics = run_t_norm_experiment(godel_t_norm, "Classical_t_norm_godel")
    print(metrics)
    metrics = run_t_norm_experiment(lukasiewicz_t_norm, "Classical_t_norm_lukasiewicz")
    print(metrics)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
