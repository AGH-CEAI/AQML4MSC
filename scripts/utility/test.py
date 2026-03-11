from torch import nn
from training.mlp_training import MLPTraining

from aqml4msc.logging import EpochMetricsTracker
from aqml4msc.models.classical_mlp import CMLP_1

model_params = {
    "lr": 0.0012191286815755042,
    "loss_fn": nn.CrossEntropyLoss(),
    "num_classes": 3,
    "input_dim": 14,
    "hidden_dim_part": [128],
    "output_dim_part": 128,
    "hidden_dim_class": [64],
}

trainer_params = {
    "max_epochs": 50,
    "enable_checkpointing": False,
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
    "batch_size": 64,
    "num_workers": 8,
    "digits": [5, 6, 7],
}

experiment_params = {
    "seed": 42,
    "n_folds": 5,
    "parent_run_name": "Classical_MLP_best_hparams",
    "model_name": "Classical_MLP_best_hparams",
}

training = MLPTraining(
    model_cls=CMLP_1,
    model_kwargs=model_params,
    trainer_kwargs=trainer_params,
    batch_size=data_params["batch_size"],
)
training.reset_model()

print(training.get_n_paramas())
# X, y = load_data()
# X, y = choose_digits(X, y, data_params["digits"])
# pipeline = ClassificationPipeline()
# metrics = pipeline.process_data(
#     X=X,
#     y=y,
#     classifier=training,
#     experiment_params=experiment_params,
#     data_params=data_params,
#     model_params=model_params,
#     trainer_params=trainer_params,
# )
