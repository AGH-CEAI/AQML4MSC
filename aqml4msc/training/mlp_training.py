import mlflow.pytorch as mlflow_pytorch
import pytorch_lightning as pl
import torch
from datasets.base_dataset import BaseDataset
from mlflow.models import ModelSignature

from aqml4msc import logging
from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.training.base_training import BaseTraining


class MLPTraining(BaseTraining):
    def __init__(self, trainer_kwargs: dict):
        self.trainer_kwargs = trainer_kwargs

    def fit(self, model: BaseMLPModel, dataset: BaseDataset):
        self.trainer = pl.Trainer(
            **self.trainer_kwargs, logger=logging.get_mlflow_logger()
        )
        train_dataloader = dataset.get_train_dataloader()
        val_dataloader = dataset.get_val_dataloader()
        self.trainer.fit(model, train_dataloader, val_dataloader)

    def predict(self, model: BaseMLPModel, dataset: BaseDataset):
        dataloader = dataset.get_test_dataloader()
        preds = self.trainer.predict(model, dataloader)
        return torch.cat(preds, dim=0).cpu().numpy()  # type: ignore

    def log_model(
        self, model: BaseMLPModel, model_name: str, signature: ModelSignature
    ):
        mlflow_pytorch.log_model(model, name=model_name, signature=signature)

    def get_n_paramas(self, model: BaseMLPModel) -> dict:
        """Reurns dict with number of trainable parameters"""
        dict = {}
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            dict[name] = params
            total_params += params
        return dict
