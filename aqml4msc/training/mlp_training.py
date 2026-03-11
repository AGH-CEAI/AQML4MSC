from typing import Tuple, Type

import mlflow.pytorch as mlflow_pytorch
import numpy as np
import pytorch_lightning as pl
import torch
from mlflow.models import ModelSignature

from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils import get_dataloader


class MLPTraining(BaseTraining):
    def __init__(
        self, model_cls: Type, model_kwargs: dict, trainer_kwargs: dict, batch_size: int
    ):
        super().__init__(model_cls=model_cls, model_kwargs=model_kwargs)
        self.trainer_kwargs = trainer_kwargs
        self.batch_size = batch_size

    def fit(
        self, train_data: Tuple, train_y: np.ndarray, val_data: Tuple, val_y: np.ndarray
    ):
        self.trainer = pl.Trainer(**self.trainer_kwargs)
        train_dataloader = get_dataloader(
            *train_data, y=train_y, batch_size=self.batch_size
        )
        val_dataloader = get_dataloader(*val_data, y=val_y, batch_size=self.batch_size)
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, val_data: Tuple):
        dataloader = get_dataloader(*val_data, y=None, batch_size=self.batch_size)
        preds = self.trainer.predict(self.model, dataloader)
        return torch.cat(preds, dim=0).cpu().numpy()  # type: ignore

    def log_model(self, model_name: str, signature: ModelSignature):
        mlflow_pytorch.log_model(self.model, name=model_name, signature=signature)

    def get_n_paramas(self) -> dict:
        """Reurns dict with number of trainable parameters"""
        dict = {}
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            dict[name] = params
            total_params += params
        return dict
