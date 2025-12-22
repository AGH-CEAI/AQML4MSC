from typing import Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch

from src.training.base_training import BaseTraining
from src.utils import get_dataloader


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

    def predict(self, val_data: Tuple, val_y: np.ndarray):
        dataloader = get_dataloader(*val_data, y=val_y, batch_size=self.batch_size)
        preds = self.trainer.predict(self.model, dataloader)
        return torch.cat(preds, dim=0).cpu().numpy()  # type: ignore
