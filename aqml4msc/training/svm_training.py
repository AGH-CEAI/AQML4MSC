from typing import Tuple, Type

import mlflow.sklearn as mlflow_sklearn
import numpy as np
from mlflow.models import ModelSignature

from aqml4msc.training.base_training import BaseTraining


class SVMTraining(BaseTraining):
    def __init__(self, model_cls: Type, model_kwargs: dict):
        super().__init__(model_cls=model_cls, model_kwargs=model_kwargs)

    def fit(
        self,
        train_data: Tuple,
        train_y: np.ndarray,
        val_data: Tuple | None = None,
        val_y: np.ndarray | None = None,
    ):
        self.model.fit(train_data, train_y)

    def predict(self, val_data: Tuple):
        data = val_data
        return self.model.predict(data)

    def log_model(self, model_name: str, signature: ModelSignature):
        mlflow_sklearn.log_model(self.model, name=model_name, signature=signature)
