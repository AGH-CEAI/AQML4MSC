from abc import ABC, abstractmethod
from typing import Tuple, Type

from datasets.base_dataset import BaseDataset
from mlflow.models import ModelSignature


class BaseTraining(ABC):
    def __init__(self, model_cls: Type, model_kwargs: dict):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = self.model_cls(**self.model_kwargs)

    @abstractmethod
    def fit(self, dataset: BaseDataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, val_data: Tuple):
        raise NotImplementedError

    @abstractmethod
    def log_model(self, model_name: str, signature: ModelSignature):
        raise NotImplementedError

    def reset_model(self):
        """Create a fresh model instance."""
        self.model = self.model_cls(**self.model_kwargs)
