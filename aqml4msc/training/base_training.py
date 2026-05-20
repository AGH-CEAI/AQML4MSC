from abc import ABC, abstractmethod

from datasets.base_dataset import BaseDataset
from mlflow.models import ModelSignature

from aqml4msc.models.base_mlp_model import BaseMLPModel


class BaseTraining(ABC):
    @abstractmethod
    def fit(self, model: BaseMLPModel, dataset: BaseDataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: BaseMLPModel, dataset: BaseDataset):
        raise NotImplementedError

    @abstractmethod
    def log_model(
        self, model: BaseMLPModel, model_name: str, signature: ModelSignature
    ):
        raise NotImplementedError
