from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from utils import get_dataloader


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.train_data: tuple
        self.train_labels: npt.NDArray[np.int_]
        self.val_data: tuple
        self.val_labels: npt.NDArray[np.int_]

    @abstractmethod
    def load_raw(self):
        raise NotImplementedError

    @abstractmethod
    def clean_data(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError

    @abstractmethod
    def get_n_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        raise NotImplementedError

    def decode_labels(self, y: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.label_encoder.inverse_transform(y)  # type: ignore

    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        return get_dataloader(
            *self.train_data, y=self.train_labels, batch_size=batch_size
        )

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        return get_dataloader(*self.val_data, y=self.val_labels, batch_size=batch_size)
