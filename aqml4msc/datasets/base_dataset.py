from abc import ABC, abstractmethod

from sklearn.preprocessing import LabelEncoder


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.label_encoder = LabelEncoder()

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
    def decode_labels(self, y: list[int]) -> list[int]:
        raise NotImplementedError
