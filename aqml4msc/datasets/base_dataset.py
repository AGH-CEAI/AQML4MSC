from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, config: dict):
        self.config = config

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
