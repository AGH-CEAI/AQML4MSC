from abc import ABC, abstractmethod


class BaseDataset(ABC):
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
