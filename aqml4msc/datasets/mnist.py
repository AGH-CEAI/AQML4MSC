from aqml4msc.datasets.base_dataset import BaseDataset


class MnistDataset(BaseDataset):
    def load_raw(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError
