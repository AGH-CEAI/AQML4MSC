import os

import numpy as np
import numpy.typing as npt

from aqml4msc.datasets.base_dataset import BaseDataset
from aqml4msc.preprocessing.mnist import preprocess_pipeline


class MnistDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)
        self.x_raw: npt.NDArray[np.float64]
        self.y_raw: npt.NDArray[np.int_]
        self.x_clean: npt.NDArray[np.float64]
        self.y_clean: npt.NDArray[np.int_]
        self.x_top: npt.NDArray[np.float64]
        self.x_bottom: npt.NDArray[np.float64]

    def load_raw(
        self,
        img_path: str = os.environ["TRAIN_VAL_IMAGES_PATH"],
        label_path: str = os.environ["TRAIN_VAL_LABELS_PATH"],
    ):
        self.x_raw, self.y_raw = np.load(img_path), np.load(label_path)

    def clean_data(self):
        indexes = np.isin(self.y_raw, self.config["digits"])
        self.x_clean, self.y_clean = self.x_raw[indexes], self.y_raw[indexes]

    def preprocess(self):
        self.x_top, self.x_bottom = preprocess_pipeline(self.x_clean)

    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError
