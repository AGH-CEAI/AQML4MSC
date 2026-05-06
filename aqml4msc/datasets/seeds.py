import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from preprocessing.seeds import (
    add_indirect_features,
    drop_columns,
    normalize_data,
    pd_to_numpy_X_y,
    seperate_X_y,
)

from aqml4msc.datasets.base_dataset import BaseDataset


class SeedsTabDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_raw: pd.DataFrame
        self.x_clean: pd.DataFrame
        self.y_clean: pd.DataFrame
        self.train_x_df: pd.DataFrame
        self.train_y_df: pd.DataFrame
        self.val_x_df: pd.DataFrame
        self.val_y_df: pd.DataFrame

    def load_raw(
        self,
        tab_data_path: str = os.environ["SEEDS_TAB_DATA_LOCATION"],
    ):
        self.data_raw = pd.read_excel(tab_data_path)

    def clean_data(self):
        self.x_clean, self.y_clean = seperate_X_y(self.data_raw)
        self.x_clean = drop_columns(self.x_clean)
        self.x_clean = add_indirect_features(self.x_clean)
        self.y_clean = pd.Series(self.label_encoder.fit_transform(self.y_clean))  # type: ignore

    def preprocess(self):
        if self.train_x_df.empty or self.val_x_df.empty:
            raise ValueError(
                "Train and validation feature dataframes must not be empty."
            )
        self.train_x_df, self.val_x_df = normalize_data(self.train_x_df, self.val_x_df)
        train_x, train_y = pd_to_numpy_X_y(self.train_x_df, self.train_y_df)
        val_x, val_y = pd_to_numpy_X_y(self.val_x_df, self.val_y_df)
        self.train_data = (train_x,)
        self.train_labels = train_y
        self.val_data = (val_x,)
        self.val_labels = val_y

    def set_splits(self, train_idx, test_idx):
        self.train_x_df = self.x_clean.iloc[train_idx]
        self.train_y_df = self.y_clean.iloc[train_idx]

        self.val_x_df = self.x_clean.iloc[test_idx]
        self.val_y_df = self.y_clean.iloc[test_idx]

    def get_n_samples(self) -> int:
        if self.y_clean.empty:
            raise ValueError("No training labels available")
        return self.y_clean.shape[0]

    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        if self.y_clean.empty:
            raise ValueError("No training labels available")
        return self.y_clean.to_numpy()


class SeedsImageDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)

    def load_raw(
        self,
        img_path: str = os.environ["SEEDS_IMAGE_DATA_LOCATION"],
        label_path: str = os.environ["SEEDS_TAB_DATA_LOCATION"],
    ):
        raise NotImplementedError

    def clean_data(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError

    def get_n_samples(self) -> int:
        raise NotImplementedError

    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        raise NotImplementedError


class SeedsDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)

    def load_raw(
        self,
        img_path: str = os.environ["SEEDS_IMAGE_DATA_LOCATION"],
        label_path: str = os.environ["SEEDS_TAB_DATA_LOCATION"],
    ):
        raise NotImplementedError

    def clean_data(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError

    def get_n_samples(self) -> int:
        raise NotImplementedError

    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        raise NotImplementedError
