import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from preprocessing.seeds import (
    add_indirect_features,
    drop_columns,
    normalize_data,
    pd_to_numpy_X_y,
    seperate_X_y,
    to_grayscale,
    pad_image,
    get_max_sizes,
)
from aqml4msc.datasets.base_dataset import BaseDataset


class SeedsTabDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_raw: pd.DataFrame
        self.x_clean: pd.DataFrame
        self.y_clean: pd.Series
        self.train_x_df: pd.DataFrame
        self.train_y_df: pd.Series
        self.val_x_df: pd.DataFrame
        self.val_y_df: pd.Series

    def load_raw(
        self,
        tab_data_path: str = os.environ["SEEDS_TAB_DATA_LOCATION"],
    ):
        self.data_raw = pd.read_excel(tab_data_path)

    def prepare_data(self):
        df_dropped = drop_columns(self.data_raw)
        df_added = add_indirect_features(df_dropped)
        self.x_clean, self.y_clean = seperate_X_y(df_added)
        self.y_clean = pd.Series(self.label_encoder.fit_transform(self.y_clean))  # type: ignore

    def set_splits(self, train_idx, test_idx):
        self.train_x_df = self.x_clean.iloc[train_idx]
        self.train_y_df = self.y_clean.iloc[train_idx]
        self.val_x_df = self.x_clean.iloc[test_idx]
        self.val_y_df = self.y_clean.iloc[test_idx]

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
        self.images_raw: list[Image.Image]
        self.labels_raw: list[str]
        self.images_prepared: npt.NDArray[np.float64]
        self.labels_extracted: list[str]
        self.labels_encoded: npt.NDArray[np.int_]
        self.train_imgs: npt.NDArray[np.float64]
        self.val_imgs: npt.NDArray[np.float64]

    def load_raw(
        self,
        img_path: str = os.environ["SEEDS_IMAGE_DATA_LOCATION"],
    ):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path not found: {img_path}")
        img_filename_list = [fname for fname in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, fname))]
        if not img_filename_list:
            raise ValueError(f"No images found in {img_path}")
        images = []
        for filename in img_filename_list:
            with Image.open(os.path.join(img_path, filename)) as img:
                images.append(img.copy())
        self.images_raw = images
        # Extract class names from filenames (e.g. "canadian10_11.jpg" -> "canadian")
        self.labels_extracted = ["".join(filter(str.isalpha, fname.split('.')[0])) for fname in img_filename_list]


    def prepare_data(self):
        """
        Pad images to max size to keep consistent shapes and convert to grayscale
        """
        if len(self.images_raw) == 0:
            raise ValueError("No images found")
        max_width, max_height = get_max_sizes(self.images_raw)
        prepared_images = []
        for img in self.images_raw:
            img = to_grayscale(img)
            img = pad_image(img, max_width, max_height)
            prepared_images.append(np.array(img))
        self.images_prepared = np.array(prepared_images)
        self.labels_encoded = self.label_encoder.fit_transform(self.labels_extracted)  # type: ignore

    def set_splits(self, train_idx, test_idx):
        if len(self.images_prepared) == 0:
            raise ValueError("No images found")
        self.train_imgs = self.images_prepared[train_idx]
        self.train_labels = self.labels_encoded[train_idx]
        self.val_imgs = self.images_prepared[test_idx]
        self.val_labels = self.labels_encoded[test_idx]

    def preprocess(self):
        images_flat: npt.NDArray[np.float64] = self.train_imgs.reshape(len(self.train_imgs), -1)
        images_flat_val: npt.NDArray[np.float64] = self.val_imgs.reshape(len(self.val_imgs), -1)
        pca = PCA(n_components=self.config["pca_components"])
        images_reduced: npt.NDArray[np.float64] = pca.fit_transform(images_flat)
        images_reduced_val: npt.NDArray[np.float64] = pca.transform(images_flat_val)
        self.train_data = (images_reduced,)
        self.val_data = (images_reduced_val,)
         

    def get_n_samples(self) -> int:
        raise NotImplementedError

    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        if len(self.labels_encoded) == 0:
            raise ValueError("No training labels available")
        return self.labels_encoded


class SeedsDataset(BaseDataset):
    def __init__(self, config: dict):
        super().__init__(config)

    def load_raw(
        self,
        img_path: str = os.environ["SEEDS_IMAGE_DATA_LOCATION"],
        label_path: str = os.environ["SEEDS_TAB_DATA_LOCATION"],
    ):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError


    def set_splits(self, train_idx, test_idx):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def get_n_samples(self) -> int:
        raise NotImplementedError

    def get_encoded_labels(self) -> npt.NDArray[np.int_]:
        raise NotImplementedError
