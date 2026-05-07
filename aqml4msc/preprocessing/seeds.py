from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torchvision import transforms

COLS_TO_DROP = ["No.", "Id"]
COL_ID = ["Id"]
COL_LABEL = "wheatvariety"
COLS_TO_KEEP = [
    "kernelarea",
    "kernelperimeter",
    "compactness",
    "kernellength",
    "kernelwidth",
    "asymmetry",
    "groovelength",
    "germarea",
    "germlength",
]
COLS_TO_ADD = [
    "germarea_kernelarea",
    "germlength_kernellength",
    "kernelwidth_kernellength",
]
COL_FEATURES = COLS_TO_ADD + COLS_TO_KEEP


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame is empty")
    return df.drop(columns=COLS_TO_DROP)


def normalize_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    df_train.loc[:,COL_FEATURES] = scaler.fit_transform(df_train.loc[:,COL_FEATURES])
    df_test.loc[:,COL_FEATURES] = scaler.transform(df_test.loc[:,COL_FEATURES])
    return df_train, df_test


def add_indirect_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new["germarea_kernelarea"] = df["germarea"] / df["kernelarea"]
    df_new["germlength_kernellength"] = df["germlength"] / df["kernellength"]
    df_new["kernelwidth_kernellength"] = df["kernelwidth"] / df["kernellength"]
    return df_new


def seperate_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[COL_FEATURES]
    y = df[COL_LABEL]
    return X, y


def pd_to_numpy_X_y(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    return X_np, y_np


def numpy_to_tensor_X_y(
    X: np.ndarray, y: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def get_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_tensor, y_tensor = numpy_to_tensor_X_y(X, y)
    return TensorDataset(X_tensor, y_tensor)


### IMAGE PREPROCESSING ########################################


def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def center_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = img.size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return img.crop((left, top, right, bottom))


def pad_image(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    org_width, org_height = img.size
    if target_height < org_height or target_width < org_width:
        raise ValueError(
            f"Target dimensions ({target_width}, {target_height}) can't be smaller than original dimensions ({org_width}, {org_height})"
        )
    new_image = Image.new(img.mode, (target_width, target_height))
    pad_size_left = (target_width - org_width) // 2
    pad_size_top = (target_height - org_height) // 2
    new_image.paste(img, (pad_size_left, pad_size_top))
    return new_image

def get_max_sizes(images: List[Image.Image]) -> Tuple[int, int]:
    w, h = [], []
    for img in images:
        w.append(img.width)
        h.append(img.height)
    return max(w), max(h)


### JOINED PREPROCESSING ########################################


def sort_dataframe(df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    ids = [iid[:-4] for iid in ids]
    return df.set_index(COL_ID).loc[ids].reset_index()


# def preprocess_all(
#     df: pd.DataFrame, imgs: List[Image.Image], ids: List[str]
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     df = sort_dataframe(df, ids)
#     imgs_tensor = preprocess_images(imgs)
#     df_x, df_y = clean_data(df)

#     numpy_x, numpy_y = pd_to_numpy_X_y(df_x, df_y)
#     tensor_x, tensor_y = numpy_to_tensor_X_y(numpy_x, numpy_y)

#     return tensor_x, imgs_tensor, tensor_y


def tensors_to_dataset(
    features: torch.Tensor, imgs: torch.Tensor, labels: torch.Tensor
) -> TensorDataset:
    return TensorDataset(features, imgs, labels)


def join_multisource(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.cat((x1, x2), dim=1)
