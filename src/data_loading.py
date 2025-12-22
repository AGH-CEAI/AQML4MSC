from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

TRAIN_VAL_IMAGES_PATH = "data/train_images.npy"
TRAIN_VAL_LABELS_PATH = "data/train_labels.npy"
TEST_IMAGES_PATH = "data/test_images.npy"
TEST_LABELS_PATH = "data/test_labels.npy"


def load_data(
    img_path: str = TRAIN_VAL_IMAGES_PATH, label_path: str = TRAIN_VAL_LABELS_PATH
) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(img_path), np.load(label_path)


def choose_digits(
    X: np.ndarray, y: np.ndarray, digits: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    indexes = np.isin(y, digits)
    X_bin, y_bin = X[indexes], y[indexes]
    # y_bin = np.where(y_bin == 0, -1, 1)
    return X_bin, y_bin
