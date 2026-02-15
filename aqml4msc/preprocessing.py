from typing import Tuple

import numpy as np
from skimage.measure import block_reduce


def cut_in_half(imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mid = imgs.shape[1] // 2
    imgs_top = imgs[:, :mid, :]
    img_bottom = imgs[:, mid:, :]
    return imgs_top, img_bottom


def binarize_images(imgs: np.ndarray) -> np.ndarray:
    return (imgs > 127).astype(np.uint8)


def max_pool(imgs: np.ndarray, block_size=2) -> np.ndarray:
    pooled = [block_reduce(img, block_size, func=np.max) for img in imgs]
    return np.stack(pooled)


def vertical_projection_mean(imgs):
    return np.mean(imgs, axis=1)


########################################


def preprocess_pipeline(imgs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # (2115, 26, 28) -> (2115, 14, 28), (2115, 14, 28)
    imgs_top, img_bottom = cut_in_half(imgs)

    imgs_top_bin = binarize_images(imgs_top)
    imgs_bottom_bin = binarize_images(img_bottom)

    # (2115, 14, 28) -> (2115, 7, 14)
    imgs_top_reduced = max_pool(imgs_top_bin)
    imgs_bottom_reduced = max_pool(imgs_bottom_bin)

    # (2115, 7, 14) -> (2115, 14)
    imgs_top_features = vertical_projection_mean(imgs_top_reduced)
    imgs_bottom_features = vertical_projection_mean(imgs_bottom_reduced)

    return imgs_top_features, imgs_bottom_features
