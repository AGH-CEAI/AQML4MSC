from typing import Iterator, Tuple

import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch import from_numpy
from torch.utils.data import DataLoader, Subset, TensorDataset


def get_dataloader(
    *X_sources: np.ndarray,
    y: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    shuffle: bool = False,
    batch_size: int = 32,
) -> DataLoader:
    if len(X_sources) == 0:
        raise ValueError("At least one X source must be provided")

    if y is not None:
        n_samples = y.shape[0]
        for i, X in enumerate(X_sources):
            if X.shape[0] != n_samples:
                raise ValueError(
                    f"X_sources[{i}] has {X.shape[0]} samples, expected {n_samples}"
                )

    tensors = [from_numpy(X).float() for X in X_sources]
    if y is not None:
        tensors.append(from_numpy(y))

    dataset = TensorDataset(*tensors)

    if indices is not None:
        dataset = Subset(dataset, indices.tolist())

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def check_cross_val_split_sizes(y: np.ndarray, n_splits=5, seed=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(y, y), 1):
        print(f"FOLD {fold}   train: {len(train_idx)}, test: {len(val_idx)}")


def get_stratified_cv_splits(
    y: np.ndarray,
    n_folds: int,
    start_idx: int = 1,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    kfold = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    yield from (
        (fold, train_idx, val_idx)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(y, y), start=start_idx)
    )


def encode_labels(y: np.ndarray) -> Tuple[LabelEncoder, np.ndarray]:
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y = label_encoder.transform(y)  # type: ignore
    return label_encoder, y
