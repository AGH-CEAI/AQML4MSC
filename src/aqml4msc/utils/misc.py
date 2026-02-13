import random
from math import prod
from typing import Iterator, Tuple

import numpy as np
import pennylane as qml
import torch
from numpy.typing import NDArray
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


def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def probe_inputs_and_weight_shapes(
    circuit: qml.QNode,
) -> dict[str, tuple[int, ...]]:
    """Brute-force, very naive and with lots of assumptions. But that's how :mod:`pennylane` wants to play.

    We assume the following:
    1) there are only 2 inputs to the QNode: inputs (however named) and weights (however named)
    2) they can be given flat
    3) the circuit uses all the input values before using any of the weights (embedding on all wires does the trick)
    4) the bad-input ValueError happen before bad-weights IndexError

    :return: _description_
    :rtype: _type_
    """
    shapes = {}
    # input_shape = PennylaneToQasm3._probe_input_shape(circuit)
    try:
        qml.specs(circuit)(np.zeros(10000), [[]])  # Must be high.
    except ValueError as e:
        if "Input state must be of length " in str(e):
            # e reads:
            #
            # Input state must be of length X or smaller to be padded; got length Y.
            shapes["inputs"] = (int(str(e).split(" ")[6]),)
        if "Features must be of length" in str(e):
            # e reads:
            #
            # Features must be of length X or less; got length Y.
            shapes["inputs"] = (int(str(e).split(" ")[5]),)

    n_weights: int = 0
    weights: NDArray[int] = np.zeros(n_weights)  # type: ignore
    while True:
        try:
            # print(f"Trying: {weights}")
            qml.specs(circuit)(np.zeros(shapes["inputs"]), weights)
            shapes["weights"] = (n_weights,)
            # print(shapes)
            return shapes
        except ValueError as e:
            if "into shape" in str(e):  # This solves `aqmlator`-models problems.
                n_weights += prod(
                    [int(s) for s in str(e).split(" ")[-1][1:-1].split(",")]
                )
                weights = np.zeros(n_weights)
                continue
            if "must have last dimension of" in str(e):
                # e reads:
                #
                # Weights tensor must have last dimension of length X; got Y
                #
                # We extract X again.
                # We assume 1 layer per layer call.
                shapes["weights"] = (1, int(str(e).split(" ")[8][:-1]))
                return shapes
            if "must be 2-dimensional or 3-dimensional" in str(e):
                # Assume 1 layer per layer call.
                weights = np.zeros((1, n_weights))
