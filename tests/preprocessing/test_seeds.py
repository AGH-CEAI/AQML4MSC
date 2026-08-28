import pytest
import pandas as pd
import numpy as np
import torch
from aqml4msc.preprocessing.seeds import (
    drop_columns,
    normalize_data,
    add_indirect_features,
    seperate_X_y,
    pd_to_numpy_X_y,
    numpy_to_tensor_X_y,
    COLS_TO_KEEP,
    COLS_TO_DROP,
    COL_LABEL,
    COL_FEATURES
)

@pytest.fixture
def sample_df():
    data = {
        "No.": [1, 2],
        "Id": ["id1", "id2"],
        "wheatvariety": [0, 1],
        "kernelarea": [10.0, 20.0],
        "kernelperimeter": [1.0, 2.0],
        "compactness": [0.5, 0.6],
        "kernellength": [5.0, 8.0],
        "kernelwidth": [2.0, 4.0],
        "asymmetry": [0.1, 0.2],
        "groovelength": [3.0, 6.0],
        "germarea": [2.0, 5.0],
        "germlength": [1.0, 3.0],
    }
    return pd.DataFrame(data)

def test_drop_columns(sample_df):
    df_dropped = drop_columns(sample_df)
    assert "No." not in df_dropped.columns
    assert "Id" not in df_dropped.columns
    assert "kernelarea" in df_dropped.columns

def test_drop_columns_empty():
    with pytest.raises(ValueError):
        drop_columns(pd.DataFrame())

def test_add_indirect_features(sample_df):
    df_added = add_indirect_features(sample_df)
    assert "germarea_kernelarea" in df_added.columns
    assert "germlength_kernellength" in df_added.columns
    assert "kernelwidth_kernellength" in df_added.columns
    
    # Check values for first row
    assert df_added["germarea_kernelarea"].iloc[0] == 2.0 / 10.0
    assert df_added["germlength_kernellength"].iloc[0] == 1.0 / 5.0

def test_seperate_X_y(sample_df):
    df_added = add_indirect_features(sample_df)
    X, y = seperate_X_y(df_added)
    
    assert y.name == COL_LABEL
    assert set(X.columns) == set(COL_FEATURES)

def test_normalize_data(sample_df):
    df_added = add_indirect_features(sample_df)
    X, _ = seperate_X_y(df_added)
    
    # Just split into two identical for testing
    X_train = X.copy()
    X_test = X.copy()
    
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # Mean should be approx 0 and std approx 1 for training data
    assert np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-7)
    # std in StandardScaler uses ddof=0, so it's exactly 1.0 for sample
    assert np.allclose(X_train_norm.std(axis=0, ddof=0), 1, atol=1e-7)

def test_pd_to_numpy_X_y(sample_df):
    df_added = add_indirect_features(sample_df)
    X, y = seperate_X_y(df_added)
    
    X_np, y_np = pd_to_numpy_X_y(X, y)
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
    assert X_np.shape == (2, len(COL_FEATURES))
    assert y_np.shape == (2,)

def test_numpy_to_tensor_X_y():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    
    X_tensor, y_tensor = numpy_to_tensor_X_y(X, y)
    assert isinstance(X_tensor, torch.Tensor)
    assert isinstance(y_tensor, torch.Tensor)
    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.long
