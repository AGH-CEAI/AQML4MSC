import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch

from aqml4msc.datasets.seeds import SeedsTabDataset

@pytest.fixture
def mock_excel_data():
    data = {
        "No.": [1, 2, 3, 4],
        "Id": ["id1", "id2", "id3", "id4"],
        "wheatvariety": ["kama", "rosa", "canadian", "kama"],
        "kernelarea": [10.0, 20.0, 30.0, 40.0],
        "kernelperimeter": [1.0, 2.0, 3.0, 4.0],
        "compactness": [0.5, 0.6, 0.7, 0.8],
        "kernellength": [5.0, 10.0, 15.0, 20.0],
        "kernelwidth": [2.0, 4.0, 6.0, 8.0],
        "asymmetry": [0.1, 0.2, 0.3, 0.4],
        "groovelength": [3.0, 6.0, 9.0, 12.0],
        "germarea": [2.0, 4.0, 6.0, 8.0],
        "germlength": [1.0, 2.0, 3.0, 4.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def seeds_dataset():
    config = {"batch_size": 2}
    return SeedsTabDataset(config)

@patch("aqml4msc.datasets.seeds.pd.read_excel")
def test_load_raw(mock_read_excel, seeds_dataset, mock_excel_data):
    mock_read_excel.return_value = mock_excel_data
    
    seeds_dataset.load_raw()
    
    mock_read_excel.assert_called_once_with("dummy_tab_data.xlsx")
    assert seeds_dataset.data_raw.equals(mock_excel_data)

@patch("aqml4msc.datasets.seeds.pd.read_excel")
def test_clean_data(mock_read_excel, seeds_dataset, mock_excel_data):
    mock_read_excel.return_value = mock_excel_data
    seeds_dataset.load_raw()
    
    seeds_dataset.clean_data()
    
    assert seeds_dataset.x_clean is not None
    assert seeds_dataset.y_clean is not None
    # No. and Id should be dropped
    assert "No." not in seeds_dataset.x_clean.columns
    # indirect features should be added
    assert "germarea_kernelarea" in seeds_dataset.x_clean.columns
    # labels should be encoded (0, 1, 2 depending on unique values)
    assert set(seeds_dataset.y_clean.unique()).issubset({0, 1, 2})

@patch("aqml4msc.datasets.seeds.pd.read_excel")
def test_set_splits_and_preprocess(mock_read_excel, seeds_dataset, mock_excel_data):
    mock_read_excel.return_value = mock_excel_data
    seeds_dataset.load_raw()
    seeds_dataset.clean_data()
    
    # Train idx 0, 1; Val idx 2, 3
    train_idx = [0, 1]
    test_idx = [2, 3]
    
    seeds_dataset.set_splits(train_idx, test_idx)
    
    assert len(seeds_dataset.train_x_df) == 2
    assert len(seeds_dataset.val_x_df) == 2
    
    seeds_dataset.preprocess()
    
    # check that train_data and val_data are set properly
    assert len(seeds_dataset.train_data) == 1  # tuple with 1 element
    assert isinstance(seeds_dataset.train_data[0], np.ndarray)
    assert isinstance(seeds_dataset.train_labels, np.ndarray)
    
    # check shape
    assert seeds_dataset.train_data[0].shape[0] == 2
    assert seeds_dataset.train_labels.shape[0] == 2

def test_get_n_samples_empty(seeds_dataset):
    # Before clean_data is called, it should raise ValueError
    seeds_dataset.y_clean = pd.Series(dtype=int)
    with pytest.raises(ValueError):
        seeds_dataset.get_n_samples()

@patch("aqml4msc.datasets.seeds.pd.read_excel")
def test_get_encoded_labels(mock_read_excel, seeds_dataset, mock_excel_data):
    mock_read_excel.return_value = mock_excel_data
    seeds_dataset.load_raw()
    seeds_dataset.clean_data()
    
    labels = seeds_dataset.get_encoded_labels()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 4
