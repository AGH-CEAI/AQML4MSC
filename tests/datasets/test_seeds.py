import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch

from aqml4msc.datasets.seeds import SeedsTabDataset, SeedsImageDataset, SeedsDataset
from PIL import Image

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
def test_prepare_data(mock_read_excel, seeds_dataset, mock_excel_data):
    mock_read_excel.return_value = mock_excel_data
    seeds_dataset.load_raw()
    
    seeds_dataset.prepare_data()
    
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
    seeds_dataset.prepare_data()
    
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
    seeds_dataset.prepare_data()
    
    labels = seeds_dataset.get_encoded_labels()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 4

@pytest.fixture
def dummy_image_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # create some dummy images
    img1 = Image.new('RGB', (10, 10), color='red')
    img1.save(img_dir / "canadian10_11.jpg")
    img2 = Image.new('RGB', (12, 12), color='blue')
    img2.save(img_dir / "kama160_14.jpg")
    img3 = Image.new('RGB', (8, 10), color='green')
    img3.save(img_dir / "rosa12_24.jpg")
    img4 = Image.new('RGB', (10, 8), color='yellow')
    img4.save(img_dir / "canadian10_12.jpg")
    return str(img_dir)

@pytest.fixture
def seeds_image_dataset():
    config = {"batch_size": 2, "pca_components": 2}
    return SeedsImageDataset(config)

def test_image_load_raw(seeds_image_dataset, dummy_image_dir):
    seeds_image_dataset.load_raw(img_path=dummy_image_dir)
    assert len(seeds_image_dataset.images_raw) == 4
    assert len(seeds_image_dataset.labels_raw) == 4
    assert set(seeds_image_dataset.labels_raw) == {
        "canadian10_11.jpg", "kama160_14.jpg", "rosa12_24.jpg", "canadian10_12.jpg"
    }

def test_image_prepare_data(seeds_image_dataset, dummy_image_dir):
    seeds_image_dataset.load_raw(img_path=dummy_image_dir)
    seeds_image_dataset.prepare_data()
    assert isinstance(seeds_image_dataset.images_prepared, np.ndarray)
    assert seeds_image_dataset.images_prepared.shape[0] == 4
    assert seeds_image_dataset.images_prepared.shape[1:] == (12, 12)
    assert len(seeds_image_dataset.labels_encoded) == 4

def test_image_set_splits_and_preprocess(seeds_image_dataset, dummy_image_dir):
    seeds_image_dataset.load_raw(img_path=dummy_image_dir)
    seeds_image_dataset.prepare_data()
    
    train_idx = [0, 1, 2]
    test_idx = [3]
    
    seeds_image_dataset.set_splits(train_idx, test_idx)
    assert len(seeds_image_dataset.train_imgs) == 3
    assert len(seeds_image_dataset.val_imgs) == 1
    
    seeds_image_dataset.preprocess()
    
    assert isinstance(seeds_image_dataset.train_data, tuple)
    assert seeds_image_dataset.train_data[0].shape == (3, 2)
    assert seeds_image_dataset.val_data[0].shape == (1, 2)

def test_image_get_encoded_labels(seeds_image_dataset, dummy_image_dir):
    seeds_image_dataset.load_raw(img_path=dummy_image_dir)
    seeds_image_dataset.prepare_data()
    labels = seeds_image_dataset.get_encoded_labels()
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 4

def test_image_get_n_samples(seeds_image_dataset, dummy_image_dir):
    seeds_image_dataset.load_raw(img_path=dummy_image_dir)
    seeds_image_dataset.prepare_data()
    assert seeds_image_dataset.get_n_samples() == 4
    
    # Also test empty scenario
    empty_dataset = SeedsImageDataset({"batch_size": 2, "pca_components": 2})
    # If prepare_data is not called, images_prepared is not set, so it raises AttributeError.
    with pytest.raises(AttributeError):
        empty_dataset.get_n_samples()

@pytest.fixture
def seeds_dataset_full():
    config = {"batch_size": 2, "pca_components": 2}
    return SeedsDataset(config)

@patch("aqml4msc.datasets.seeds.SeedsTabDataset.load_raw")
@patch("aqml4msc.datasets.seeds.SeedsImageDataset.load_raw")
def test_seedsdataset_load_raw(mock_img_load, mock_tab_load, seeds_dataset_full):
    # The default paths are usually mocked or set via os.environ in real code, 
    # but here we can just pass dummy paths.
    seeds_dataset_full.load_raw("dummy_img_path", "dummy_tab_path")
    mock_img_load.assert_called_once_with("dummy_img_path")
    mock_tab_load.assert_called_once_with("dummy_tab_path")

@patch("aqml4msc.datasets.seeds.SeedsImageDataset.get_raw_labels")
@patch("aqml4msc.datasets.seeds.SeedsTabDataset.sort_by_label")
@patch("aqml4msc.datasets.seeds.SeedsImageDataset.prepare_data")
@patch("aqml4msc.datasets.seeds.SeedsTabDataset.prepare_data")
def test_seedsdataset_prepare_data(mock_tab_prep, mock_img_prep, mock_tab_sort, mock_img_get_raw, seeds_dataset_full):
    mock_img_get_raw.return_value = ["canadian10_11.jpg", "kama160_14.jpg"]
    seeds_dataset_full.prepare_data()
    mock_img_get_raw.assert_called_once()
    mock_tab_sort.assert_called_once_with(["canadian10_11.jpg", "kama160_14.jpg"])
    mock_img_prep.assert_called_once()
    mock_tab_prep.assert_called_once()

@patch("aqml4msc.datasets.seeds.SeedsImageDataset.set_splits")
@patch("aqml4msc.datasets.seeds.SeedsTabDataset.set_splits")
def test_seedsdataset_set_splits(mock_tab_splits, mock_img_splits, seeds_dataset_full):
    train_idx, test_idx = [0, 1], [2, 3]
    seeds_dataset_full.set_splits(train_idx, test_idx)
    mock_img_splits.assert_called_once_with(train_idx, test_idx)
    mock_tab_splits.assert_called_once_with(train_idx, test_idx)

def test_seedsdataset_preprocess_and_getters(seeds_dataset_full):
    # Instead of patching everything, let's inject mocks for the datasets
    from unittest.mock import MagicMock
    
    seeds_dataset_full.image_dataset = MagicMock()
    seeds_dataset_full.tab_dataset = MagicMock()
    
    seeds_dataset_full.tab_dataset.train_data = (np.array([[1]]), np.array([[2]]))
    seeds_dataset_full.image_dataset.train_data = (np.array([[3]]),)
    seeds_dataset_full.tab_dataset.val_data = (np.array([[4]]), np.array([[5]]))
    seeds_dataset_full.image_dataset.val_data = (np.array([[6]]),)
    
    seeds_dataset_full.tab_dataset.train_labels = np.array([0])
    seeds_dataset_full.tab_dataset.val_labels = np.array([1])
    
    seeds_dataset_full.preprocess()
    
    seeds_dataset_full.image_dataset.preprocess.assert_called_once()
    seeds_dataset_full.tab_dataset.preprocess.assert_called_once()
    
    # train_data should be tab_dataset.train_data + image_dataset.train_data
    assert len(seeds_dataset_full.train_data) == 3
    assert len(seeds_dataset_full.val_data) == 3
    assert np.array_equal(seeds_dataset_full.train_labels, np.array([0]))
    assert np.array_equal(seeds_dataset_full.val_labels, np.array([1]))

def test_seedsdataset_getters(seeds_dataset_full):
    from unittest.mock import MagicMock
    seeds_dataset_full.image_dataset = MagicMock()
    seeds_dataset_full.tab_dataset = MagicMock()
    
    # test get_n_samples match
    seeds_dataset_full.image_dataset.get_n_samples.return_value = 10
    seeds_dataset_full.tab_dataset.get_n_samples.return_value = 10
    assert seeds_dataset_full.get_n_samples() == 10
    
    # test get_n_samples mismatch
    seeds_dataset_full.tab_dataset.get_n_samples.return_value = 5
    with pytest.raises(ValueError, match="Number of samples in image and tab datasets do not match"):
        seeds_dataset_full.get_n_samples()
        
    # test get_encoded_labels match
    labels = np.array([1, 2, 3])
    seeds_dataset_full.image_dataset.get_encoded_labels.return_value = labels
    seeds_dataset_full.tab_dataset.get_encoded_labels.return_value = labels
    assert np.array_equal(seeds_dataset_full.get_encoded_labels(), labels)
    
    # test get_encoded_labels mismatch
    seeds_dataset_full.tab_dataset.get_encoded_labels.return_value = np.array([1, 2, 4])
    with pytest.raises(ValueError, match="Labels in image and tab datasets do not match"):
        seeds_dataset_full.get_encoded_labels()
