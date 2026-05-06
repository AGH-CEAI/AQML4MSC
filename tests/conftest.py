import os
import pytest

# Set default dummy environment variables for tests before modules are imported
os.environ["SEEDS_TAB_DATA_LOCATION"] = "dummy_tab_data.xlsx"
os.environ["SEEDS_IMAGE_DATA_LOCATION"] = "dummy_image_data/"
