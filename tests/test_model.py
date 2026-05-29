import os
import sys
import pytest
import pandas as pd

# Ensure src directory is in Python path for test import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_prep import load_and_validate_data
from train import run_training_pipeline
from generate_data import generate_synthetic_housing_data

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/housing.csv'))
    # Always regenerate fresh test data to make sure outliers are consistently placed
    generate_synthetic_housing_data(data_path, n_samples=100)

def test_outlier_removal():
    # The generate_synthetic_housing_data function injects 5 outliers in the first 5 records.
    # We load raw data first:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/housing.csv'))
    raw_df = pd.read_csv(data_path)
    
    # Preprocess
    clean_df = load_and_validate_data(data_path)
    
    # The clean df should be smaller than raw df because outliers were filtered
    assert len(clean_df) < len(raw_df), "Outliers were not successfully removed"
    # Specifically, the extreme outliers injected at the top should be gone
    assert len(raw_df) - len(clean_df) >= 5, "Fewer outliers removed than injected"

def test_data_schema_validation():
    # Test that error is raised if columns are missing
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/housing.csv'))
    temp_df = pd.read_csv(data_path).drop(columns=['Price'])
    
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/temp_invalid.csv'))
    temp_df.to_csv(temp_path, index=False)
    
    with pytest.raises(ValueError, match="Missing mandatory column"):
        load_and_validate_data(temp_path)
        
    if os.path.exists(temp_path):
        os.remove(temp_path)
