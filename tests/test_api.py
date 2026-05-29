import os
import pytest
import sys

# Ensure src directory is in Python path for test import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import app
from train import run_training_pipeline
from generate_data import generate_synthetic_housing_data

@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    # If housing data is not generated, generate it
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/housing.csv'))
    if not os.path.exists(data_path):
        generate_synthetic_housing_data(data_path)
    
    # If model is not trained, train it
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/production_artifacts.pkl'))
    if not os.path.exists(model_path):
        run_training_pipeline()

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get('/health')
    assert res.status_code == 200
    assert res.get_json() == {"status": "healthy"}

def test_successful_prediction(client):
    payload = {"SquareFeet": 2000, "Bedrooms": 3, "Bathrooms": 2, "AgeOfHouse": 5}
    res = client.post('/predict', json=payload)
    assert res.status_code == 200
    assert "estimated_price" in res.get_json()

def test_malformed_payload(client):
    invalid_payload = {"SquareFeet": "Two Thousand", "Bedrooms": 3}
    res = client.post('/predict', json=invalid_payload)
    assert res.status_code == 400
