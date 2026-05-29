# House Price Predictor - Production Pipeline

This repository contains a production-ready end-to-end Machine Learning pipeline and microservice for predicting house prices based on physical attributes (Square Feet, Bedrooms, Bathrooms, and Age of House).

## Features
- **Statistical Schema Validation**: Automatic check for mandatory columns, missing values, and data types.
- **Outlier Filtering**: Trimming data using the Interquartile Range (IQR) method.
- **Reproducible ML Pipeline**: Automated training with fixed random seeds and scaling transforms.
- **Exploratory Data Analysis**: Jupyter Notebook (`eda.ipynb`) verifying Multiple Linear Regression assumptions (Linearity, Multicollinearity with VIF, and Homoscedasticity).
- **Flask API Microservice**: Input payload validation, features transformation, and real-time inference.
- **Comprehensive Testing**: Pytest-driven test suites covering both the ML model pipeline and API endpoints.
- **Production Containerization**: Dockerfile configuration powered by Gunicorn.

---

## Directory Structure
```
house-predictor/
├── data/
│   └── housing.csv                   # Dataset (auto-generated or custom)
├── notebooks/
│   └── eda.ipynb                    # Assumptions verification notebook
├── src/
│   ├── __init__.py
│   ├── app.py                       # Flask Microservice
│   ├── data_prep.py                 # Data validation & outlier removal
│   ├── generate_data.py             # Synthetic data generator helper
│   ├── train.py                     # ML pipeline training & export script
│   └── utils.py                     # Utility helpers (logging)
├── tests/
│   ├── __init__.py
│   ├── test_api.py                  # Flask API test suite
│   └── test_model.py                # ML pipeline test suite
├── Dockerfile                       # Multi-stage production container
├── requirements.txt                 # Complete project dependencies
└── README.md                        # Documentation
```

---

## Setup Instructions

### 1. Initialize Virtual Environment
Create and activate the virtual environment to isolate project dependencies:

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
To generate synthetic data (includes outliers) and train the multiple linear regression model:
```bash
python src/generate_data.py
python src/train.py
```
This produces `data/housing.csv` and serializes the model artifacts to `models/production_artifacts.pkl`.

---

## Assumptions Checking (EDA)
Start the Jupyter Notebook server:
```bash
jupyter notebook notebooks/eda.ipynb
```
Follow the steps in the notebook to view Linearity, VIF calculations (multicollinearity), and Homoscedasticity checks.

---

## Running the API Microservice

### Locally (Development Server)
Run the development Flask server:
```bash
python src/app.py
```
The service will start on `http://localhost:5000`.

### Local Inference Request (cURL)
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"SquareFeet\": 1850, \"Bedrooms\": 3, \"Bathrooms\": 2, \"AgeOfHouse\": 12}"
```

---

## Run Automated Tests
Execute the pytest suite covering the training pipeline and endpoints:
```bash
pytest tests/
```

---

## Docker Containerization

### Build the Image
```bash
docker build -t house-predictor .
```

### Run the Container
```bash
docker run -p 5000:5000 house-predictor
```
The service will run via Gunicorn bound to `0.0.0.0:5000` with 4 workers.
