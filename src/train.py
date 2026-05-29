import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Ensure src can import modules relatively regardless of current working directory
try:
    from src.data_prep import load_and_validate_data
except ImportError:
    from data_prep import load_and_validate_data

def run_training_pipeline():
    # 1. Load validated data
    # Look for data/housing.csv. If we are running from src/, look one level up.
    data_path = 'data/housing.csv'
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '../data/housing.csv')
        
    df = load_and_validate_data(data_path)
    
    X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'AgeOfHouse']]
    y = df['Price']
    
    # 2. Split with fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scale numerical variables (prevents SquareFeet dominating coefficients)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate Performance
    predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"--- Training Complete --- \n R2 Score: {r2:.4f} \n RMSE: ${rmse:,.2f}")
    
    # 6. Save Model and Scaler together as a dictionary artifact
    # Determine the model directory path relative to project root
    model_dir = 'models'
    if not os.path.exists(model_dir) and 'src' in os.getcwd():
        model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'features': list(X.columns)
    }
    artifact_path = os.path.join(model_dir, 'production_artifacts.pkl')
    joblib.dump(artifacts, artifact_path)
    print(f"Artifacts safely compiled inside {artifact_path}.")

if __name__ == "__main__":
    run_training_pipeline()
