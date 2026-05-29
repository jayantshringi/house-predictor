from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Helper to load artifacts lazily or handle missing files
ARTIFACT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../models/production_artifacts.pkl'))

model = None
scaler = None

def load_artifacts():
    global model, scaler
    if model is None or scaler is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError(f"Model artifacts file not found at {ARTIFACT_PATH}. Please run training first.")
        artifacts = joblib.load(ARTIFACT_PATH)
        model = artifacts['model']
        scaler = artifacts['scaler']

@app.route('/health', methods=['GET'])
def health():
    try:
        load_artifacts()
        status = "healthy"
    except Exception as e:
        status = f"unhealthy: {str(e)}"
    return jsonify({"status": status}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_artifacts()
    except Exception as e:
        return jsonify({"error": "Model not ready", "details": str(e)}), 503

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Missing JSON request payload"}), 400
            
        # Explicit Schema Validation
        required = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'AgeOfHouse']
        for field in required:
            if field not in payload or not isinstance(payload[field], (int, float)):
                return jsonify({"error": f"Invalid or missing parameter: {field}"}), 400
        
        # Format payload input array
        raw_features = np.array([[
            payload['SquareFeet'],
            payload['Bedrooms'],
            payload['Bathrooms'],
            payload['AgeOfHouse']
        ]])
        
        # Apply the exact training scaler to incoming production traffic
        scaled_features = scaler.transform(raw_features)
        
        # Run inference
        prediction = model.predict(scaled_features)[0]
        
        return jsonify({
            "estimated_price": float(round(prediction, 2)),
            "currency": "USD"
        }), 200

    except Exception as e:
        return jsonify({"error": "Internal execution failure", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
