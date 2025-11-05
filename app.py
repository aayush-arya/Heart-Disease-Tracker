import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import os 
from io import StringIO

# --- Configuration ---
MODEL_FILENAME = 'final_heart_predictor.joblib'
EXPECTED_FEATURES = 13 # Updated for Cleveland dataset
app = Flask(__name__)
CORS(app) 

# --- Global Model Loading ---
final_pipeline = None
try:
    # Load the entire saved pipeline
    final_pipeline = joblib.load(MODEL_FILENAME)
    print("Model pipeline successfully loaded from final_heart_predictor.joblib")
except Exception as e:
    # This error occurs if setup.sh hasn't run yet on the server
    print(f"CRITICAL ERROR: Model file '{MODEL_FILENAME}' not found. Setup script needs to run.")
    print(f"DETAILS: {e}")

# --- Serve the index.html file directly on the root path ---
@app.route('/', methods=['GET'])
def index():
    try:
        # Read the index.html content for serving
        with open('index.html', 'r') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Error: index.html not found. Please ensure it is in the same directory as app.py.", 500

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if final_pipeline is None:
        return jsonify({'error': 'Model not loaded on server. Server setup failed.'}), 500

    try:
        data = request.get_json(silent=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format. Expected {"features": [...]}'}), 400

        features = data['features']

        # Ensure we have exactly 13 features
        if len(features) != EXPECTED_FEATURES:
            return jsonify({
                'error': f'Expected {EXPECTED_FEATURES} features, but received {len(features)}. Check index.html.'
            }), 400

        # Convert list of features into a pandas DataFrame (required by the pipeline)
        feature_df = pd.DataFrame([features])
        
        # Make prediction and get probabilities
        prediction = final_pipeline.predict(feature_df)[0]
        prediction_proba = final_pipeline.predict_proba(feature_df)[0]
        
        # In Cleveland data, 1 is presence of disease
        proba_no_disease = prediction_proba[0]
        proba_disease = prediction_proba[1]

        result_text = "High likelihood of heart disease (1)" if prediction == 1 else "Low likelihood of no disease (0)"
        
        return jsonify({
            'prediction': int(prediction),
            'result_text': result_text,
            'probability_no_disease': proba_no_disease,
            'probability_disease': proba_disease,
            'model_used': final_pipeline.steps[-1][0] 
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

# --- Start the Server ---
if __name__ == '__main__':
    print("--- Starting Full-Stack Deployment API ---")
    print(f"Deployment URL: http://127.0.0.1:5000/")
    # When deployed, the HOST is set by the platform (like Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)