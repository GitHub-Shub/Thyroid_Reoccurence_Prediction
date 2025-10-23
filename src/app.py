from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import logging
from preprocess import logger  # Reuse logger from preprocess.py for project-wide logging

app = Flask(__name__, template_folder='../web/templates', static_folder='../web/static')

# Paths (relative to src/)
PROCESSED_DIR = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\data\\processed'
MODELS_DIR = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\models'
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.pkl')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
COMPARISON_PATH = os.path.join(MODELS_DIR, 'model_comparison.csv')

# Load model and preprocessor
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    
    # Load best model name dynamically
    if os.path.exists(COMPARISON_PATH):
        comparison_df = pd.read_csv(COMPARISON_PATH)
        BEST_MODEL_NAME = comparison_df.iloc[0]['Model']  # Top model by AUC
    else:
        BEST_MODEL_NAME = 'RandomForest'  # Fallback
    
    logger.info("Model and preprocessor loaded successfully")
    print("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/preprocessor: {str(e)}")
    raise

# For prediction: Simplified mapping (expand to full features as needed)
# This assumes the preprocessor expects all original features; fill missing with defaults
DEFAULT_FEATURES = {
    'Smoking': 'No',
    'Hx Smoking': 'No',
    'Hx Radiotherapy': 'No',
    'Thyroid Function': 'Euthyroid',
    'Physical Examination': 'Single nodular goiter-right',
    'Adenopathy': 'No',
    'Pathology': 'Papillary',
    'Focality': 'Uni-Focal',
    'T': 'T1a',
    'N': 'N0',
    'M': 'M0',
    'Stage': 'I',
    'Response': 'Excellent'
}

@app.route('/')
def index():
    return render_template('index.html', best_model_name=BEST_MODEL_NAME)

@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.json
        # Prepare input DataFrame with all required features
        input_data = DEFAULT_FEATURES.copy()
        input_data.update({
            'Age': int(data.get('age', 40)),
            'Gender': data.get('gender', 'F'),
            'Risk': data.get('risk', 'Low')
            # Add more: e.g., 'Smoking': data.get('smoking', 'No'), etc.
        })
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Predict probability of recurrence (class 1: Yes)
        prob = model.predict_proba(processed_input)[0][1]
        
        # Determine risk level
        if prob < 0.3:
            risk_class = 'low'
            risk_level = 'Low Risk'
        elif prob < 0.7:
            risk_class = 'medium'
            risk_level = 'Medium Risk'
        else:
            risk_class = 'high'
            risk_level = 'High Risk'
        
        logger.info(f"Prediction made: {prob:.4f} for input {data}")
        return jsonify({
            'probability': float(prob),
            'risk_level': risk_level,
            'risk_class': risk_class
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask web app")
    print("Starting Flask web app")
    app.run(debug=True, host='0.0.0.0', port=5000)