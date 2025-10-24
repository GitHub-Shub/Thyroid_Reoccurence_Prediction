from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from preprocess import logger  # reuse global logger

app = Flask(__name__,
            template_folder=r'C:\Users\Harpreet\Desktop\Problem Statements\6_Thyroid_Reoccurence_Prediction\web\static\templates',
            static_folder=r'C:\Users\Harpreet\Desktop\Problem Statements\6_Thyroid_Reoccurence_Prediction\web\static')

# Paths
PROCESSED_DIR = r'C:\Users\Harpreet\Desktop\Problem Statements\6_Thyroid_Reoccurence_Prediction\data\processed'
MODELS_DIR = r'C:\Users\Harpreet\Desktop\Problem Statements\6_Thyroid_Reoccurence_Prediction\models'
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.pkl')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
COMPARISON_PATH = os.path.join(MODELS_DIR, 'model_comparison.csv')

# Load model and preprocessor
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    if os.path.exists(COMPARISON_PATH):
        comparison_df = pd.read_csv(COMPARISON_PATH)
        BEST_MODEL_NAME = comparison_df.iloc[0]['Model']
    else:
        BEST_MODEL_NAME = 'RandomForest'
    logger.info("Model and preprocessor loaded successfully")
    print("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/preprocessor: {str(e)}")
    raise

# Default features
DEFAULT_FEATURES = {
    'Smoking': 'No',
    'Hx Smoking': 'No',
    'Hx Radiothreapy': 'No',  # note exact spelling in model
    'Thyroid Function': 'Euthyroid',
    'Physical Examination': 'Single nodular goiter-right',
    'Adenopathy': 'No',
    'Pathology': 'Papillary',
    'Focality': 'Uni-Focal',
    'T': 'T1a',
    'N': 'N0',
    'M': 'M0',
    'Stage': 'I',
    'Response': 'Excellent',
    'Risk': 'Low',
    'Gender': 'F',
    'Age': 40
}


@app.route('/')
def index():
    return render_template('index.html', best_model_name=BEST_MODEL_NAME)


@app.route('/', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    try:
        data = request.get_json() or request.form.to_dict()
        print("Received raw data:", data)

        # Normalize capitalization
        def clean(v): 
            return v.strip().capitalize() if isinstance(v, str) else v

        input_data = DEFAULT_FEATURES.copy()
        input_data.update({
            'Age': int(data.get('age', 40)),
            'Gender': clean(data.get('gender', 'F')),
            'Risk': clean(data.get('risk', 'Low')),
            'Smoking': clean(data.get('smoking', 'No')),
            'Hx Smoking': clean(data.get('hx_smoking', 'No')),
            'Hx Radiotherapy': clean(data.get('hx_radiotherapy', 'No')),  # ✅ fixed spelling
            'Thyroid Function': clean(data.get('thyroid_function', 'Euthyroid')),
            'Physical Examination': clean(data.get('physical_examination', 'Single nodular goiter-right')),
            'Adenopathy': clean(data.get('adenopathy', 'No')),
            'Pathology': clean(data.get('pathology', 'Papillary')),
            'Focality': clean(data.get('focality', 'Uni-Focal')),
            'T': data.get('t', 'T1a').upper(),
            'N': data.get('n', 'N0').upper(),
            'M': data.get('m', 'M0').upper(),
            'Stage': data.get('stage', 'I').upper(),
            'Response': clean(data.get('response', 'Excellent'))
        })

        input_df = pd.DataFrame([input_data])
        print("Processed input:", input_df)

        processed_input = preprocessor.transform(input_df)
        prob = model.predict_proba(processed_input)[0][1]

        # Stable classification thresholds
        if prob < 0.3:
            risk_class, risk_level = 'low', 'Low Risk'
        elif prob < 0.7:
            risk_class, risk_level = 'medium', 'Medium Risk'
        else:
            risk_class, risk_level = 'high', 'High Risk'

        return jsonify({
            'probability': round(float(prob), 4),
            'risk_level': risk_level,
            'risk_class': risk_class
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask web app")
    print("Starting Flask web app")
    app.run(debug=False, host='0.0.0.0', port=5000)  # ✅ debug=False
