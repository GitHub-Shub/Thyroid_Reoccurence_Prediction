"""
preprocess.py: Preprocessing script for Thyroid Cancer Recurrence Prediction Dataset

This script loads the raw CSV, performs cleaning, encoding, feature engineering (if any),
splits into train/test sets, and saves processed files. It ensures reproducibility
by saving encoders and scalers. Logging is configured for both file and console output.

Usage:
    python src/preprocess.py

Dependencies: pandas, scikit-learn, joblib
(Install via: pip install pandas scikit-learn joblib)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import logging

# Constants
RAW_DATA_PATH = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\data\\raw\\Thyroid_Diff.csv'
PROCESSED_DIR = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\data\\processed'
LOGS_DIR = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\logs'
LOG_FILE = os.path.join(LOGS_DIR, 'thyroid_prediction.log')
ENCODING_MAPPING_PATH = os.path.join(PROCESSED_DIR, 'encoding_mapping.json')
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features.csv')
TRAIN_TARGET_PATH = os.path.join(PROCESSED_DIR, 'train_target.csv')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features.csv')
TEST_TARGET_PATH = os.path.join(PROCESSED_DIR, 'test_target.csv')
SCALER_PATH = os.path.join(PROCESSED_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.pkl')

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging for the overall project
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also outputs to console for real-time feedback
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the raw dataset."""
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info("Data loaded successfully")
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info("\nDataset Info:")
        logger.info(df.info())
        logger.info("\nMissing Values:")
        logger.info(df.isnull().sum())
        print("Data loaded successfully")  # User-requested print for better understanding
        return df
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """Basic cleaning: Handle any inconsistencies, but dataset has no nulls."""
    try:
        # Map binary targets: Recurred 'Yes' -> 1, 'No' -> 0
        df['Recurred'] = df['Recurred'].map({'Yes': 1, 'No': 0})
        
        # Ensure Age is numeric (already is)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # Optional: Bin Age into categories for non-linearity (e.g., <30 young, 30-55 middle, >55 old)
        # df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 30, 55, 100], labels=['Young', 'Middle', 'Old'])
        
        logger.info("Data cleaning completed successfully")
        logger.info(f"Cleaned data shape: {df.shape}")
        print("Data cleaned successfully")  # User-requested print for better understanding
        return df
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

def prepare_features_and_target(df):
    """Separate features (X) and target (y). Drop target from features."""
    try:
        target_col = 'Recurred'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        logger.info("Features and target separated successfully")
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Target distribution: {y.value_counts(normalize=True)}")  # Check imbalance
        print("Features and target prepared successfully")  # User-requested print for better understanding
        return X, y
    except Exception as e:
        logger.error(f"Error preparing features and target: {str(e)}")
        raise

def define_preprocessing_pipeline():
    """Define column types and preprocessing steps."""
    # Identify column types (using global X for reference; in practice, pass X)
    global X  # Assuming X is available from prepare_features_and_target
    numeric_features = ['Age']  # Only continuous numeric
    categorical_features = [col for col in X.columns if col not in numeric_features]  # All others are categorical
    
    # Preprocessor: Scale numerics, one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Separate label encoders for ordinal-like (if any, e.g., Risk, T, N, M, Stage)
    # For simplicity, treat all cat as nominal; but if ordinal, use LabelEncoder separately
    ordinal_cols = ['Risk', 'T', 'N', 'M', 'Stage']  # These could be ordinal
    for col in ordinal_cols:
        if col in categorical_features:
            le = LabelEncoder()
            # Fit and transform would happen in fit_transform below
    
    logger.info("Preprocessing pipeline defined successfully")
    print("Preprocessing pipeline defined successfully")  # User-requested print for better understanding
    return preprocessor, numeric_features, categorical_features

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split to preserve class balance."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info("Data split completed successfully")
        logger.info(f"Train split: {X_train.shape[0]} samples ({y_train.value_counts(normalize=True)})")
        logger.info(f"Test split: {X_test.shape[0]} samples ({y_test.value_counts(normalize=True)})")
        print("Data split successfully")  # User-requested print for better understanding
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data split: {str(e)}")
        raise

def fit_and_transform(preprocessor, X_train, X_test, y_train, y_test):
    """Fit preprocessor on train, transform train/test."""
    try:
        # Fit on train
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert back to DataFrames for readability (optional, but useful)
        # Get feature names after encoding
        numeric_transformer_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
        cat_transformer_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(numeric_transformer_names) + list(cat_transformer_names)
        
        X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)
        
        # Save preprocessor for later use (e.g., inference)
        joblib.dump(preprocessor, ENCODER_PATH)
        
        # Save mappings (for categorical encoders, etc.)
        mappings = {}
        cat_encoder = preprocessor.named_transformers_['cat']
        for i, col in enumerate(categorical_features):
            mappings[col] = dict(zip(cat_encoder.categories_[i], range(len(cat_encoder.categories_[i]))))
        with open(ENCODING_MAPPING_PATH, 'w') as f:
            json.dump(mappings, f, indent=4)
        
        # Save scaler separately if needed
        scaler = preprocessor.named_transformers_['num']
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info("Data transformation completed successfully")
        logger.info(f"Processed train shape: {X_train_df.shape}")
        logger.info(f"Processed test shape: {X_test_df.shape}")
        print("Data transformed successfully")  # User-requested print for better understanding
        return X_train_df, X_test_df, y_train, y_test
    except Exception as e:
        logger.error(f"Error during fit and transform: {str(e)}")
        raise

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save to CSV files."""
    try:
        X_train.to_csv(TRAIN_FEATURES_PATH, index=False)
        y_train.to_csv(TRAIN_TARGET_PATH, index=False)
        X_test.to_csv(TEST_FEATURES_PATH, index=False)
        y_test.to_csv(TEST_TARGET_PATH, index=False)
        logger.info("Processed files saved successfully!")
        print("Processed files saved successfully!")  # User-requested print for better understanding
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

def main():
    """Main execution function."""
    logger.info("Starting preprocessing pipeline")
    print("Starting preprocessing pipeline")  # User-requested print for better understanding
    
    global X, numeric_features, categorical_features  # For pipeline definition
    
    # Step 1: Load
    df = load_data()
    
    # Step 2: Clean
    df = clean_data(df)
    
    # Step 3: Prepare X, y
    X, y = prepare_features_and_target(df)
    
    # Step 4: Define pipeline
    preprocessor, numeric_features, categorical_features = define_preprocessing_pipeline()
    
    # Step 5: Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 6: Fit & Transform
    X_train_proc, X_test_proc, y_train, y_test = fit_and_transform(preprocessor, X_train, X_test, y_train, y_test)
    
    # Step 7: Save
    save_processed_data(X_train_proc, X_test_proc, y_train, y_test)
    
    logger.info("Preprocessing pipeline completed successfully")
    print("Preprocessing pipeline completed successfully")  # User-requested print for better understanding

if __name__ == "__main__":
    main()