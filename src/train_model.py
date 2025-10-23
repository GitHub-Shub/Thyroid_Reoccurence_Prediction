

"""
train_model.py: Model training script for Thyroid Cancer Recurrence Prediction

This script loads the processed train/test data, trains multiple classification algorithms
(with class weighting for imbalance where applicable), evaluates performance, compares them,
and saves the best model based on test AUC-ROC. It uses the preprocessor for consistency
and logs the process. Detailed performance metrics (full classification report, confusion matrix,
AUC, CV scores) are printed for each model.

Usage:
    python src/train_model.py

Dependencies: pandas, scikit-learn, joblib
(Assumes preprocess.py has been run to generate data/processed/* files)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import os
import logging
from preprocess import logger  # Reuse logger from preprocess.py for project-wide logging

# Constants
PROCESSED_DIR = 'C:\\Users\\Harpreet\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\data\\processed'
MODELS_DIR = 'C:\\Users\\Harpreet\\Desktop\\Problem Statements\\6_Thyroid_Reoccurence_Prediction\\models'
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'train_features.csv')
TRAIN_TARGET_PATH = os.path.join(PROCESSED_DIR, 'train_target.csv')
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'test_features.csv')
TEST_TARGET_PATH = os.path.join(PROCESSED_DIR, 'test_target.csv')
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, 'preprocessor.pkl')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
RESULTS_PATH = os.path.join(MODELS_DIR, 'model_comparison.csv')  # To save comparison metrics

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Define classifiers with basic params (tunable later)
CLASSIFIERS = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=10),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100, max_depth=10, n_jobs=-1),
    'SVC': SVC(class_weight='balanced', random_state=42, probability=True, kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'GaussianNB': GaussianNB()  # No class_weight, but handles imbalance via priors
}

def load_processed_data():
    """Load processed train/test features and targets."""
    try:
        X_train = pd.read_csv(TRAIN_FEATURES_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).squeeze()  # Single column
        X_test = pd.read_csv(TEST_FEATURES_PATH)
        y_test = pd.read_csv(TEST_TARGET_PATH).squeeze()
        
        # Load preprocessor (not needed for inference here, but for consistency check)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        logger.info("Processed data loaded successfully")
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print("Processed data loaded successfully")  # User-requested print for better understanding
        print(f"Train set shape: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test set shape: {X_test.shape}, Target distribution: {y_test.value_counts().to_dict()}")
        return X_train, y_train, X_test, y_test, preprocessor
    except FileNotFoundError as e:
        logger.error(f"Processed files not found. Run preprocess.py first: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train all classifiers, evaluate, and return results."""
    try:
        results = []
        best_auc = 0
        best_model = None
        best_name = None
        
        for name, model in CLASSIFIERS.items():
            logger.info(f"Training {name}...")
            print(f"\n{'='*60}")
            print(f"Training and Evaluating: {name}")
            print(f"{'='*60}")
            
            # Train
            model.fit(X_train, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0  # SVC has prob=True
            
            # Full classification report
            report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
            
            # Weighted F1 from report
            report_dict = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
            f1 = report_dict['weighted avg']['f1-score']
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation F1 on train
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            cv_f1 = cv_scores.mean()
            
            # Detailed prints
            print(f"\nTest AUC-ROC: {auc:.4f}")
            print(f"\nClassification Report for {name}:")
            print(report)
            print(f"\nConfusion Matrix for {name}:")
            print(cm)
            print(f"\n5-Fold Cross-Validation F1 Scores: {cv_scores}")
            print(f"CV F1 Mean: {cv_f1:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store results
            results.append({
                'Model': name,
                'Test_AUC': auc,
                'Test_F1': f1,
                'Precision_No': report_dict['No']['precision'],
                'Recall_No': report_dict['No']['recall'],
                'F1_No': report_dict['No']['f1-score'],
                'Precision_Yes': report_dict['Yes']['precision'],
                'Recall_Yes': report_dict['Yes']['recall'],
                'F1_Yes': report_dict['Yes']['f1-score'],
                'CV_F1_Mean': cv_f1,
                'CV_F1_Std': cv_scores.std()
            })
            
            logger.info(f"{name} - Test AUC: {auc:.4f}, Test F1: {f1:.4f}, CV F1: {cv_f1:.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"{name} Classification Report:\n{report}")
            logger.info(f"{name} Confusion Matrix:\n{cm}")
            
            # Track best
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_name = name
        
        # Save comparison to CSV with all metrics
        results_df = pd.DataFrame(results).sort_values('Test_AUC', ascending=False)
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Model comparison saved to {RESULTS_PATH}")
        print(f"\n{'='*60}")
        print("Model Comparison Summary (Sorted by Test AUC):")
        print(f"{'='*60}")
        print(results_df.round(4).to_string(index=False))
        print(f"\nModel comparison saved to {RESULTS_PATH}")
        logger.info(f"Best model: {best_name} with AUC: {best_auc:.4f}")
        print(f"\nBest model: {best_name} with AUC: {best_auc:.4f}")
        
        return best_model, best_name, results_df
    except Exception as e:
        logger.error(f"Error during model training/evaluation: {str(e)}")
        raise

def save_model(model, model_name):
    """Save the best trained model."""
    try:
        # Save with name for reference
        model_path = os.path.join(MODELS_DIR, f'{model_name.replace(" ", "_")}_model.pkl')
        joblib.dump(model, model_path)
        joblib.dump(model, BEST_MODEL_PATH)  # Also save as generic best
        logger.info(f"Best model ({model_name}) saved successfully to {BEST_MODEL_PATH} and {model_path}")
        print(f"\nBest model ({model_name}) saved successfully to {BEST_MODEL_PATH}")
        print("Model saved successfully")  # User-requested print for better understanding
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main execution function."""
    logger.info("Starting multi-model training pipeline")
    print("Starting multi-model training pipeline")  # User-requested print for better understanding
    
    # Step 1: Load data
    X_train, y_train, X_test, y_test, _ = load_processed_data()
    
    # Step 2: Train & Evaluate all models
    best_model, best_name, results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Step 3: Save best model
    save_model(best_model, best_name)
    
    logger.info("Multi-model training pipeline completed successfully")
    print("\nMulti-model training pipeline completed successfully")  # User-requested print for better understanding

if __name__ == "__main__":
    main()