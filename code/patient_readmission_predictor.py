"""
Patient Readmission Risk Prediction System
AI Development Workflow Assignment - Case Study Implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class PatientReadmissionPredictor:
    """
    AI system to predict 30-day patient readmission risk
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic patient data for demonstration
        """
        np.random.seed(42)
        
        # Patient demographics
        age = np.random.normal(65, 15, n_samples)
        gender = np.random.choice(['M', 'F'], n_samples)
        
        # Clinical indicators
        length_of_stay = np.random.exponential(5, n_samples)
        num_diagnoses = np.random.poisson(3, n_samples)
        num_procedures = np.random.poisson(2, n_samples)
        
        # Lab values (simplified)
        hemoglobin = np.random.normal(12, 2, n_samples)
        creatinine = np.random.exponential(1.2, n_samples)
        
        # Comorbidities
        diabetes = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        hypertension = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        heart_failure = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Discharge disposition
        discharge_home = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # Create readmission target (simplified logic)
        readmission_prob = (
            0.1 + 
            0.002 * age + 
            0.01 * length_of_stay + 
            0.05 * num_diagnoses + 
            0.1 * heart_failure + 
            0.05 * diabetes -
            0.1 * discharge_home
        )
        
        readmission = np.random.binomial(1, np.clip(readmission_prob, 0, 1), n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'length_of_stay': length_of_stay,
            'num_diagnoses': num_diagnoses,
            'num_procedures': num_procedures,
            'hemoglobin': hemoglobin,
            'creatinine': creatinine,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_failure': heart_failure,
            'discharge_home': discharge_home,
            'readmission_30_days': readmission
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess patient data for model training
        """
        # Handle missing values
        data = data.fillna(data.median(numeric_only=True))
        
        # Encode categorical variables
        categorical_cols = ['gender']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Feature engineering
        data['age_risk_score'] = np.where(data['age'] > 70, 1, 0)
        data['high_complexity'] = np.where(data['num_diagnoses'] > 5, 1, 0)
        data['anemia_indicator'] = np.where(data['hemoglobin'] < 10, 1, 0)
        
        return data
    
    def train_model(self, data):
        """
        Train the readmission prediction model
        """
        # Separate features and target
        X = data.drop('readmission_30_days', axis=1)
        y = data['readmission_30_days']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def predict_readmission_risk(self, patient_data):
        """
        Predict readmission risk for new patients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Preprocess input data
        processed_data = self.preprocess_data(patient_data.copy())
        
        # Scale numerical features
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numerical_cols] = self.scaler.transform(processed_data[numerical_cols])
        
        # Make predictions
        risk_scores = self.model.predict_proba(processed_data)[:, 1]
        risk_categories = ['Low' if score < 0.3 else 'Medium' if score < 0.7 else 'High' 
                          for score in risk_scores]
        
        return risk_scores, risk_categories
    
    def get_feature_importance(self):
        """
        Get feature importance for model interpretability
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    """
    Main execution function
    """
    # Initialize predictor
    predictor = PatientReadmissionPredictor()
    
    # Generate synthetic data
    print("Generating synthetic patient data...")
    data = predictor.generate_synthetic_data(n_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = predictor.preprocess_data(data)
    
    # Train model
    print("Training readmission prediction model...")
    X_test, y_test, y_pred, y_pred_proba = predictor.train_model(processed_data)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head(10))
    
    # Example prediction for new patient
    print("\nExample: Predicting risk for new patient...")
    new_patient = pd.DataFrame({
        'age': [75],
        'gender': ['M'],
        'length_of_stay': [8],
        'num_diagnoses': [6],
        'num_procedures': [3],
        'hemoglobin': [9.5],
        'creatinine': [2.1],
        'diabetes': [1],
        'hypertension': [1],
        'heart_failure': [1],
        'discharge_home': [0]
    })
    
    risk_scores, risk_categories = predictor.predict_readmission_risk(new_patient)
    print(f"Readmission Risk Score: {risk_scores[0]:.3f}")
    print(f"Risk Category: {risk_categories[0]}")

if __name__ == "__main__":
    main()