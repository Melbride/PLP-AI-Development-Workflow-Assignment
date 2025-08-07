"""
Student Dropout Prediction System
AI Development Workflow Assignment - Part 1 Implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt

class StudentDropoutPredictor:
    """
    AI system to predict student dropout risk
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic student data
        """
        np.random.seed(42)
        
        # Student demographics
        age = np.random.normal(20, 2, n_samples)
        gender = np.random.choice(['M', 'F'], n_samples)
        major = np.random.choice(['Engineering', 'Business', 'Arts', 'Science'], n_samples)
        
        # Academic performance
        high_school_gpa = np.random.normal(3.2, 0.5, n_samples)
        first_semester_gpa = np.random.normal(2.8, 0.7, n_samples)
        credit_hours = np.random.normal(15, 3, n_samples)
        
        # Engagement metrics
        attendance_rate = np.random.beta(8, 2, n_samples)  # Skewed towards high attendance
        lms_logins = np.random.poisson(25, n_samples)
        assignment_submission_rate = np.random.beta(7, 3, n_samples)
        
        # Financial indicators
        financial_aid = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        work_hours = np.random.exponential(10, n_samples)
        
        # Create dropout target
        dropout_prob = (
            0.1 - 
            0.2 * (first_semester_gpa / 4.0) -
            0.15 * attendance_rate -
            0.1 * assignment_submission_rate +
            0.05 * (work_hours / 40) +
            0.02 * np.where(major == 'Engineering', 1, 0)
        )
        
        dropout = np.random.binomial(1, np.clip(dropout_prob, 0, 1), n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'major': major,
            'high_school_gpa': np.clip(high_school_gpa, 0, 4),
            'first_semester_gpa': np.clip(first_semester_gpa, 0, 4),
            'credit_hours': np.clip(credit_hours, 6, 21),
            'attendance_rate': attendance_rate,
            'lms_logins': lms_logins,
            'assignment_submission_rate': assignment_submission_rate,
            'financial_aid': financial_aid,
            'work_hours': np.clip(work_hours, 0, 40),
            'dropout': dropout
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess student data
        """
        # Handle missing values (median for numerical, mode for categorical)
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col].fillna(data[col].median(), inplace=True)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'major']
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Feature engineering
        data['gpa_decline'] = data['high_school_gpa'] - data['first_semester_gpa']
        data['low_engagement'] = np.where(
            (data['attendance_rate'] < 0.7) | (data['assignment_submission_rate'] < 0.7), 1, 0
        )
        data['high_work_load'] = np.where(data['work_hours'] > 20, 1, 0)
        
        return data
    
    def train_model(self, data):
        """
        Train the dropout prediction model
        """
        # Separate features and target
        X = data.drop('dropout', axis=1)
        y = data['dropout']
        
        # Split data (70% train, 15% validation, 15% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val_scaled[numerical_cols] = self.scaler.transform(X_val[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Model Performance on Test Set:")
        print(classification_report(y_test, y_pred))
        
        # Calculate precision at 80% recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        target_recall = 0.8
        idx = np.argmax(recall >= target_recall)
        precision_at_80_recall = precision[idx]
        
        print(f"\nKPI - Precision at 80% Recall: {precision_at_80_recall:.3f}")
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def get_feature_importance(self):
        """
        Get feature importance for interpretability
        """
        feature_names = self.model.feature_names_in_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def main():
    """
    Main execution function
    """
    # Initialize predictor
    predictor = StudentDropoutPredictor()
    
    # Generate synthetic data
    print("Generating synthetic student data...")
    data = predictor.generate_synthetic_data(n_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = predictor.preprocess_data(data)
    
    # Train model
    print("Training dropout prediction model...")
    X_test, y_test, y_pred, y_pred_proba = predictor.train_model(processed_data)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head(10))

if __name__ == "__main__":
    main()