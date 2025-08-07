"""
AI Development Workflow Assignment - Main Runner
Executes all components of the assignment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from student_dropout_predictor import StudentDropoutPredictor
from patient_readmission_predictor import PatientReadmissionPredictor
from bias_analysis import demonstrate_healthcare_bias
from workflow_diagram import create_workflow_diagram

def run_complete_assignment():
    """
    Execute all components of the assignment
    """
    print("="*80)
    print("AI DEVELOPMENT WORKFLOW ASSIGNMENT - COMPLETE EXECUTION")
    print("="*80)
    
    # Part 1: Student Dropout Prediction
    print("\n" + "="*50)
    print("PART 1: STUDENT DROPOUT PREDICTION")
    print("="*50)
    
    try:
        dropout_predictor = StudentDropoutPredictor()
        student_data = dropout_predictor.generate_synthetic_data(1000)
        processed_student_data = dropout_predictor.preprocess_data(student_data)
        dropout_predictor.train_model(processed_student_data)
        
        importance_df = dropout_predictor.get_feature_importance()
        print("\nTop 5 Most Important Features for Student Dropout:")
        print(importance_df.head().to_string(index=False))
        
    except Exception as e:
        print(f"Error in student dropout prediction: {e}")
    
    # Part 2: Patient Readmission Prediction (Case Study)
    print("\n" + "="*50)
    print("PART 2: PATIENT READMISSION PREDICTION (CASE STUDY)")
    print("="*50)
    
    try:
        readmission_predictor = PatientReadmissionPredictor()
        patient_data = readmission_predictor.generate_synthetic_data(1000)
        processed_patient_data = readmission_predictor.preprocess_data(patient_data)
        readmission_predictor.train_model(processed_patient_data)
        
        importance_df = readmission_predictor.get_feature_importance()
        print("\nTop 5 Most Important Features for Readmission Prediction:")
        print(importance_df.head().to_string(index=False))
        
    except Exception as e:
        print(f"Error in patient readmission prediction: {e}")
    
    # Part 3: Bias Analysis
    print("\n" + "="*50)
    print("PART 3: BIAS ANALYSIS AND ETHICS")
    print("="*50)
    
    try:
        demonstrate_healthcare_bias()
    except Exception as e:
        print(f"Error in bias analysis: {e}")
    
    # Part 4: Workflow Diagram
    print("\n" + "="*50)
    print("PART 4: WORKFLOW DIAGRAM GENERATION")
    print("="*50)
    
    try:
        create_workflow_diagram()
        print("Workflow diagram generated successfully!")
    except Exception as e:
        print(f"Error generating workflow diagram: {e}")
    
    print("\n" + "="*80)
    print("ASSIGNMENT EXECUTION COMPLETED")
    print("="*80)
    print("\nFiles generated:")
    print("- Complete assignment document: docs/AI_Development_Workflow_Assignment.md")
    print("- Student dropout predictor: code/student_dropout_predictor.py")
    print("- Patient readmission predictor: code/patient_readmission_predictor.py")
    print("- Bias analysis tool: code/bias_analysis.py")
    print("- Workflow diagram: code/workflow_diagram.py")
    print("- Project documentation: README.md")
    print("- Dependencies: requirements.txt")

if __name__ == "__main__":
    run_complete_assignment()