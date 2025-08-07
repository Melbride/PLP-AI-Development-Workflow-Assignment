"""
Bias Analysis and Fairness Assessment
AI Development Workflow Assignment - Ethics Implementation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class BiasAnalyzer:
    """
    Tool for analyzing algorithmic bias in AI models
    """
    
    def __init__(self):
        pass
    
    def demographic_parity_analysis(self, y_true, y_pred, sensitive_attribute):
        """
        Analyze demographic parity across groups
        """
        results = {}
        unique_groups = np.unique(sensitive_attribute)
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            group_predictions = y_pred[mask]
            positive_rate = np.mean(group_predictions)
            results[f'Group_{group}'] = {
                'positive_rate': positive_rate,
                'sample_size': np.sum(mask)
            }
        
        return results
    
    def equalized_odds_analysis(self, y_true, y_pred, sensitive_attribute):
        """
        Analyze equalized odds (TPR and FPR equality across groups)
        """
        results = {}
        unique_groups = np.unique(sensitive_attribute)
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[f'Group_{group}'] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'precision': precision_score(group_true, group_pred, zero_division=0),
                'recall': recall_score(group_true, group_pred, zero_division=0)
            }
        
        return results
    
    def generate_bias_report(self, model_name, y_true, y_pred, sensitive_attribute, 
                           attribute_name="sensitive_attribute"):
        """
        Generate comprehensive bias analysis report
        """
        print(f"=== BIAS ANALYSIS REPORT: {model_name} ===\n")
        
        # Overall model performance
        overall_precision = precision_score(y_true, y_pred)
        overall_recall = recall_score(y_true, y_pred)
        print(f"Overall Model Performance:")
        print(f"Precision: {overall_precision:.3f}")
        print(f"Recall: {overall_recall:.3f}\n")
        
        # Demographic parity analysis
        print(f"1. DEMOGRAPHIC PARITY ANALYSIS ({attribute_name}):")
        dp_results = self.demographic_parity_analysis(y_true, y_pred, sensitive_attribute)
        for group, metrics in dp_results.items():
            print(f"{group}: Positive Rate = {metrics['positive_rate']:.3f}, "
                  f"Sample Size = {metrics['sample_size']}")
        
        # Calculate demographic parity difference
        positive_rates = [metrics['positive_rate'] for metrics in dp_results.values()]
        dp_difference = max(positive_rates) - min(positive_rates)
        print(f"Demographic Parity Difference: {dp_difference:.3f}")
        print(f"Bias Level: {'HIGH' if dp_difference > 0.1 else 'MODERATE' if dp_difference > 0.05 else 'LOW'}\n")
        
        # Equalized odds analysis
        print(f"2. EQUALIZED ODDS ANALYSIS ({attribute_name}):")
        eo_results = self.equalized_odds_analysis(y_true, y_pred, sensitive_attribute)
        for group, metrics in eo_results.items():
            print(f"{group}:")
            print(f"  TPR: {metrics['true_positive_rate']:.3f}")
            print(f"  FPR: {metrics['false_positive_rate']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
        
        # Calculate equalized odds difference
        tpr_values = [metrics['true_positive_rate'] for metrics in eo_results.values()]
        fpr_values = [metrics['false_positive_rate'] for metrics in eo_results.values()]
        tpr_difference = max(tpr_values) - min(tpr_values)
        fpr_difference = max(fpr_values) - min(fpr_values)
        
        print(f"\nTPR Difference: {tpr_difference:.3f}")
        print(f"FPR Difference: {fpr_difference:.3f}")
        print(f"Equalized Odds Bias: {'HIGH' if max(tpr_difference, fpr_difference) > 0.1 else 'MODERATE' if max(tpr_difference, fpr_difference) > 0.05 else 'LOW'}\n")
        
        return {
            'demographic_parity': dp_results,
            'equalized_odds': eo_results,
            'dp_difference': dp_difference,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference
        }

def demonstrate_healthcare_bias():
    """
    Demonstrate bias analysis for healthcare readmission prediction
    """
    # Generate synthetic data with bias
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased dataset
    race = np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.6, 0.25, 0.15])
    income_level = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
    
    # Simulate biased predictions (model performs worse for minorities)
    y_true = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% actual readmission rate
    
    # Create biased predictions
    y_pred = y_true.copy()
    
    # Introduce bias: higher false negative rate for minorities
    minority_mask = (race == 'Black') | (race == 'Hispanic')
    minority_indices = np.where(minority_mask & (y_true == 1))[0]
    
    # Randomly flip 30% of true positives to false negatives for minorities
    flip_indices = np.random.choice(minority_indices, size=int(0.3 * len(minority_indices)), replace=False)
    y_pred[flip_indices] = 0
    
    # Analyze bias
    analyzer = BiasAnalyzer()
    
    print("HEALTHCARE READMISSION PREDICTION - BIAS ANALYSIS")
    print("=" * 60)
    
    # Race-based bias analysis
    race_encoded = np.array([0 if r == 'White' else 1 for r in race])
    bias_report = analyzer.generate_bias_report(
        "Healthcare Readmission Model", y_true, y_pred, race_encoded, "Race (0=White, 1=Minority)"
    )
    
    return bias_report

def main():
    """
    Main execution function
    """
    print("Running bias analysis demonstration...\n")
    
    # Demonstrate healthcare bias
    healthcare_bias = demonstrate_healthcare_bias()
    
    print("\n" + "="*60)
    print("BIAS MITIGATION RECOMMENDATIONS:")
    print("="*60)
    print("1. Collect more diverse training data")
    print("2. Use fairness-aware machine learning algorithms")
    print("3. Implement regular bias audits")
    print("4. Apply post-processing fairness corrections")
    print("5. Engage diverse stakeholders in model development")

if __name__ == "__main__":
    main()