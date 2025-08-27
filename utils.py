import pandas as pd
import numpy as np
import os
import zipfile
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def load_kaggle_dataset():
    """
    Load hospital readmission dataset from Kaggle or create synthetic data for testing
    """
    try:
        # Try to use Kaggle API
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        dataset_name = "dubradave/hospital-readmissions"
        api.dataset_download_files(dataset_name, path='./data', unzip=True)
        
        # Load the main dataset file
        data_files = os.listdir('./data')
        csv_files = [f for f in data_files if f.endswith('.csv')]
        
        if csv_files:
            main_file = csv_files[0]  # Take the first CSV file
            data = pd.read_csv(f'./data/{main_file}')
            print(f"Loaded dataset with {len(data)} records and {len(data.columns)} columns")
            return data
        else:
            raise FileNotFoundError("No CSV files found in dataset")
    
    except Exception as e:
        print(f"Failed to load from Kaggle: {str(e)}")
        print("Generating synthetic dataset for demonstration...")
        return generate_synthetic_hospital_data()

def generate_synthetic_hospital_data(n_samples=10000):
    """
    Generate synthetic hospital readmission data for testing
    This is only used when real data is not available
    """
    np.random.seed(42)
    
    # Patient demographics
    age = np.random.normal(65, 15, n_samples).clip(18, 100)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    race = np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], 
                           n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    # Admission details
    admission_type = np.random.choice(['Emergency', 'Urgent', 'Elective'], 
                                    n_samples, p=[0.5, 0.3, 0.2])
    admission_source = np.random.choice(['Emergency Room', 'Physician Referral', 'Transfer'], 
                                      n_samples, p=[0.4, 0.4, 0.2])
    
    # Hospital stay details
    time_in_hospital = np.random.poisson(4, n_samples) + 1
    time_in_hospital = np.clip(time_in_hospital, 1, 14)
    
    # Medical procedures and tests
    num_lab_procedures = np.random.poisson(25, n_samples)
    num_procedures = np.random.poisson(1, n_samples)
    num_medications = np.random.poisson(10, n_samples)
    
    # Comorbidities (simplified)
    diabetes = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    hypertension = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    heart_disease = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    
    # Discharge details
    discharge_disposition = np.random.choice(['Home', 'SNF', 'Home Health', 'Rehab'], 
                                           n_samples, p=[0.6, 0.2, 0.15, 0.05])
    
    # Create readmission target (with realistic correlations)
    # Higher readmission risk for: older patients, longer stays, emergency admissions, multiple comorbidities
    readmission_prob = (
        0.1 +  # Base rate
        0.02 * (age - 50) / 20 +  # Age factor
        0.03 * (time_in_hospital > 5) +  # Long stay
        0.04 * (admission_type == 'Emergency') +  # Emergency admission
        0.03 * diabetes +  # Diabetes
        0.02 * hypertension +  # Hypertension
        0.04 * heart_disease +  # Heart disease
        0.02 * (num_medications > 15) +  # Polypharmacy
        np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    
    readmission_prob = np.clip(readmission_prob, 0, 1)
    readmitted = np.random.binomial(1, readmission_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'admission_type': admission_type,
        'admission_source': admission_source,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'discharge_disposition': discharge_disposition,
        'readmitted': readmitted
    })
    
    print(f"Generated synthetic dataset with {len(data)} records")
    print(f"Readmission rate: {data['readmitted'].mean():.1%}")
    
    return data

def validate_performance_targets(results):
    """
    Validate model performance against defined targets
    """
    targets = {
        'accuracy': 0.85,
        'recall': 0.90,
        'precision': 0.80,
        'roc_auc': 0.90
    }
    
    validation_results = {}
    
    for model_name, metrics in results.items():
        model_validation = {}
        for metric, target in targets.items():
            if metric in metrics:
                actual_value = metrics[metric]
                meets_target = actual_value >= target
                model_validation[metric] = (actual_value, meets_target)
        validation_results[model_name] = model_validation
    
    return validation_results

def calculate_clinical_risk_score(probability, age, time_in_hospital, num_comorbidities):
    """
    Calculate a clinical risk score incorporating multiple factors
    """
    base_score = probability * 100
    
    # Age adjustment
    age_factor = max(0, (age - 65) * 0.5)
    
    # Length of stay adjustment
    los_factor = max(0, (time_in_hospital - 3) * 2)
    
    # Comorbidity adjustment
    comorbidity_factor = num_comorbidities * 3
    
    # Combined risk score
    risk_score = min(100, base_score + age_factor + los_factor + comorbidity_factor)
    
    return risk_score

def format_clinical_recommendations(risk_score, top_risk_factors):
    """
    Generate clinical recommendations based on risk score and contributing factors
    """
    recommendations = []
    
    if risk_score >= 70:
        recommendations.extend([
            "🔴 HIGH RISK - Immediate intervention required",
            "• Schedule follow-up within 24-48 hours",
            "• Consider case management consultation",
            "• Review discharge medication reconciliation",
            "• Assess social support and home environment",
            "• Consider home health services"
        ])
    elif risk_score >= 40:
        recommendations.extend([
            "🟡 MEDIUM RISK - Enhanced monitoring recommended", 
            "• Schedule follow-up within 7-10 days",
            "• Provide comprehensive discharge education",
            "• Review medication adherence plan",
            "• Consider telehealth monitoring"
        ])
    else:
        recommendations.extend([
            "🟢 LOW RISK - Standard discharge procedures",
            "• Routine follow-up care as appropriate",
            "• Standard discharge instructions",
            "• Patient education materials"
        ])
    
    # Add specific recommendations based on top risk factors
    if top_risk_factors:
        recommendations.append("\nSpecific interventions based on key risk factors:")
        for factor, importance in top_risk_factors[:3]:
            if 'age' in factor.lower():
                recommendations.append("• Consider geriatric consultation for elderly patients")
            elif 'diabetes' in factor.lower():
                recommendations.append("• Ensure diabetes management plan is in place")
            elif 'time_in_hospital' in factor.lower():
                recommendations.append("• Review reasons for extended stay and address complications")
            elif 'medication' in factor.lower():
                recommendations.append("• Comprehensive medication review and counseling")
    
    return recommendations

def export_model_results(results, filename='model_performance_report.csv'):
    """
    Export model results to CSV for reporting
    """
    df = pd.DataFrame(results).T
    df.to_csv(filename)
    return df

def create_patient_summary(patient_data, prediction_results):
    """
    Create a comprehensive patient summary with risk assessment
    """
    summary = {
        'patient_id': patient_data.get('patient_id', 'Unknown'),
        'demographics': {
            'age': patient_data.get('age'),
            'gender': patient_data.get('gender'),
            'race': patient_data.get('race')
        },
        'admission_details': {
            'admission_type': patient_data.get('admission_type'),
            'time_in_hospital': patient_data.get('time_in_hospital'),
            'discharge_disposition': patient_data.get('discharge_disposition')
        },
        'risk_assessment': prediction_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    return summary
