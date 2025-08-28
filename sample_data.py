"""
Sample data generator for Hospital Readmission prediction demo.
This creates synthetic data for demonstration purposes when real data is not available.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_data(n_samples=1000):
    """
    Generate sample hospital readmission data for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample hospital readmission data
    """
    
    # Generate synthetic features
    np.random.seed(42)
    
    # Patient demographics
    age = np.random.normal(65, 15, n_samples)
    age = np.clip(age, 18, 100)
    
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Medical features
    num_lab_procedures = np.random.poisson(25, n_samples)
    num_procedures = np.random.poisson(1.5, n_samples)
    num_medications = np.random.poisson(15, n_samples)
    num_outpatient = np.random.poisson(2, n_samples)
    num_emergency = np.random.poisson(1, n_samples)
    num_inpatient = np.random.poisson(0.5, n_samples)
    
    # Time in hospital
    time_in_hospital = np.random.poisson(4, n_samples)
    time_in_hospital = np.clip(time_in_hospital, 1, 14)
    
    # Number of diagnoses
    number_diagnoses = np.random.poisson(8, n_samples)
    number_diagnoses = np.clip(number_diagnoses, 1, 16)
    
    # Medical specialty
    medical_specialty = np.random.choice([
        'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
        'Cardiology', 'Surgery-General', 'Orthopedics', 'Other'
    ], n_samples)
    
    # Admission type
    admission_type = np.random.choice([
        'Emergency', 'Elective', 'Urgent', 'Newborn'
    ], n_samples)
    
    # Discharge disposition
    discharge_disposition = np.random.choice([
        'Discharged to home', 'Discharged/transferred to another facility',
        'Discharged/transferred to home with care', 'Expired'
    ], n_samples)
    
    # Admission source
    admission_source = np.random.choice([
        'Emergency Room', 'Physician Referral', 'Transfer from a hospital',
        'Transfer from a Skilled Nursing Facility (SNF)', 'Clinic Referral'
    ], n_samples)
    
    # Race
    race = np.random.choice([
        'Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'
    ], n_samples)
    
    # Diabetes medication
    diabetes_med = np.random.choice(['Yes', 'No'], n_samples)
    
    # Change in medication
    change = np.random.choice(['No', 'Ch'], n_samples)
    
    # Generate target variable (readmission) with some correlation to features
    # Higher risk for older patients, more procedures, longer stays
    risk_score = (
        (age - 65) / 15 * 0.3 +  # Age factor
        (num_procedures - 1.5) / 1.5 * 0.2 +  # Procedure factor
        (time_in_hospital - 4) / 4 * 0.3 +  # Length of stay factor
        (num_medications - 15) / 15 * 0.1 +  # Medication factor
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Convert risk score to probability
    readmission_prob = 1 / (1 + np.exp(-risk_score))
    readmitted = np.random.binomial(1, readmission_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age.astype(int),
        'gender': gender,
        'race': race,
        'admission_type': admission_type,
        'admission_source': admission_source,
        'discharge_disposition': discharge_disposition,
        'medical_specialty': medical_specialty,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': num_outpatient,
        'number_emergency': num_emergency,
        'number_inpatient': num_inpatient,
        'number_diagnoses': number_diagnoses,
        'diabetesMed': diabetes_med,
        'change': change,
        'readmitted': readmitted
    })
    
    return data

def save_sample_data(filepath="data/sample_hospital_readmissions.csv"):
    """
    Generate and save sample data to CSV file.
    
    Args:
        filepath (str): Path to save the sample data
    """
    try:
        # Generate sample data
        sample_data = generate_sample_data(1000)
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to CSV
        sample_data.to_csv(filepath, index=False)
        print(f"Sample data saved to {filepath}")
        print(f"Shape: {sample_data.shape}")
        print(f"Readmission rate: {sample_data['readmitted'].mean():.2%}")
        
        return sample_data
        
    except Exception as e:
        print(f"Error saving sample data: {e}")
        return None

if __name__ == "__main__":
    # Generate and save sample data
    sample_data = save_sample_data()
    if sample_data is not None:
        print("\nSample data preview:")
        print(sample_data.head())
        print("\nData types:")
        print(sample_data.dtypes)
