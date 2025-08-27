# AI-Powered Hospital Readmission Risk Prediction

## Overview

This is a comprehensive healthcare AI system designed to predict hospital readmissions within 30 days using advanced machine learning techniques. The application leverages multiple ML algorithms including Random Forest, Gradient Boosting, XGBoost, and Logistic Regression to analyze patient demographics, medical history, and clinical indicators. The system provides explainable AI capabilities through SHAP values, enabling healthcare professionals to understand prediction rationale and make informed clinical decisions. Built with Streamlit for an interactive web interface, the system includes data preprocessing pipelines, model training and evaluation, explainability features, and a clinical decision support dashboard.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Multi-page interface with sidebar navigation for different system functions
- **Interactive Dashboard**: Real-time visualization of model performance, predictions, and explanations
- **Clinical Decision Support Interface**: User-friendly tools for healthcare professionals to input patient data and receive predictions

### Backend Architecture
- **Modular Python Design**: Separated concerns across specialized classes:
  - `DataProcessor`: Handles data cleaning, preprocessing, and feature engineering
  - `ModelTrainer`: Manages multiple ML model training and evaluation
  - `ModelExplainer`: Provides SHAP-based explainability for model predictions
  - `utils`: Contains utility functions for data loading and validation

### Data Processing Pipeline
- **Automated Target Detection**: Intelligent identification of readmission target variables
- **Comprehensive Preprocessing**: Missing value imputation, categorical encoding, outlier handling
- **Feature Engineering**: Creation of derived features and selection of relevant predictors
- **Class Imbalance Handling**: SMOTE oversampling and undersampling techniques
- **Data Scaling**: StandardScaler for numerical feature normalization

### Machine Learning Framework
- **Multi-Model Ensemble**: Support for Random Forest, Gradient Boosting, XGBoost, and Logistic Regression
- **Class Weight Optimization**: Automatic handling of imbalanced datasets common in healthcare
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Cross-Validation**: Stratified splitting to ensure representative train/test splits

### Explainability System
- **SHAP Integration**: Tree explainers for ensemble models and kernel explainers for linear models
- **Feature Importance Analysis**: Multiple methods to identify key predictive factors
- **Clinical Interpretability**: Fallback mechanisms to ensure explanations are always available
- **Visual Explanations**: Plot generation for feature importance and individual prediction explanations

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Primary machine learning framework for model training and evaluation
- **XGBoost**: Advanced gradient boosting implementation
- **imbalanced-learn**: Specialized tools for handling class imbalance (SMOTE, RandomUnderSampler)
- **SHAP**: Model explainability and interpretability framework

### Data Science Stack
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing foundation
- **matplotlib/seaborn**: Statistical visualization and plotting
- **plotly**: Interactive web-based visualizations for dashboard

### Web Interface
- **Streamlit**: Complete web application framework for ML model deployment
- **Session State Management**: Persistent state across page navigation

### Data Sources
- **Kaggle API**: Integration for downloading hospital readmission datasets
- **Synthetic Data Generation**: Fallback system when external data is unavailable
- **CSV File Processing**: Standard data ingestion pipeline

### Development Tools
- **warnings**: Error and warning management for clean user experience
- **os**: File system operations for data handling
- **zipfile/requests**: Data download and extraction utilities