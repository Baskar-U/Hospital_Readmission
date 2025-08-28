import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
import os
import warnings
warnings.filterwarnings('ignore')

# Import modules with error handling
try:
    from data_processor import DataProcessor
    from model_trainer import ModelTrainer
    from explainer import ModelExplainer
    from utils import load_kaggle_dataset, validate_performance_targets
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error loading modules: {e}")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="AI-Powered Hospital Readmission Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and description
st.title("🏥 AI-Powered Hospital Readmission Risk Prediction")
st.markdown("""
This system uses advanced machine learning to predict hospital readmissions within 30 days, 
enabling proactive interventions and improved patient outcomes.
""")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'explainer_ready' not in st.session_state:
    st.session_state.explainer_ready = False

# Check if modules are loaded
if not MODULES_LOADED:
    st.error("""
    ⚠️ **Deployment Issue Detected**
    
    Some required modules could not be loaded. This might be due to:
    - Missing dependencies
    - Import errors in custom modules
    
    **Troubleshooting:**
    1. Check that all required packages are in `requirements.txt`
    2. Verify that all Python files are properly formatted
    3. Ensure no syntax errors in custom modules
    
    **For deployment:**
    - Make sure all files are committed to your repository
    - Check the deployment logs for specific error messages
    """)
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Data Loading & Preprocessing",
    "Model Training & Evaluation", 
    "Model Explainability",
    "Clinical Decision Support",
    "Performance Dashboard"
])

# Data Loading & Preprocessing Page
if page == "Data Loading & Preprocessing":
    st.header("📊 Data Loading & Preprocessing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Information")
        st.markdown("""
        **Source**: Kaggle Hospital Readmissions Dataset  
        **Objective**: Predict 30-day hospital readmissions  
        **Features**: Demographics, comorbidities, prior admissions, length of stay, discharge details
        """)
        
        # Check if data file exists locally
        data_file_path = "data/hospital_readmissions.csv"
        if os.path.exists(data_file_path):
            st.success("✅ Local data file found!")
            if st.button("Load Local Dataset"):
                with st.spinner("Loading local dataset..."):
                    try:
                        data = pd.read_csv(data_file_path)
                        if data is not None and not data.empty:
                            # Initialize data processor
                            processor = DataProcessor()
                            
                            # Store in session state
                            st.session_state.raw_data = data
                            st.session_state.processor = processor
                            st.session_state.data_loaded = True
                            
                            st.success(f"Dataset loaded successfully! Shape: {data.shape}")
                            st.rerun()
                        else:
                            st.error("Local data file is empty or invalid.")
                    except Exception as e:
                        st.error(f"Error loading local data: {e}")
        else:
            st.warning("⚠️ No local data file found. You can:")
            
            # Option 1: Upload file
            st.subheader("Option 1: Upload Data File")
            uploaded_file = st.file_uploader("Upload your hospital_readmissions.csv file", type=['csv'])
            if uploaded_file is not None:
                with st.spinner("Loading uploaded dataset..."):
                    try:
                        data = pd.read_csv(uploaded_file)
                        if data is not None and not data.empty:
                            # Initialize data processor
                            processor = DataProcessor()
                            
                            # Store in session state
                            st.session_state.raw_data = data
                            st.session_state.processor = processor
                            st.session_state.data_loaded = True
                            
                            st.success(f"Dataset uploaded successfully! Shape: {data.shape}")
                            st.rerun()
                        else:
                            st.error("Uploaded file is empty or invalid.")
                    except Exception as e:
                        st.error(f"Error loading uploaded file: {e}")
            
            # Option 2: Generate Sample Data
            st.subheader("Option 2: Generate Sample Data")
            st.info("""
            **Demo Mode**: Generate synthetic hospital readmission data for demonstration purposes.
            This is perfect for testing the application without real patient data.
            """)
            
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample dataset..."):
                    try:
                        from sample_data import generate_sample_data
                        data = generate_sample_data(1000)
                        
                        if data is not None:
                            # Initialize data processor
                            processor = DataProcessor()
                            
                            # Store in session state
                            st.session_state.raw_data = data
                            st.session_state.processor = processor
                            st.session_state.data_loaded = True
                            
                            st.success(f"Sample dataset generated successfully! Shape: {data.shape}")
                            st.rerun()
                        else:
                            st.error("Failed to generate sample data.")
                    except Exception as e:
                        st.error(f"Error generating sample data: {e}")
            
            # Option 3: Kaggle API
            st.subheader("Option 3: Load from Kaggle")
            st.info("""
            **Note**: Kaggle API requires authentication and may not work in all deployment environments.
            For deployment, it's recommended to upload your data file directly or use sample data.
            """)
            
            kaggle_username = st.text_input("Kaggle Username", value=os.getenv("KAGGLE_USERNAME", ""))
            kaggle_key = st.text_input("Kaggle API Key", value=os.getenv("KAGGLE_KEY", ""), type="password")
            
            if st.button("Load from Kaggle"):
                if kaggle_username and kaggle_key:
                    with st.spinner("Loading dataset from Kaggle..."):
                        try:
                            # Set environment variables for Kaggle API
                            os.environ['KAGGLE_USERNAME'] = kaggle_username
                            os.environ['KAGGLE_KEY'] = kaggle_key
                            
                            # Load dataset
                            data = load_kaggle_dataset()
                            
                            if data is not None:
                                # Initialize data processor
                                processor = DataProcessor()
                                
                                # Store in session state
                                st.session_state.raw_data = data
                                st.session_state.processor = processor
                                st.session_state.data_loaded = True
                                
                                st.success("Dataset loaded successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to load dataset. Please check your Kaggle credentials and internet connection.")
                        except Exception as e:
                            st.error(f"Error loading from Kaggle: {e}")
                else:
                    st.error("Please provide both Kaggle username and API key.")
        
        # Display data info if loaded
        if st.session_state.data_loaded and hasattr(st.session_state, 'raw_data'):
            st.subheader("Dataset Overview")
            data = st.session_state.raw_data
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Records", f"{len(data):,}")
            with col_info2:
                st.metric("Features", f"{len(data.columns)}")
            with col_info3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show first few rows
            st.subheader("First 5 Rows")
            st.dataframe(data.head())
            
            # Show data types and missing values
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.subheader("Data Types")
                st.dataframe(data.dtypes.to_frame('Data Type'))
            with col_stats2:
                st.subheader("Missing Values")
                missing_data = data.isnull().sum().to_frame('Missing Count')
                missing_data['Percentage'] = (missing_data['Missing Count'] / len(data)) * 100
                st.dataframe(missing_data)
    
    with col2:
        st.subheader("Data Loading Status")
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            st.info("""
            **Next Steps:**
            1. Go to "Model Training & Evaluation"
            2. Train your ML models
            3. Evaluate performance
            """)
        else:
            st.warning("⚠️ No Data Loaded")
            st.info("""
            **To get started:**
            1. Upload your data file, or
            2. Provide Kaggle credentials
            3. Load the dataset
            """)
        
        # Deployment info
        st.subheader("Deployment Info")
        st.info("""
        **For Production:**
        - Upload data files directly
        - Avoid Kaggle API for reliability
        - Use local data processing
        """)

# Model Training & Evaluation Page
elif page == "Model Training & Evaluation":
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first!")
        st.stop()
    
    if not hasattr(st.session_state, 'preprocessed') or not st.session_state.preprocessed:
        st.warning("Please complete data preprocessing first!")
        st.stop()
    
    # Model selection
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_models = st.multiselect(
            "Select Models to Train",
            ["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"],
            default=["Random Forest", "XGBoost", "Logistic Regression"]
        )
        
        handle_imbalance = st.checkbox("Handle Class Imbalance", value=True)
        imbalance_method = st.selectbox(
            "Imbalance Handling Method",
            ["SMOTE", "Class Weight", "Undersampling"],
            disabled=not handle_imbalance
        )
    
    with col2:
        st.write("**Performance Targets**")
        st.write("- Accuracy: ≥85%")
        st.write("- Recall: ≥90%")
        st.write("- Precision: ≥80%")
        st.write("- ROC-AUC: ≥0.9")
    
    if st.button("Train Models"):
        if selected_models:
            with st.spinner("Training models..."):
                try:
                    # Initialize model trainer
                    trainer = ModelTrainer(
                        handle_imbalance=handle_imbalance,
                        imbalance_method=imbalance_method.lower()
                    )
                    
                    # Train models
                    results = trainer.train_models(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        selected_models
                    )
                    
                    # Store results
                    st.session_state.model_results = results
                    st.session_state.trainer = trainer
                    st.session_state.models_trained = True
                    
                    st.success("Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.warning("Please select at least one model to train.")
    
    # Display results
    if st.session_state.models_trained:
        st.subheader("Model Performance")
        
        results = st.session_state.model_results
        
        # Performance metrics table
        performance_df = pd.DataFrame(results).T
        st.dataframe(performance_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Performance targets validation
        st.subheader("Performance Target Analysis")
        targets_met = validate_performance_targets(results)
        
        for model, targets in targets_met.items():
            st.write(f"**{model}**")
            cols = st.columns(4)
            
            for i, (metric, (value, met)) in enumerate(targets.items()):
                with cols[i]:
                    color = "normal" if met else "inverse"
                    st.metric(
                        metric.upper(),
                        f"{value:.3f}",
                        delta="✓" if met else "✗",
                        delta_color=color
                    )
        
        # ROC Curves
        st.subheader("ROC Curves Comparison")
        
        fig = go.Figure()
        
        trainer = st.session_state.trainer
        for model_name in results.keys():
            if hasattr(trainer, 'models') and model_name in trainer.models:
                model = trainer.models[model_name]
                y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                auc = roc_auc_score(st.session_state.y_test, y_pred_proba)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})',
                    line=dict(width=2)
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Model Explainability Page
elif page == "Model Explainability":
    st.header("🔍 Model Explainability")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        st.stop()
    
    # Model selection for explanation
    available_models = list(st.session_state.model_results.keys())
    selected_model = st.selectbox("Select Model to Explain", available_models)
    
    if st.button("Generate Explanations"):
        with st.spinner("Generating model explanations..."):
            try:
                # Initialize explainer
                explainer = ModelExplainer()
                
                # Get model and data
                model = st.session_state.trainer.models[selected_model]
                X_test = st.session_state.X_test
                feature_names = st.session_state.feature_names
                
                # Generate SHAP explanations
                shap_values, explainer_obj = explainer.generate_shap_explanations(
                    model, X_test, feature_names
                )
                
                # Store in session state
                st.session_state.shap_values = shap_values
                st.session_state.explainer_obj = explainer_obj
                st.session_state.selected_model_name = selected_model
                st.session_state.explainer_ready = True
                
                st.success("Explanations generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating explanations: {str(e)}")
    
    # Display explanations
    if st.session_state.explainer_ready:
        st.subheader("Feature Importance Analysis")
        
        # Global feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Global Feature Importance (SHAP)**")
            
            # Calculate mean absolute SHAP values
            shap_values = st.session_state.shap_values
            feature_names = st.session_state.feature_names
            
            if hasattr(shap_values, 'values'):
                mean_shap = np.mean(np.abs(shap_values.values), axis=0)
            else:
                mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Ensure arrays are 1-dimensional
            if len(mean_shap.shape) > 1:
                mean_shap = mean_shap.flatten()
            
            # Ensure feature_names and mean_shap have the same length
            min_length = min(len(feature_names), len(mean_shap))
            feature_names_subset = feature_names[:min_length]
            mean_shap_subset = mean_shap[:min_length]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names_subset,
                'Importance': mean_shap_subset
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**SHAP Summary Plot**")
            
            # Create SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if SHAP_AVAILABLE and shap is not None:
                # Ensure SHAP values and X_test have the same number of rows
                shap_values = st.session_state.shap_values
                X_test = st.session_state.X_test
                
                # Get the actual SHAP values array
                if hasattr(shap_values, 'values'):
                    shap_array = shap_values.values
                else:
                    shap_array = shap_values
                
                # Ensure both arrays have the same number of samples
                min_samples = min(len(shap_array), len(X_test))
                shap_subset = shap_array[:min_samples]
                X_test_subset = X_test.iloc[:min_samples]
                
                try:
                    shap.summary_plot(
                        shap_subset,
                        X_test_subset,
                        feature_names=feature_names,
                        show=False,
                        max_display=15
                    )
                except Exception as e:
                    st.error(f"Error creating SHAP summary plot: {str(e)}")
                    # Fall back to basic plot
                    if len(shap_subset.shape) >= 2:
                        mean_importance = np.mean(np.abs(shap_subset), axis=0)
                        if len(mean_importance.shape) > 1:
                            mean_importance = mean_importance.flatten()
                        
                        min_length = min(len(feature_names), len(mean_importance))
                        feature_names_subset = feature_names[:min_length]
                        mean_importance_subset = mean_importance[:min_length]
                        
                        importance_df = pd.DataFrame({
                            'feature': feature_names_subset,
                            'importance': mean_importance_subset
                        }).sort_values('importance', ascending=True).tail(15)
                        
                        ax.barh(range(len(importance_df)), importance_df['importance'])
                        ax.set_yticks(range(len(importance_df)))
                        ax.set_yticklabels(importance_df['feature'])
                        ax.set_xlabel('Mean |Feature Importance|')
                        ax.set_title('Feature Importance Analysis')
            else:
                # Fallback plot
                shap_values = st.session_state.shap_values
                if hasattr(shap_values, 'shape') and len(shap_values.shape) >= 2:
                    mean_importance = np.mean(np.abs(shap_values), axis=0)
                    
                    # Ensure arrays are 1-dimensional
                    if len(mean_importance.shape) > 1:
                        mean_importance = mean_importance.flatten()
                    
                    # Ensure feature_names and mean_importance have the same length
                    min_length = min(len(feature_names), len(mean_importance))
                    feature_names_subset = feature_names[:min_length]
                    mean_importance_subset = mean_importance[:min_length]
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names_subset,
                        'importance': mean_importance_subset
                    }).sort_values('importance', ascending=True).tail(15)
                    
                    ax.barh(range(len(importance_df)), importance_df['importance'])
                    ax.set_yticks(range(len(importance_df)))
                    ax.set_yticklabels(importance_df['feature'])
                    ax.set_xlabel('Mean |Feature Importance|')
                    ax.set_title('Feature Importance Analysis')
            st.pyplot(fig)
        
        # Individual prediction explanation
        st.subheader("Individual Prediction Explanation")
        
        sample_idx = st.number_input(
            "Select Sample Index",
            min_value=0,
            max_value=len(st.session_state.X_test)-1,
            value=0
        )
        
        if st.button("Explain Individual Prediction"):
            # Get prediction for selected sample
            model = st.session_state.trainer.models[st.session_state.selected_model_name]
            X_sample = st.session_state.X_test.iloc[[sample_idx]]
            
            prediction = model.predict(X_sample)[0]
            probability = model.predict_proba(X_sample)[0, 1]
            
            st.write(f"**Prediction**: {'Readmission' if prediction == 1 else 'No Readmission'}")
            st.write(f"**Probability**: {probability:.3f}")
            
            # SHAP waterfall plot
            if hasattr(st.session_state.shap_values, 'values'):
                sample_shap = st.session_state.shap_values[sample_idx]
            else:
                sample_shap = st.session_state.shap_values[sample_idx]
            
            # Ensure sample_shap is 1-dimensional
            if len(sample_shap.shape) > 1:
                sample_shap = sample_shap.flatten()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if SHAP_AVAILABLE and shap is not None:
                try:
                    # Ensure we have the right data for the selected sample
                    if sample_idx < len(st.session_state.X_test):
                        sample_data = st.session_state.X_test.iloc[sample_idx].values
                        
                        # Ensure sample_shap and sample_data have compatible dimensions
                        if len(sample_shap) != len(sample_data):
                            # Truncate to the smaller length
                            min_length = min(len(sample_shap), len(sample_data))
                            sample_shap_truncated = sample_shap[:min_length]
                            sample_data_truncated = sample_data[:min_length]
                            feature_names_truncated = feature_names[:min_length]
                        else:
                            sample_shap_truncated = sample_shap
                            sample_data_truncated = sample_data
                            feature_names_truncated = feature_names
                        
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=sample_shap_truncated,
                                base_values=st.session_state.explainer_obj.expected_value,
                                data=sample_data_truncated,
                                feature_names=feature_names_truncated
                            ),
                            show=False
                        )
                    else:
                        st.error("Selected sample index is out of range")
                except Exception as e:
                    st.error(f"Error creating waterfall plot: {str(e)}")
                    # Fall back to basic bar plot
            else:
                # Fallback: create basic bar plot
                sample_values = sample_shap  # Already flattened above
                feature_contributions = list(zip(feature_names[:len(sample_values)], sample_values))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Take top 15 features
                top_features = feature_contributions[:15]
                features, values = zip(*top_features) if top_features else ([], [])
                
                if features:
                    colors = ['red' if v < 0 else 'blue' for v in values]
                    ax.barh(range(len(features)), values, color=colors)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel('Feature Contribution')
                    ax.set_title(f'Feature Contributions for Sample {sample_idx}')
                    ax.grid(axis='x', alpha=0.3)
                
            st.pyplot(fig)

# Clinical Decision Support Page
elif page == "Clinical Decision Support":
    st.header("🏥 Clinical Decision Support")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        st.stop()
    
    st.subheader("Patient Risk Assessment")
    
    # Patient input form
    with st.form("patient_assessment"):
        col1, col2, col3 = st.columns(3)
        
        # Sample patient input fields (based on common hospital readmission features)
        with col1:
            st.write("**Demographics**")
            age = st.number_input("Age", min_value=0, max_value=120, value=65)
            gender = st.selectbox("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["Caucasian", "African American", "Hispanic", "Asian", "Other"])
        
        with col2:
            st.write("**Clinical Information**")
            admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
            discharge_disposition = st.selectbox("Discharge Disposition", ["Home", "SNF", "Home Health", "Rehab"])
            time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=3)
        
        with col3:
            st.write("**Medical History**")
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, value=20)
            num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=1)
            num_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=10)
        
        submitted = st.form_submit_button("Assess Risk")
        
        if submitted:
            # Create patient data (simplified for demonstration)
            # In a real system, this would match the exact preprocessing pipeline
            patient_data = pd.DataFrame({
                'age': [age],
                'time_in_hospital': [time_in_hospital],
                'num_lab_procedures': [num_lab_procedures],
                'num_procedures': [num_procedures],
                'num_medications': [num_medications],
                'gender_Male': [1 if gender == 'Male' else 0],
                'race_AfricanAmerican': [1 if race == 'African American' else 0],
                'race_Caucasian': [1 if race == 'Caucasian' else 0],
                'admission_type_id_Emergency': [1 if admission_type == 'Emergency' else 0],
                'discharge_disposition_id_Home': [1 if discharge_disposition == 'Home' else 0]
            })
            
            # Pad with zeros for missing features (simplified approach)
            required_features = len(st.session_state.feature_names)
            current_features = len(patient_data.columns)
            
            if current_features < required_features:
                for i in range(required_features - current_features):
                    patient_data[f'feature_{i}'] = 0
            
            # Ensure column order matches training data
            patient_data = patient_data.reindex(columns=st.session_state.feature_names, fill_value=0)
            
            # Get predictions from all trained models
            st.subheader("Risk Assessment Results")
            
            model_predictions = {}
            for model_name, model in st.session_state.trainer.models.items():
                try:
                    probability = model.predict_proba(patient_data)[0, 1]
                    prediction = model.predict(patient_data)[0]
                    model_predictions[model_name] = {
                        'probability': probability,
                        'prediction': prediction
                    }
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Model Predictions**")
                for model_name, pred in model_predictions.items():
                    risk_level = "High" if pred['probability'] > 0.7 else "Medium" if pred['probability'] > 0.3 else "Low"
                    color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                    
                    st.write(f"{color} **{model_name}**: {pred['probability']:.1%} ({risk_level} Risk)")
            
            with col2:
                st.write("**Clinical Recommendations**")
                
                # Average probability across models
                avg_prob = np.mean([pred['probability'] for pred in model_predictions.values()])
                
                if avg_prob > 0.7:
                    st.error("**HIGH RISK** - Immediate intervention recommended")
                    st.write("Recommendations:")
                    st.write("- Schedule follow-up within 48-72 hours")
                    st.write("- Consider discharge planning consultation")
                    st.write("- Review medication adherence")
                    st.write("- Assess social support systems")
                elif avg_prob > 0.3:
                    st.warning("**MEDIUM RISK** - Enhanced monitoring recommended")
                    st.write("Recommendations:")
                    st.write("- Schedule follow-up within 1 week")
                    st.write("- Provide comprehensive discharge instructions")
                    st.write("- Consider home health services")
                else:
                    st.success("**LOW RISK** - Standard discharge procedures")
                    st.write("Recommendations:")
                    st.write("- Standard follow-up care")
                    st.write("- Routine discharge instructions")
                    st.write("- Patient education materials")

# Performance Dashboard Page
elif page == "Performance Dashboard":
    st.header("📊 Performance Dashboard")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        st.stop()
    
    # Performance overview
    st.subheader("Model Performance Overview")
    
    results = st.session_state.model_results
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Find best performing model for each metric
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    best_auc = max(results.items(), key=lambda x: x[1]['roc_auc'])
    
    with col1:
        st.metric(
            "Best Accuracy",
            f"{best_accuracy[1]['accuracy']:.3f}",
            f"{best_accuracy[0]}"
        )
    
    with col2:
        st.metric(
            "Best Recall",
            f"{best_recall[1]['recall']:.3f}",
            f"{best_recall[0]}"
        )
    
    with col3:
        st.metric(
            "Best Precision", 
            f"{best_precision[1]['precision']:.3f}",
            f"{best_precision[0]}"
        )
    
    with col4:
        st.metric(
            "Best ROC-AUC",
            f"{best_auc[1]['roc_auc']:.3f}",
            f"{best_auc[0]}"
        )
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for model comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig = go.Figure()
        
        for model_name, model_results in results.items():
            values = [model_results[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart for individual metrics
        performance_df = pd.DataFrame(results).T
        
        selected_metric = st.selectbox("Select Metric", metrics)
        
        fig = px.bar(
            x=performance_df.index,
            y=performance_df[selected_metric],
            title=f"{selected_metric.upper()} Comparison",
            labels={'x': 'Model', 'y': selected_metric.upper()}
        )
        
        # Add target line if applicable
        targets = {'accuracy': 0.85, 'recall': 0.90, 'precision': 0.80, 'roc_auc': 0.90}
        if selected_metric in targets:
            fig.add_hline(
                y=targets[selected_metric],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target: {targets[selected_metric]}"
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    if hasattr(st.session_state, 'trainer'):
        trainer = st.session_state.trainer
        cols = st.columns(min(len(results), 3))
        
        for i, (model_name, model) in enumerate(trainer.models.items()):
            if i >= 3:  # Limit to 3 models for display
                break
                
            with cols[i]:
                y_pred = model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{model_name}\nConfusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
    
    # Feature importance comparison (if available)
    if st.session_state.explainer_ready:
        st.subheader("Feature Importance Analysis")
        
        # Display top features across models
        shap_values = st.session_state.shap_values
        feature_names = st.session_state.feature_names
        
        if hasattr(shap_values, 'values'):
            mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Ensure arrays are 1-dimensional
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()
        
        # Ensure feature_names and mean_shap have the same length
        min_length = min(len(feature_names), len(mean_shap))
        feature_names_subset = feature_names[:min_length]
        mean_shap_subset = mean_shap[:min_length]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names_subset,
            'Importance': mean_shap_subset
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features (SHAP values)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**AI-Powered Hospital Readmission Risk Prediction System**  
Built with Streamlit, scikit-learn, XGBoost, and SHAP for explainable AI in healthcare.
""")
