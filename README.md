# AI-Powered Hospital Readmission Risk Prediction

A machine learning application that predicts the risk of hospital readmission for patients using various clinical and demographic features.

## 🏥 Project Overview

This application uses machine learning algorithms to analyze patient data and predict the likelihood of hospital readmission within 30 days of discharge. The system provides both predictions and explanations for its decisions, making it useful for healthcare professionals.

## 🚀 Features

- **Risk Prediction**: Predicts readmission risk using multiple ML algorithms
- **Interactive Dashboard**: Streamlit-based web interface for easy interaction
- **Model Explanations**: SHAP-based explanations for model predictions
- **Data Visualization**: Comprehensive charts and graphs for data analysis
- **Mobile Responsive**: Works on desktop and mobile devices

## 📁 Project Structure

```
HospitalReadmission/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data preprocessing and feature engineering
├── model_trainer.py       # Model training and evaluation
├── explainer.py          # SHAP explanations and model interpretability
├── utils.py              # Utility functions and helpers
├── data/                 # Data directory (data files ignored by Git)
│   └── .gitkeep         # Maintains directory structure
├── pyproject.toml        # Project dependencies and configuration
├── .gitignore           # Git ignore rules for sensitive data
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd HospitalReadmission
   ```

2. **Install dependencies** (using UV):
   ```bash
   uv sync
   ```

3. **Add your data**:
   - Place your `hospital_readmissions.csv` file in the `data/` directory
   - The data file is ignored by Git for privacy and security

## 🚀 Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`
   - The application will load and you can start making predictions

## 📊 Data Requirements

The application expects a CSV file with the following columns:
- Patient demographics (age, gender, etc.)
- Clinical features (diagnosis codes, procedures, etc.)
- Hospital stay information (length of stay, discharge disposition, etc.)
- Previous medical history

## 🔒 Data Privacy

- **Sensitive data is protected**: The `.gitignore` file ensures that data files, model files, and configuration files with sensitive information are not committed to the repository
- **Local processing**: All data processing happens locally on your machine
- **No data transmission**: No patient data is sent to external servers

## 🤖 Models

The application includes multiple machine learning models:
- Random Forest
- XGBoost
- Logistic Regression
- Support Vector Machine

## 📈 Model Performance

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🔍 Model Interpretability

- **SHAP Explanations**: Understand which features contribute to predictions
- **Feature Importance**: Visualize the most important features
- **Individual Predictions**: Get detailed explanations for specific patient cases

## 🛡️ Security Features

- Data files are excluded from version control
- Configuration files with sensitive data are ignored
- Local processing ensures data privacy
- No external API calls for data processing

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with local healthcare data regulations.

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used for actual clinical decision-making without proper validation and regulatory approval.

## 🆘 Support

If you encounter any issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Verify your data format matches the expected structure
4. Check that your data file is in the `data/` directory

## 🔄 Updates

- Keep your dependencies updated: `uv sync`
- Check for new releases and updates
- Review the changelog for breaking changes
