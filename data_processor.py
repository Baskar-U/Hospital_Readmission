import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def preprocess_data(self, data):
        """
        Comprehensive data preprocessing pipeline for hospital readmission prediction
        """
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Handle target variable
        target_col = self._identify_target_column(df)
        if target_col is None:
            raise ValueError("Could not identify target column for readmission prediction")
        
        # Prepare target variable
        y = self._prepare_target_variable(df, target_col)
        X = df.drop(columns=[target_col])
        
        # Data cleaning and preprocessing steps
        X = self._handle_missing_values(X)
        X = self._encode_categorical_variables(X)
        X = self._create_derived_features(X)
        X = self._handle_outliers(X)
        
        # Feature selection and engineering
        X = self._select_relevant_features(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        X_train_scaled = self._scale_features(X_train, fit=True)
        X_test_scaled = self._scale_features(X_test, fit=False)
        
        # Convert to DataFrame with proper column names
        self.feature_names = list(X.columns)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names
    
    def _identify_target_column(self, df):
        """Identify the target column for readmission prediction"""
        possible_targets = [
            'readmitted', 'readmission', 'target', 'outcome',
            'readmitted_within_30_days', 'is_readmitted'
        ]
        
        for col in possible_targets:
            if col in df.columns:
                return col
        
        # If no exact match, look for columns with similar patterns
        for col in df.columns:
            if any(pattern in col.lower() for pattern in ['readmit', 'target', 'outcome']):
                return col
        
        return None
    
    def _prepare_target_variable(self, df, target_col):
        """Prepare the target variable for binary classification"""
        target = df[target_col].copy()
        
        # Handle different target variable formats
        if target.dtype == 'object':
            # String values like 'YES'/'NO', '>30'/'<30', etc.
            unique_vals = target.unique()
            if len(unique_vals) == 2:
                # Binary encoding
                positive_indicators = ['yes', 'true', '1', '>30', 'readmitted']
                target = target.astype(str).str.lower()
                target = target.apply(lambda x: 1 if any(indicator in x for indicator in positive_indicators) else 0)
            else:
                # Multi-class to binary (readmission vs no readmission)
                no_readmission_indicators = ['no', '<30', 'not_readmitted', '0']
                target = target.astype(str).str.lower()
                target = target.apply(lambda x: 0 if any(indicator in x for indicator in no_readmission_indicators) else 1)
        
        return target.astype(int)
    
    def _handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        df = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            if 'numerical' not in self.imputers:
                self.imputers['numerical'] = SimpleImputer(strategy='median')
                df[numerical_cols] = self.imputers['numerical'].fit_transform(df[numerical_cols])
            else:
                df[numerical_cols] = self.imputers['numerical'].transform(df[numerical_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            if 'categorical' not in self.imputers:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = self.imputers['categorical'].fit_transform(df[categorical_cols])
            else:
                df[categorical_cols] = self.imputers['categorical'].transform(df[categorical_cols])
        
        return df
    
    def _encode_categorical_variables(self, df):
        """Encode categorical variables using appropriate methods"""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                # Use one-hot encoding for low cardinality, label encoding for high cardinality
                if df[col].nunique() <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                else:
                    # Label encoding for high cardinality
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform using existing encoder
                if df[col].nunique() <= 10 and col in df.columns:
                    # Handle new categories in one-hot encoding
                    df[col] = df[col].astype(str)
                    # This is simplified - in production, you'd handle new categories properly
                else:
                    # Handle new categories in label encoding
                    unique_vals = set(self.label_encoders[col].classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] if x in unique_vals else -1
                    )
        
        return df
    
    def _create_derived_features(self, df):
        """Create derived features based on domain knowledge"""
        df = df.copy()
        
        # Create interaction features for important medical combinations
        if 'age' in df.columns and 'time_in_hospital' in df.columns:
            df['age_hospital_interaction'] = df['age'] * df['time_in_hospital']
        
        # Create risk score features
        if 'num_lab_procedures' in df.columns and 'num_procedures' in df.columns:
            df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures']
        
        # Age groups
        if 'age' in df.columns:
            df['age_group_elderly'] = (df['age'] >= 65).astype(int)
            df['age_group_very_elderly'] = (df['age'] >= 75).astype(int)
        
        # Length of stay categories
        if 'time_in_hospital' in df.columns:
            df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
            df['very_long_stay'] = (df['time_in_hospital'] > 14).astype(int)
        
        return df
    
    def _handle_outliers(self, df):
        """Handle outliers using IQR method for numerical columns"""
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _select_relevant_features(self, df):
        """Select most relevant features for the model"""
        # Remove features with very low variance
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].var() < 1e-6:
                df = df.drop(columns=[col])
        
        # Remove highly correlated features
        corr_matrix = df[df.select_dtypes(include=[np.number]).columns].corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.95
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        df = df.drop(columns=high_corr_features)
        
        return df
    
    def _scale_features(self, X, fit=False):
        """Scale numerical features"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        
        if fit:
            X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X_scaled.values
