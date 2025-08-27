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
            # Convert to string and clean up
            target = target.astype(str).str.lower().str.strip()
            
            # Handle case where values might be concatenated (like 'yesnonoyesyes...')
            # First check if we have simple binary values
            unique_vals = target.unique()
            
            # Check for simple binary cases first
            simple_binary_patterns = {
                'yes': 1, 'no': 0, 'true': 1, 'false': 0, 
                '1': 1, '0': 0, '>30': 1, '<30': 0, 'readmitted': 1, 'not_readmitted': 0
            }
            
            # If all values are simple binary patterns, use direct mapping
            if all(val in simple_binary_patterns for val in unique_vals):
                target = target.map(simple_binary_patterns)
            else:
                # Handle more complex cases or concatenated strings
                def classify_target(x):
                    x = str(x).lower().strip()
                    
                    # Handle concatenated yes/no strings by counting occurrences
                    if 'yes' in x and 'no' in x:
                        yes_count = x.count('yes')
                        no_count = x.count('no')
                        return 1 if yes_count > no_count else 0
                    elif 'yes' in x or 'true' in x or '>30' in x or 'readmitted' in x:
                        return 1
                    elif 'no' in x or 'false' in x or '<30' in x or 'not_readmitted' in x:
                        return 0
                    else:
                        # Default to 0 for unknown patterns
                        return 0
                
                target = target.apply(classify_target)
        
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
        X_scaled = X.copy()
        
        # Convert all columns to numeric first
        for col in X_scaled.columns:
            if X_scaled[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    X_scaled[col] = pd.to_numeric(X_scaled[col], errors='coerce')
                    # Fill any NaN values that resulted from conversion
                    X_scaled[col] = X_scaled[col].fillna(0)
                except:
                    # If conversion fails, treat as categorical (already encoded)
                    X_scaled[col] = X_scaled[col].astype('category').cat.codes
        
        # Ensure all columns are numeric
        numerical_cols = X_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            if fit:
                X_scaled[numerical_cols] = self.scaler.fit_transform(X_scaled[numerical_cols])
            else:
                X_scaled[numerical_cols] = self.scaler.transform(X_scaled[numerical_cols])
        
        # Convert to float64 to ensure compatibility
        X_scaled = X_scaled.astype('float64')
        
        return X_scaled.values
