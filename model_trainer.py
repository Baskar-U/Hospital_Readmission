import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, handle_imbalance=True, imbalance_method='smote'):
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        self.models = {}
        self.sampler = None
        
    def train_models(self, X_train, y_train, X_test, y_test, model_names):
        """Train multiple models and return performance metrics"""
        
        # Handle class imbalance if requested
        if self.handle_imbalance:
            X_train_balanced, y_train_balanced = self._handle_class_imbalance(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        results = {}
        
        for model_name in model_names:
            print(f"Training {model_name}...")
            
            # Initialize model
            model = self._initialize_model(model_name, y_train_balanced)
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            
            # Store model
            self.models[model_name] = model
            
            # Evaluate model
            results[model_name] = self._evaluate_model(model, X_test, y_test)
        
        return results
    
    def _handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using specified method"""
        
        if self.imbalance_method == 'smote':
            self.sampler = SMOTE(random_state=42)
        elif self.imbalance_method == 'undersampling':
            self.sampler = RandomUnderSampler(random_state=42)
        else:
            # Class weight handling - return original data
            return X_train, y_train
        
        X_balanced, y_balanced = self.sampler.fit_resample(X_train, y_train)
        return X_balanced, y_balanced
    
    def _initialize_model(self, model_name, y_train):
        """Initialize model with appropriate parameters"""
        
        # Calculate class weights for imbalanced data
        class_weights = None
        if self.handle_imbalance and self.imbalance_method == 'class_weight':
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
        
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict if class_weights is not None else None,
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        elif model_name == "XGBoost":
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = 1
            if self.handle_imbalance and self.imbalance_method == 'class_weight':
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                scale_pos_weight = neg_count / pos_count
            
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight if self.handle_imbalance and self.imbalance_method == 'class_weight' else 1,
                random_state=42,
                eval_metric='logloss'
            )
        
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                class_weight=class_weight_dict if class_weights is not None else None,
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self, model_name):
        """Get feature importance from trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    
    def predict(self, model_name, X):
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """Get prediction probabilities using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.models[model_name].predict_proba(X)
