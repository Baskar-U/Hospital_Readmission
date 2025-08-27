import numpy as np
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        
    def generate_shap_explanations(self, model, X_test, feature_names, max_samples=1000):
        """Generate SHAP explanations for model predictions"""
        
        # Limit samples for faster computation
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
        else:
            X_sample = X_test
        
        if not SHAP_AVAILABLE:
            # Fallback: use feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                # Create mock SHAP values structure
                mock_shap = np.tile(importance, (len(X_sample), 1))
                mock_explainer = type('MockExplainer', (), {'expected_value': 0.5})()
                return mock_shap, mock_explainer
            else:
                raise ValueError("SHAP not available and model has no feature_importances_")
        
        try:
            # Try TreeExplainer first (for tree-based models)
            if hasattr(model, 'estimators_') or hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                # Create background dataset (sample from training data if available)
                background_size = min(100, len(X_sample))
                background = shap.sample(X_sample, background_size)
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, get positive class SHAP values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Positive class
            
            # Store explainer and values
            self.explainers[str(model)] = explainer
            self.shap_values[str(model)] = shap_values
            
            return shap_values, explainer
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {str(e)}")
            # Fallback: return feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                # Create mock SHAP values structure
                mock_shap = np.tile(importance, (len(X_sample), 1))
                return mock_shap, None
            else:
                raise e
    
    def plot_summary(self, shap_values, X_test, feature_names, max_display=20):
        """Create SHAP summary plot"""
        
        plt.figure(figsize=(10, 8))
        
        if SHAP_AVAILABLE and shap is not None:
            shap.summary_plot(
                shap_values, 
                X_test,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
        else:
            # Fallback: create basic feature importance plot
            if hasattr(shap_values, 'shape') and len(shap_values.shape) == 2:
                mean_importance = np.mean(np.abs(shap_values), axis=0)
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(mean_importance)],
                    'importance': mean_importance
                }).sort_values('importance', ascending=True).tail(max_display)
                
                plt.barh(range(len(importance_df)), importance_df['importance'])
                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Mean |Feature Importance|')
                plt.title('Feature Importance (Fallback)')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waterfall(self, shap_values, explainer, X_test, feature_names, sample_idx=0):
        """Create SHAP waterfall plot for individual prediction"""
        
        plt.figure(figsize=(12, 8))
        
        if SHAP_AVAILABLE and shap is not None:
            if hasattr(shap_values, 'values'):
                # New SHAP format
                shap.waterfall_plot(shap_values[sample_idx], show=False)
            else:
                # Legacy format
                expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=expected_value,
                        data=X_test.iloc[sample_idx].values if hasattr(X_test, 'iloc') else X_test[sample_idx],
                        feature_names=feature_names
                    ),
                    show=False
                )
        else:
            # Fallback: create basic bar plot of feature contributions
            sample_values = shap_values[sample_idx] if len(shap_values.shape) > 1 else shap_values
            feature_contributions = list(zip(feature_names[:len(sample_values)], sample_values))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Take top 15 features
            top_features = feature_contributions[:15]
            features, values = zip(*top_features)
            
            colors = ['red' if v < 0 else 'blue' for v in values]
            plt.barh(range(len(features)), values, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Contribution')
            plt.title(f'Feature Contributions for Sample {sample_idx} (Fallback)')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_force(self, shap_values, explainer, X_test, sample_idx=0):
        """Create SHAP force plot for individual prediction"""
        
        expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        force_plot = shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            X_test.iloc[sample_idx] if hasattr(X_test, 'iloc') else X_test[sample_idx],
            matplotlib=True,
            show=False
        )
        
        return force_plot
    
    def get_feature_importance(self, shap_values, feature_names, top_k=10):
        """Get top-k most important features based on mean absolute SHAP values"""
        
        if hasattr(shap_values, 'values'):
            mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_k)
        
        return importance_df
    
    def explain_prediction(self, model, X_sample, feature_names, prediction_idx=0):
        """Provide detailed explanation for a specific prediction"""
        
        # Get prediction
        prediction = model.predict(X_sample.iloc[[prediction_idx]] if hasattr(X_sample, 'iloc') else X_sample[[prediction_idx]])
        probability = model.predict_proba(X_sample.iloc[[prediction_idx]] if hasattr(X_sample, 'iloc') else X_sample[[prediction_idx]])[0]
        
        # Get SHAP values for this prediction
        model_key = str(model)
        if model_key in self.shap_values:
            shap_vals = self.shap_values[model_key][prediction_idx]
            
            # Create explanation dictionary
            explanation = {
                'prediction': prediction[0],
                'probability': {
                    'no_readmission': probability[0],
                    'readmission': probability[1]
                },
                'top_positive_factors': [],
                'top_negative_factors': []
            }
            
            # Get top positive and negative contributing factors
            feature_contributions = list(zip(feature_names, shap_vals))
            feature_contributions.sort(key=lambda x: x[1], reverse=True)
            
            explanation['top_positive_factors'] = feature_contributions[:5]
            explanation['top_negative_factors'] = feature_contributions[-5:]
            
            return explanation
        
        else:
            return {
                'prediction': prediction[0],
                'probability': {
                    'no_readmission': probability[0],
                    'readmission': probability[1]
                },
                'explanation': 'SHAP values not available for this model'
            }
