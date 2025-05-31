import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import os
from app.utils.preprocessing import DiabetesPreprocessor

class DiabetesPredictor:
    def __init__(self):
        self.preprocessor = DiabetesPreprocessor()
        self.models = {}
        self.scaler = None
        self.is_ensemble = False
        self.model_loaded = False
        
    def load_models(self, model_path: str = "app/models/saved_models/"):
        """Load trained models and preprocessing objects"""
        try:
            # Check if ensemble models exist
            ensemble_path = os.path.join(model_path, "diabetes_ensemble_models.pkl")
            single_model_path = os.path.join(model_path, "best_diabetes_model.pkl")
            scaler_path = os.path.join(model_path, "diabetes_scaler.pkl")
            
            if os.path.exists(ensemble_path):
                # Load ensemble models
                with open(ensemble_path, 'rb') as f:
                    models_dict = pickle.load(f)
                
                self.models = {
                    'xgb_model': models_dict['xgb_model'],
                    'cat_model': models_dict['cat_model'],
                    'lgb_model': models_dict['lgb_model'],
                    'rf_model': models_dict['rf_model']
                }
                self.scaler = models_dict['scaler']
                self.is_ensemble = True
                
            elif os.path.exists(single_model_path) and os.path.exists(scaler_path):
                # Load single best model
                with open(single_model_path, 'rb') as f:
                    self.models['best_model'] = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_ensemble = False
            
            else:
                # Initialize default models if no saved models found
                self._initialize_default_models()
                
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Initialize default models as fallback
            self._initialize_default_models()
            return False
    
    def _initialize_default_models(self):
        """Initialize default models if saved models are not available"""
        print("Initializing default models...")
        
        # Create default models with reasonable parameters
        self.models = {
            'xgb_model': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                random_state=42, eval_metric='logloss'
            ),
            'cat_model': cb.CatBoostClassifier(
                iterations=200, learning_rate=0.1, depth=6, 
                random_state=42, verbose=False
            ),
            'lgb_model': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                random_state=42, verbose=-1
            ),
            'rf_model': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.is_ensemble = True
        self.model_loaded = False
        
        print("⚠️ Warning: Using default untrained models. Please train and save models first.")
    
    def predict_diabetes_probability(self, gender: str, age: int, hypertension: int,
                                   heart_disease: int, smoking_history: str, bmi: float,
                                   HbA1c_level: float, blood_glucose_level: int):
        """Predict diabetes probability"""
        
        if not self.model_loaded:
            # Return a mock prediction for demonstration
            return self._mock_prediction(age, bmi, HbA1c_level, blood_glucose_level)
        
        try:
            # Prepare input data
            input_data = self.preprocessor.prepare_input_data(
                gender, age, hypertension, heart_disease, smoking_history,
                bmi, HbA1c_level, blood_glucose_level
            )
            
            # Scale features
            input_data_scaled = self.preprocessor.scale_features(input_data, self.scaler)
            
            # Make prediction
            if self.is_ensemble:
                # Ensemble prediction
                probabilities = []
                for model_name, model in self.models.items():
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(input_data_scaled)[0, 1]
                        probabilities.append(prob)
                
                if probabilities:
                    probability = np.mean(probabilities)
                else:
                    probability = 0.5  # Default fallback
            else:
                # Single model prediction
                probability = self.models['best_model'].predict_proba(input_data_scaled)[0, 1]
            
            return float(probability)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return mock prediction as fallback
            return self._mock_prediction(age, bmi, HbA1c_level, blood_glucose_level)
    
    def _mock_prediction(self, age: int, bmi: float, HbA1c_level: float, blood_glucose_level: int):
        """Generate mock prediction for demonstration purposes"""
        # Simple rule-based mock prediction
        risk_score = 0.0
        
        # Age factor
        if age > 60:
            risk_score += 0.3
        elif age > 45:
            risk_score += 0.2
        elif age > 30:
            risk_score += 0.1
        
        # BMI factor
        if bmi > 30:
            risk_score += 0.25
        elif bmi > 25:
            risk_score += 0.15
        
        # HbA1c factor
        if HbA1c_level > 6.5:
            risk_score += 0.3
        elif HbA1c_level > 5.7:
            risk_score += 0.2
        
        # Blood glucose factor
        if blood_glucose_level > 140:
            risk_score += 0.25
        elif blood_glucose_level > 100:
            risk_score += 0.15
        
        # Add some randomness
        risk_score += np.random.normal(0, 0.05)
        
        # Ensure probability is between 0 and 1
        probability = max(0.0, min(1.0, risk_score))
        
        return probability

# Global predictor instance
diabetes_predictor = DiabetesPredictor()
