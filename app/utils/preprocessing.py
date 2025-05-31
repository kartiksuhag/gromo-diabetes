import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DiabetesPreprocessor:
    def __init__(self):
        self.scaler = None
        self.numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        self.smoking_map = {
            'never': 0, 'no info': 1, 'current': 2,
            'former': 3, 'ever': 4, 'not current': 5
        }
    
    def encode_categorical_features(self, gender: str, smoking_history: str):
        """Encode categorical features"""
        # Encode gender
        gender_encoded = 1 if gender.lower() == 'male' else 0
        
        # Encode smoking history
        smoking_encoded = self.smoking_map.get(smoking_history.lower(), 0)
        
        return gender_encoded, smoking_encoded
    
    def prepare_input_data(self, gender: str, age: int, hypertension: int, 
                          heart_disease: int, smoking_history: str, bmi: float,
                          HbA1c_level: float, blood_glucose_level: int):
        """Prepare input data for prediction"""
        
        # Encode categorical variables
        gender_encoded, smoking_encoded = self.encode_categorical_features(gender, smoking_history)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender_encoded],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_encoded],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })
        
        return input_data
    
    def scale_features(self, input_data: pd.DataFrame, scaler: StandardScaler):
        """Scale numerical features"""
        input_data_scaled = input_data.copy()
        input_data_scaled[self.numerical_features] = scaler.transform(input_data[self.numerical_features])
        return input_data_scaled
    
    def categorize_risk(self, probability: float):
        """Categorize diabetes risk based on probability"""
        if probability < 0.2:
            return "🟢 LOW RISK", "#4CAF50", "Maintain healthy lifestyle habits"
        elif probability < 0.5:
            return "🟡 MODERATE RISK", "#FF9800", "Consider lifestyle modifications and regular check-ups"
        elif probability < 0.8:
            return "🟠 HIGH RISK", "#FF5722", "Consult healthcare provider and implement preventive measures"
        else:
            return "🔴 VERY HIGH RISK", "#F44336", "Immediate medical consultation recommended"
