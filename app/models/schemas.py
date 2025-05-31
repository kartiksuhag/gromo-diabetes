from pydantic import BaseModel, Field
from typing import Optional

class DiabetesPredictionRequest(BaseModel):
    gender: str = Field(..., description="Gender: 'Male' or 'Female'")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension: 0=No, 1=Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease: 0=No, 1=Yes")
    smoking_history: str = Field(..., description="Smoking history: 'never', 'No Info', 'current', 'former', 'ever', 'not current'")
    bmi: float = Field(..., ge=10, le=50, description="Body Mass Index")
    HbA1c_level: float = Field(..., ge=3, le=15, description="HbA1c level")
    blood_glucose_level: int = Field(..., ge=50, le=400, description="Blood glucose level")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "age": 45,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 28.5,
                "HbA1c_level": 6.2,
                "blood_glucose_level": 120
            }
        }

class DiabetesPredictionResponse(BaseModel):
    probability: float = Field(..., description="Diabetes probability (0-1)")
    percentage: float = Field(..., description="Diabetes probability as percentage")
    risk_level: str = Field(..., description="Risk level category")
    risk_color: str = Field(..., description="Risk level color code")
    recommendation: str = Field(..., description="Health recommendation")

class HealthStatus(BaseModel):
    status: str
    message: str
    timestamp: str
