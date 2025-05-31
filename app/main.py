from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os
from contextlib import asynccontextmanager

from app.models.schemas import (
    DiabetesPredictionRequest, 
    DiabetesPredictionResponse, 
    HealthStatus
)
from app.models.diabetes_model import diabetes_predictor
from app.utils.preprocessing import DiabetesPreprocessor

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Diabetes Prediction API...")
    
    # Load models on startup
    model_loaded = diabetes_predictor.load_models()
    if model_loaded:
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Models not found, using default configuration")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Diabetes Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Advanced ML-powered diabetes risk assessment API using ensemble methods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize preprocessor
preprocessor = DiabetesPreprocessor()

@app.get("/", response_model=HealthStatus)
async def root():
    """Root endpoint - API health check"""
    return HealthStatus(
        status="healthy",
        message="Diabetes Prediction API is running successfully! 🩺",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    return HealthStatus(
        status="healthy",
        message="API is operational",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=DiabetesPredictionResponse)
async def predict_diabetes(request: DiabetesPredictionRequest):
    """
    Predict diabetes probability based on patient data
    
    This endpoint uses an ensemble of machine learning models including:
    - XGBoost
    - CatBoost  
    - LightGBM
    - Random Forest
    
    Returns probability, risk level, and health recommendations.
    """
    try:
        # Validate input ranges
        if not (0 <= request.age <= 120):
            raise HTTPException(status_code=400, detail="Age must be between 0 and 120")
        
        if not (10 <= request.bmi <= 50):
            raise HTTPException(status_code=400, detail="BMI must be between 10 and 50")
        
        if not (3 <= request.HbA1c_level <= 15):
            raise HTTPException(status_code=400, detail="HbA1c level must be between 3 and 15")
        
        if not (50 <= request.blood_glucose_level <= 400):
            raise HTTPException(status_code=400, detail="Blood glucose level must be between 50 and 400")
        
        # Validate categorical inputs
        valid_genders = ['male', 'female']
        if request.gender.lower() not in valid_genders:
            raise HTTPException(status_code=400, detail="Gender must be 'Male' or 'Female'")
        
        valid_smoking = ['never', 'no info', 'current', 'former', 'ever', 'not current']
        if request.smoking_history.lower() not in valid_smoking:
            raise HTTPException(status_code=400, detail=f"Invalid smoking history. Must be one of: {valid_smoking}")
        
        # Make prediction
        probability = diabetes_predictor.predict_diabetes_probability(
            gender=request.gender,
            age=request.age,
            hypertension=request.hypertension,
            heart_disease=request.heart_disease,
            smoking_history=request.smoking_history,
            bmi=request.bmi,
            HbA1c_level=request.HbA1c_level,
            blood_glucose_level=request.blood_glucose_level
        )
        
        # Calculate percentage
        percentage = probability * 100
        
        # Get risk categorization
        risk_level, risk_color, recommendation = preprocessor.categorize_risk(probability)
        
        return DiabetesPredictionResponse(
            probability=round(probability, 4),
            percentage=round(percentage, 2),
            risk_level=risk_level,
            risk_color=risk_color,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded models"""
    return {
        "model_type": "ensemble" if diabetes_predictor.is_ensemble else "single",
        "models_loaded": list(diabetes_predictor.models.keys()),
        "model_status": "trained" if diabetes_predictor.model_loaded else "default",
        "features_used": [
            "gender", "age", "hypertension", "heart_disease", 
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ],
        "risk_categories": {
            "low": "0-20% probability",
            "moderate": "20-50% probability", 
            "high": "50-80% probability",
            "very_high": "80-100% probability"
        }
    }

@app.post("/batch-predict")
async def batch_predict(requests: list[DiabetesPredictionRequest]):
    """
    Batch prediction for multiple patients
    """
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    results = []
    for i, request in enumerate(requests):
        try:
            probability = diabetes_predictor.predict_diabetes_probability(
                gender=request.gender,
                age=request.age,
                hypertension=request.hypertension,
                heart_disease=request.heart_disease,
                smoking_history=request.smoking_history,
                bmi=request.bmi,
                HbA1c_level=request.HbA1c_level,
                blood_glucose_level=request.blood_glucose_level
            )
            
            percentage = probability * 100
            risk_level, risk_color, recommendation = preprocessor.categorize_risk(probability)
            
            results.append({
                "index": i,
                "probability": round(probability, 4),
                "percentage": round(percentage, 2),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "recommendation": recommendation
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })
    
    return {"predictions": results}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
