from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get the absolute path to the directory containing this script
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Verify models directory exists
if not MODELS_DIR.exists():
    logger.error(f"Models directory not found at: {MODELS_DIR}")
    raise RuntimeError("Models directory not found")

# Define model paths
MODEL_PATHS = {
    "model": MODELS_DIR / "sustainable_crop_model.pkl",
    "scaler": MODELS_DIR / "sustainable_crop_scaler.pkl",
    "encoder": MODELS_DIR / "sustainable_crop_label_encoder.pkl"
}

# Verify all model files exist
for name, path in MODEL_PATHS.items():
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Required model file not found: {path}")

# Load ML models
try:
    logger.info("Loading ML models...")
    with open(MODEL_PATHS["model"], "rb") as f:
        model = pickle.load(f)
    
    with open(MODEL_PATHS["scaler"], "rb") as f:
        scaler = pickle.load(f)
    
    with open(MODEL_PATHS["encoder"], "rb") as f:
        encoder = pickle.load(f)
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise RuntimeError("Failed to load ML models") from e

# Input model matching your Flutter app's request format
class SoilInput(BaseModel):
    n_effect: float
    p_effect: float
    k_effect: float
    ph_effect: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "n_effect": -68,
                "p_effect": -99,
                "k_effect": -17,
                "ph_effect": -0.17
            }]
        }
    }

@app.post("/predict")
async def predict(input: SoilInput):
    try:
        logger.info(f"Received prediction request: {input}")
        
        # 1. Prepare input array with the exact feature order expected by your model
        input_array = np.array([[
            input.n_effect,
            input.p_effect,
            input.k_effect,
            input.ph_effect
        ]])
        
        # 2. Scale the input features
        scaled_input = scaler.transform(input_array)
        
        # 3. Get probabilities for all crops
        probabilities = model.predict_proba(scaled_input)[0]
        
        # 4. Get indices of top 3 crops (sorted descending by probability)
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # 5. Prepare the response
        results = []
        for idx in top_3_indices:
            crop_name = encoder.inverse_transform([idx])[0]
            suitability = float(probabilities[idx] * 100)  # Convert to percentage
            
            results.append({
                "crop": str(crop_name),  # Ensure string conversion
                "suitability": round(suitability, 2)  # Round to 2 decimal places
            })
        
        logger.info(f"Prediction results: {results}")
        return {
            "status": "success",
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
        )

@app.get("/")
async def health_check():
    return {
        "status": "API is running",
        "models_loaded": all([
            model is not None,
            scaler is not None,
            encoder is not None
        ])
    }

@app.get("/model-info")
async def model_info():
    """Endpoint to verify model information"""
    return {
        "model_path": str(MODEL_PATHS["model"]),
        "model_type": type(model).__name__,
        "features_expected": getattr(model, "n_features_in_", "Unknown"),
        "classes_available": (encoder.classes_.tolist() 
                            if hasattr(encoder, "classes_") 
                            else "Unknown")
    }
