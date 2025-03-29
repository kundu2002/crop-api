from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
)

# Input model matching your Flutter app's request format
class SoilInput(BaseModel):
    n_effect: float
    p_effect: float
    k_effect: float
    ph_effect: float

    class Config:
        # This will allow using field names like 'n_effect' in the JSON
        schema_extra = {
            "example": {
                "n_effect": -68,
                "p_effect": -99,
                "k_effect": -17,
                "ph_effect": -0.17
            }
        }

# Load your models (replace with your actual loading code)
with open("app/models/sustainable_crop_model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("app/models/sustainable_crop_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
with open("app/models/sustainable_crop_label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

@app.post("/predict")
def predict(input: SoilInput):
    try:
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
                "crop": crop_name,
                "suitability": round(suitability, 2)  # Round to 2 decimal places
            })
        
        return {
            "status": "success",
            "predictions": results  # This will be List<Map<String, dynamic>> in Flutter
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
def health_check():
    return {"status": "API is running"}
