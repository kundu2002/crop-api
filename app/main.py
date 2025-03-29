from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Flutter app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"]
)

# Load all three models
with open("models/sustainable_crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/sustainable_crop_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/sustainable_crop_label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

class SoilInput(BaseModel):
    n: float
    p: float
    k: float
    ph: float

@app.post("/predict")
def predict(input: SoilInput):
    try:
        # 1. Prepare input array
        input_array = np.array([
            [input.n_effect, input.p_effect, input.k_effect, input.ph_effect]
        ])
        
        # 2. Scale features
        scaled_input = scaler.transform(input_array)
        
        # 3. Get probabilities for all classes
        probabilities = model.predict_proba(scaled_input)[0]
        
        # 4. Get top 3 crop indices (sorted descending)
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # 5. Prepare response
        results = []
        for idx in top_3_indices:
            crop_name = encoder.inverse_transform([idx])[0]
            suitability = float(probabilities[idx] * 100)  # Convert to percentage
        
            results.append({
                "crop": crop_name,
                "suitability": round(suitability, 2)  # Rounds to 2 decimal places
            })
        
        return {"predictions": results}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "API is running"}
