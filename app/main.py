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
        # Scale input
        scaled_input = scaler.transform([[input.n, input.p, input.k, input.ph]])
        
        # Predict
        prediction = model.predict(scaled_input)[0]
        crop_name = encoder.inverse_transform([prediction])[0]
        
        # Get probabilities (if available)
        try:
            proba = model.predict_proba(scaled_input)[0]
            confidence = float(np.max(proba))
        except:
            confidence = None
            
        return {
            "crop": crop_name,
            "confidence": confidence,
            "error": None
        }
        
    except Exception as e:
        return {"crop": None, "error": str(e)}

@app.get("/")
def health_check():
    return {"status": "API is running"}
