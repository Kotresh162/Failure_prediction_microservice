from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("failure_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)  
    prediction = model.predict(features_scaled)
    return {"failure_prediction": int(prediction[0])}

# 🔹 Run with: `uvicorn server:app --reload`
