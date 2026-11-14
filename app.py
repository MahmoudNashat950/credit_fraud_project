
import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
with open("model.pkl", "rb") as file:
    artifacts = pickle.load(file)

model = artifacts["model"]
scaler = artifacts["scaler"]
threshold = artifacts["best_threshold"]

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)


class Transaction(BaseModel):
    V17: float
    V14: float
    V12: float
    V10: float
    V16: float
    V7: float
    V3: float
    V11: float
    V4: float
    V18: float


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):

    features = pd.DataFrame([{
        "V17": transaction.V17,
        "V14": transaction.V14,
        "V12": transaction.V12,
        "V10": transaction.V10,
        "V16": transaction.V16,
        "V7": transaction.V7,
        "V3": transaction.V3,
        "V11": transaction.V11,
        "V4": transaction.V4,
        "V18": transaction.V18
    }])

    scaled_features = scaler.transform(features)

    probability = float(
        model.predict_proba(scaled_features)[0][1]
    )

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(probability, 4),
        "threshold": threshold
    }
