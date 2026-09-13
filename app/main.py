from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import model
from app.preprocessing import preprocess_features

app = FastAPI(title="Naive Bayes API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: Optional[List[str]] = Field(None, description="List of feature values")
    text: Optional[str] = Field(None, description="Email text to classify")

    @property
    def input_values(self):
        if self.text is not None:
            return self.text
        if self.features is not None:
            return self.features
        raise ValueError("provide either 'features' or 'text'")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model": "naive_bayes",
        "message": "Model service is ready",
        "service": "api",
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    try:
        processed = preprocess_features(request.input_values)
        prediction, probability = model.predict_label(processed)
        return {
            "success": True,
            "status": 200,
            "message": "Naive Bayes prediction successful",
            "data": {
                "model": "naive_bayes",
                "endpoint": "/api/v1/predict",
                "prediction": prediction,
                "probability": probability,
                "health_status": "healthy",
            },
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
