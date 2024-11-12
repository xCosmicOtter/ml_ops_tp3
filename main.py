import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()
mlflow.set_tracking_uri("http://127.0.0.1:8080") 
MODEL_URI = "models:/tracking-quickstart/latest"
model = mlflow.pyfunc.load_model(MODEL_URI)

class PredictionRequest(BaseModel):
    data: List[List[float]]

class UpdateModelRequest(BaseModel):
    version: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        data = np.array(request.data)
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-model")
async def update_model(request: UpdateModelRequest):
    global model
    try:
        new_model_uri = f"models:/tracking-quickstart/{request.version}"
        model = mlflow.pyfunc.load_model(new_model_uri)
        return {"status": "Model updated successfully", "version": request.version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))