import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")

app = FastAPI(title="Diabetes MLOps API")
MODEL_NAME = "diabetes_model"
MODEL_ALIAS = "champion"

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
except Exception as error:
    print(f"Could not load the MLflow model: {error}")
    model = None


class PatientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., allow_inf_nan=False)
    sex: float = Field(..., allow_inf_nan=False)
    bmi: float = Field(..., allow_inf_nan=False)
    bp: float = Field(..., allow_inf_nan=False)
    s1: float = Field(..., allow_inf_nan=False)
    s2: float = Field(..., allow_inf_nan=False)
    s3: float = Field(..., allow_inf_nan=False)
    s4: float = Field(..., allow_inf_nan=False)
    s5: float = Field(..., allow_inf_nan=False)
    s6: float = Field(..., allow_inf_nan=False)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="The model is unavailable; run train.py first")
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="The model is unavailable; run train.py first")
    return {"model_name": MODEL_NAME, "model_alias": MODEL_ALIAS, "model_type": "LinearRegression"}


@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="The model is unavailable; run train.py first")
    try:
        input_data = pd.DataFrame([data.model_dump()], columns=[
            "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
        ])
        prediction = float(model.predict(input_data)[0])
        return {"prediction": prediction, "model_name": MODEL_NAME, "model_alias": MODEL_ALIAS}
    except Exception as error:
        raise HTTPException(status_code=500, detail="Error while making the prediction") from error
