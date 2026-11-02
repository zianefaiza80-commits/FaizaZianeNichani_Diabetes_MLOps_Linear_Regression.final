import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient

def run_training():
    print("Starting training pipeline...")
    
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes-Regression")
    
    # 2. Load Data
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()
    
    # 3. Clean and Validate Data
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Train and Log Experiment
    model = LinearRegression()
    with mlflow.start_run(run_name="LinearRegression_Script") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3)
        )
        run_id = run.info.run_id
        
    print(f"Model trained successfully. Run ID: {run_id}")
    print(f"Metrics -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")
    
    # 6. Register Model and Promote to Champion
    model_name = "diabetes_model"
    model_uri = f"runs:/{run_id}/model"
    
    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    client = MlflowClient()
    client.set_registered_model_alias(
        model_name,
        "champion",
        registered_version.version
    )
    print(f"Model '{model_name}' version {registered_version.version} promoted to 'champion' alias.")

if __name__ == "__main__":
    run_training()
