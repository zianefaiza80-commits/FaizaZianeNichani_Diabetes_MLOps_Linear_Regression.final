import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "Diabetes-Regression"
MODEL_NAME = "diabetes_model"
MODEL_ALIAS = "champion"


def main() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    model = LinearRegression()

    with mlflow.start_run(run_name="LinearRegression") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        mlflow.log_params({"model_type": "LinearRegression", "test_size": 0.2, "random_state": 42})
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(3))
        model_uri = f"runs:/{run.info.run_id}/model"

    version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version.version)

    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
    print(f"Registered {MODEL_NAME} version {version.version} as @{MODEL_ALIAS}")


if __name__ == "__main__":
    main()
