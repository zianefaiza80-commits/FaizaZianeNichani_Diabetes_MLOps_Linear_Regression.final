# Diabetes dataset with MLOps and Linear Regression

Project Overview

This project implements a simple end-to-end Machine Learning Operations (MLOps) workflow using the Diabetes dataset from scikit-learn.

The goal is to predict diabetes disease progression using patient-related numerical features and a Linear Regression model.

The project demonstrates data preparation, model training, evaluation, experiment tracking, model registration, API serving, automated testing, and continuous integration.

Project Objectives:

Load and validate the dataset.
Clean the data and handle missing or invalid values.
Train a Linear Regression model.
Evaluate model performance using standard regression metrics.
Track experiments using MLflow.
Register the trained model and use the "champion" alias.
Create a REST API using FastAPI.
Test API endpoints and error handling.
Automate tests with GitHub Actions.

Dataset
The project uses the Diabetes dataset provided by scikit-learn.

The dataset contains numerical patient-related features and a target representing a quantitative measure of disease progression.

The model predicts the target value. It does not diagnose diabetes or replace medical advice.

Model and Evaluation
Selected model: Linear Regression.

The model is evaluated using:

Mean Squared Error (MSE): measures the average squared prediction error. Lower values are better.
Root Mean Squared Error (RMSE): measures prediction error in the target's units. Lower values are better.
R² Score: measures how well the model explains variation in the target. Values closer to 1 generally indicate a better fit.
Actual evaluation results are generated when the notebook is executed.

MLOps Workflow

The project follows these steps:

Data extraction and validation.

Data cleaning and preparation.

Train-test split.

Linear Regression model training.

Model evaluation.

Experiment tracking with MLflow.

Model registration and "champion" alias assignment.

Model loading for prediction.

REST API development with FastAPI.

Automated testing with pytest.

Continuous integration with GitHub Actions.

API Endpoints

The API provides the following endpoints:

Method| Endpoint| Description GET| "/health"| Checks service health and model availability. GET| "/model-info"| Returns information about the loaded model. POST| "/predict"| Receives numerical input features and returns a prediction.

The prediction endpoint expects the ten numerical features used by the scikit-learn Diabetes dataset.

Error Handling and Testing

The project includes tests for:

Successful prediction requests.
Invalid input data.
Model or service unavailability.
Tests are implemented using pytest and FastAPI's TestClient.

Technologies Used

Python
pandas
NumPy
scikit-learn
MLflow
FastAPI
Pydantic
pytest
Git and GitHub
GitHub Actions

Repository Structure

diabetes-mlops/ ├── Proyecto_MLOps_Diabetes_Regresion_Lineal.ipynb ├── Diabetes_MLOps_Linear_Regression.ipynb ├── diabetes.csv ├── api.py ├── test_api.py ├── requirements.txt ├── tests.yml └── README.md

The exact filenames and locations should match the files committed to the repository.

Installation and Execution
Install the project dependencies:

pip install -r requirements.txt

Run the notebook in Google Colab, executing the cells in order to prepare the data, train and evaluate the model, and register it with MLflow.

After the model has been registered and the required MLflow tracking environment is available, start the API with:

uvicorn api:app --reload

Open the interactive API documentation at:

"http://127.0.0.1:8000/docs"

Run the automated tests with:

pytest -v

These commands assume the required files, dependencies, and MLflow configuration are available in the current environment.

Continuous Integration

GitHub Actions is used to automate testing when changes are pushed to the repository or a pull request is opened.

The workflow installs the dependencies and runs the automated tests to help identify errors early.

The workflow file must be located inside ".github/workflows/" for GitHub Actions to detect it automatically.

Limitations and Future Improvements

Possible improvements include:

Adding more robust data validation.
Comparing Linear Regression with other regression models.
Improving experiment and model version management.
Persisting MLflow tracking data and model artifacts.
Deploying the API to a hosting platform.
Adding monitoring for prediction latency and model performance.

Conclusion

This project demonstrates the main concepts of an end-to-end MLOps workflow, from data preparation and model training to experiment tracking, API serving, testing, and continuous integration.

It provides an educational foundation for understanding how a machine learning model can be organized and served as part of a software project.

Project type: Educational MLOps project Model: Linear Regression Dataset: scikit-learn Diabetes dataset


#########################################################################################################################################################

