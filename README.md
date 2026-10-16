## Diabetes MLOps Project with Linear Regression

## Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow using the Diabetes dataset included with scikit-learn. It trains a Linear Regression model to estimate a quantitative measure of disease progression from ten numerical patient features.

The project covers data preparation, model training and evaluation, experiment tracking and model registration with MLflow, prediction serving with FastAPI, automated tests, and continuous integration with GitHub Actions.

## Project Objectives

- Load and validate the scikit-learn Diabetes dataset.
- Train and evaluate a Linear Regression model.
- Track experiment parameters and metrics with MLflow.
- Register the trained model and assign it the `champion` alias.
- Serve predictions through a REST API built with FastAPI.
- Validate API inputs and handle unavailable models.
- Run automated tests with pytest.
- Run tests automatically with GitHub Actions.

## Dataset

The project uses `sklearn.datasets.load_diabetes`, so no external dataset download is required. The dataset contains 442 samples, ten baseline patient features, and a numeric target representing disease progression.

The features are `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, and `s6`. The feature values are preprocessed/scaled by scikit-learn.

## Model and Evaluation

The selected model is scikit-learn's `LinearRegression`. The training script uses an 80/20 train/test split with `random_state=42` and logs these evaluation metrics to MLflow:

- **RMSE (Root Mean Squared Error):** average prediction error in the target's units; lower is better.
- **MAE (Mean Absolute Error):** average absolute prediction error; lower is better.
- **R² (Coefficient of Determination):** indicates how much target variation is explained by the model. It can be negative; values closer to 1 generally indicate a better fit.

The exact metric values are produced when `train.py` is run.

## MLOps Workflow

1. Load the Diabetes dataset from scikit-learn.
2. Split the data into training and test sets.
3. Train the Linear Regression model.
4. Evaluate it with RMSE, MAE, and R².
5. Track the run and model with MLflow using a local SQLite tracking database.
6. Register the model as `diabetes_model` and set the `champion` alias.
7. Load the registered model in the FastAPI service.
8. Validate requests and return predictions through the API.
9. Run tests locally and automatically through GitHub Actions.

## Repository Structure

```text
Diabetes_MLOps_Linear_Regression/
├── .github/
│   └── workflows/
│       └── tests.yml                 # GitHub Actions CI workflow
├── api.py                            # FastAPI application
├── train.py                          # Train, evaluate, and register the model
├── test_api.py                       # API tests
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── FaizaZianeNichani_Diabetes_MLOps_Linear_Regression.final.ipynb
                                      # Original project notebook
```

The MLflow database and model artifacts are generated locally during training and are not included in the repository.

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- MLflow
- FastAPI and Pydantic
- pytest and HTTPX
- GitHub Actions

## Installation

Use Python 3.10 or newer. From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and Register the Model

Run the training script from the project directory:

```bash
python train.py
```

This command trains and evaluates the model, creates the local `mlflow.db` tracking database, logs the run and model artifacts, registers the model as `diabetes_model`, and assigns the `champion` alias.

Run training before starting the API. If no registered model is available, the API reports that the model is unavailable.

## Start the API

Start the development server from the project directory:

```bash
uvicorn api:app --reload
```

The interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Checks whether the model is available. |
| `GET` | `/model-info` | Returns the model name, alias, and model type. |
| `POST` | `/predict` | Validates ten feature values and returns a prediction. |

Example request to `POST /predict`:

```json
{
  "age": 0.04,
  "sex": 0.05,
  "bmi": 0.06,
  "bp": 0.07,
  "s1": 0.08,
  "s2": 0.09,
  "s3": 0.10,
  "s4": 0.11,
  "s5": 0.12,
  "s6": 0.13
}
```

All ten values are required and must be finite numbers. Invalid requests receive a validation error. If the model cannot be loaded, the service returns HTTP 503 for model-dependent endpoints.

## Run the Tests

With the dependencies installed, run:

```bash
pytest -v
```

The API tests cover a successful prediction, invalid input, and the unavailable-model health response. Tests use a fake model for the successful prediction case and do not require a previously trained model.

## Continuous Integration

GitHub Actions runs the tests on pushes and pull requests. The workflow file must be located at:

```text
.github/workflows/tests.yml
```

GitHub does not automatically discover workflow files stored only in the repository root.

## MLflow Files and Local Artifacts

Training creates the MLflow tracking database and model artifacts in the local project environment. These generated files are excluded from version control. Run `python train.py` again in a fresh checkout to create a new local model before starting the API.

## Limitations and Future Improvements

- Compare Linear Regression with regularized and non-linear models.
- Use cross-validation to obtain more robust performance estimates.
- Add model quality thresholds before assigning the `champion` alias.
- Add API integration tests for every endpoint and error case.
- Configure persistent or remote MLflow tracking and artifact storage for deployment.
- Add monitoring for prediction latency, data drift, and model performance.
- Deploy the API to a production hosting platform with appropriate security controls.

## Conclusion

This educational project demonstrates a basic MLOps workflow from dataset loading and model evaluation through MLflow registration, API serving, automated tests, and continuous integration. Its predictions are estimates of a benchmark progression target, not clinical diagnoses.
