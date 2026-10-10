# Diabetes MLOps: regresión lineal

Proyecto educativo que entrena una regresión lineal con el conjunto Diabetes integrado en scikit-learn, registra métricas y el modelo en MLflow, y expone predicciones mediante FastAPI. Predice una medida cuantitativa de progresión de la enfermedad; no diagnostica diabetes.

## Requisitos

Python 3.10 o posterior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Entrenar y ejecutar

Desde esta carpeta, ejecuta el entrenamiento. Crea `mlflow.db`, registra el modelo `diabetes_model` y asigna el alias `champion`:

```bash
python train.py
```

Después inicia la API:

```bash
uvicorn api:app --reload
```

Documentación interactiva: <http://127.0.0.1:8000/docs>. La API acepta diez valores numéricos (`age`, `sex`, `bmi`, `bp`, `s1`…`s6`) en `POST /predict`. `GET /health` comprueba la disponibilidad y `GET /model-info` informa del modelo. Debes ejecutar `train.py` antes de iniciar la API.

## Pruebas

```bash
pytest -v
```

GitHub Actions ejecuta las pruebas en cada push y pull request desde `.github/workflows/tests.yml`.

## Archivos

- `train.py`: entrenamiento, métricas y registro en MLflow.
- `api.py`: servicio FastAPI y validación de entradas.
- `test_api.py`: pruebas automatizadas.
- `FaizaZianeNichani_Diabetes_MLOps_Linear_Regression.ipynb`: cuaderno original del proyecto.
- `.github/workflows/tests.yml`: integración continua.

La base MLflow, los modelos generados y los entornos virtuales se crean localmente y no se incluyen en el proyecto.
