# ML-API 

Contains multiple projects :
- Binary Classification : In construction
- Mutliclass Classification : Iris Classifier
- Regression : House Price Estimator
- Regression : Wine Quality Estimator
- Mutlilabel Classification : TO DO (probably ANN (CNN) - object detection)
- TimeSeries Forecasting : TO DO
- LLM : TO DO

Small project showing MLOps skills : API FastAPI + HTML page to **predict** flower type based on quantitative characteristics (open dataset Iris).

## 0) Setting up the virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Training the modeld
```bash
python .\scripts\train_model.py
```

## 1b) Check training quality with MLFlow
```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

## 2) Launching the API 
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  # app.main et pas app/main....
```

Quick tests :
```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

## 3) If local use : 
- Open `public/index.html` with Live Server (has to be on 5500)
- Change `API_URL` if the API is on another machine.

## 4) Docker (don't forget the port mapping)
```bash
docker build -t iris-api .
docker run -e ML_API_KEY=secret_key -p 8000:8000 iris-api
```

## 5) Deployment
- As of 2025-08-29, the API is hosted on Render : https://ml-api-nbek.onrender.com

---
 Thanks for reading through !


 Update

## pyproject.toml :
Required for installing utils scripts and be able to use them with any Python.exe. Otherwise, if we just include utils, we need to launch Python from the root of the folder, so that python can find utils with 'from utils import xxx'

### => Need to install from utils with : pip install -e .

## Choice between Airflown, Prefect & Flyte for steps pipelining in the inner_loop : 
Airflow & Flyte being unix-based + Airflow being very "industrial" (hard to manage on solo), I go with Prefect

## Architecture :
- Prefect: Automation Pipeline to chain scripts to train models
- DVC: Saves outputs of every script and versions them
- MLFlow : Saves metrics inside the training/scoring/eval
- Scikit-learn pipeline : allows our future predictions to be treated with the same preprocessing/feature engineering as our training/validation set.