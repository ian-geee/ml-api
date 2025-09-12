# Train a House Price model (Belgium) with only 4 inputs and save artifacts into app/models
# - Filters to type_of_property == 'house'
# - Uses features: habitable_surface, bedrooms_count, post_code, num_facade
from pathlib import Path
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# --- MLflow ---
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# --- Paths (assume repo layout: <root>/data and <root>/app/models) ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "immoweb2022_houses-only_4features_no-outliers.csv"   # put the filtered CSV here
MODELS = ROOT / "app" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES_NUM = ["habitable_surface", "bedrooms_count", "num_facade"]
FEATURES_CAT = ["post_code"]
FEATURE_ORDER = FEATURES_NUM + FEATURES_CAT
TARGET = "price"

EXPERIMENT_NAME = "house_price_be_4features"
MODEL_ARTIFACT_PATH = "model"          # dossier d'artifact MLflow pour le modèle
DEFAULT_TRACKING_URI = "file:./mlruns" # local par défaut (aucun serveur requis)

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Place houses_4feat.csv under {path.parent}.\n"
            f"You can build it from all_entriess.csv by filtering type_of_property == 'house' and keeping the 4 columns + price."
        )
    df = pd.read_csv(path)
    # safety: coerce post_code to string for OHE, ensure numeric types for nums
    df["post_code"] = df["post_code"].astype(str).str.strip()
    for c in FEATURES_NUM + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, FEATURES_NUM),
        ("cat", cat_pipe, FEATURES_CAT),
    ])
    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=600,
        validation_fraction=0.1, random_state=42
    )
    return Pipeline([("prep", preprocess), ("model", model)])

def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test_log: np.ndarray) -> float:
    # RMSE in original € scale
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    rmse = root_mean_squared_error(np.expm1(y_test_log), y_pred)
    return rmse

def save_artifacts(pipe: Pipeline, rmse: float):
    # Save joblib + metadata exactly as before
    joblib.dump(pipe, MODELS / "house_model.joblib")
    meta = {
        "feature_order": FEATURE_ORDER,
        "categorical": FEATURES_CAT,
        "target": TARGET,
        "transform": "log1p",
        "currency": "EUR",
        "model_version": "house-4feat-v1",
        "rmse_eur": rmse,
    }
    with open(MODELS / "house_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved app/models/house_model.joblib & house_meta.json")

def log_to_mlflow(pipe: Pipeline, X_sample: pd.DataFrame, yhat_sample: np.ndarray, rmse: float, n_rows: int):
    # Paramètres principaux + métriques
    mlflow.log_param("algorithm", "HistGradientBoostingRegressor")
    mlflow.log_param("features_num", FEATURES_NUM)
    mlflow.log_param("features_cat", FEATURES_CAT)
    mlflow.log_param("dataset_rows", n_rows)
    mlflow.log_metric("rmse_eur", rmse)

    # Signature d'IO (schéma) + exemple d'entrée
    signature = infer_signature(X_sample, yhat_sample)
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path=MODEL_ARTIFACT_PATH,
        signature=signature,
        input_example=X_sample.head(1)
        # Note: pas de registered_model_name ici car le Model Registry nécessite un backend DB.
    )

    # Logger aussi les artefacts sauvegardés (joblib + meta)
    mlflow.log_artifacts(str(MODELS))

def run():
    setup_mlflow()

    print(f"Loading {DATA} ...")
    df = load_data(DATA)

    missing = [c for c in FEATURE_ORDER + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_ORDER].copy()
    y = df[TARGET].copy()

    # Drop rows without target
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]

    # Log1p transform for stability (prices are skewed)
    y_log = np.log1p(y.clip(lower=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(X_train)

    print("Training ...")
    with mlflow.start_run(run_name="train-house-price-4feat"):
        pipe.fit(X_train, y_train)

        # Debug prints identiques à ta version
        print(X_test.head())
        print(X_test["post_code"].dtype)

        rmse = evaluate(pipe, X_test, y_test)
        print(f"RMSE ≈ €{rmse:,.0f}")

        # Sauvegarde locale inchangée
        save_artifacts(pipe, rmse)

        # Log MLflow (modèle + metrics + artefacts)
        yhat_sample = pipe.predict(X_train.head(5))
        log_to_mlflow(pipe, X_train.head(5), yhat_sample, rmse, n_rows=len(df))

def main():
    run()

if __name__ == "__main__":
    main()
