# Train a Wine Quality model (GradientBoostingRegressor) and save artifacts into app/models
# - Uses merged dataset: data/wine_quality_merged.csv (columns lower_snake_case)
# - Optionally filters to red wines only (type == "red"), as in the notebook
# - Mirrors the structure of train_house.py (paths, pipeline, save joblib + meta JSON)
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# --- Paths (assume repo layout: <root>/data and <root>/app/models) ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "winequality_only_key_features.csv"   # expects a column 'type' with values like 'red'/'white'
MODELS = ROOT / "app" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# --- Config ---
TARGET = "quality"
# numeric features from the notebook (order matters for feature_importances_ mapping)
FEATURES_NUM = [
    "volatile_acidity",
    "total_sulfur_dioxide",
    "sulphates",
    "alcohol",
]
# set to "red" to match the notebook focus, or None to train on all
FILTER_WINE_TYPE = "red"   # or None

# columns on which to remove outliers with IQR bounds (k=2)
OUTLIER_COLS = FEATURES_NUM
IQR_K = 2.0

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\\n"
                                f"Expected a merged CSV with a 'type' column and the standard wine features.")
    df = pd.read_csv(path, index_col=False)
    # normalize column names as in the notebook
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    # basic sanitization
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower()
    # ensure numerics
    for c in FEATURES_NUM + [TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def remove_outliers_iqr_2bounds(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
    ])
    model = GradientBoostingRegressor(random_state=42)
    return Pipeline([("prep", preprocess), ("regressor", model)])

def fit_with_grid(pipe: Pipeline, X_train, y_train) -> GridSearchCV:
    # grid as in the notebook
    param_grid = {
        "regressor": [GradientBoostingRegressor(random_state=42)],
        # 1) dynamique
        "regressor__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
        "regressor__n_estimators": [50, 100, 200],
        # 2) forme
        "regressor__max_depth": [2, 3, 4],
        "regressor__min_samples_leaf": [1, 5, 10],
        "regressor__max_features": [None, "sqrt", "log2"],
        # 3) régularisation stochastique
        "regressor__subsample": [0.6, 0.8, 1.0],
        # 5) early stopping
        "regressor__n_iter_no_change": [None, 5],
        "regressor__validation_fraction": [0.1],
    }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid



if __name__ == "__main__":
    print(f"Loading {DATA} ...")
    df = load_data(DATA)
    print("Initial shape:", df.shape)

    # optional filter to red wines (as in the notebook)
    if FILTER_WINE_TYPE is not None and "type" in df.columns:
        before = df.shape[0]
        df = df[df["type"] == FILTER_WINE_TYPE].copy()
        print(f"Filtered to type == '{FILTER_WINE_TYPE}': {before} -> {df.shape[0]} rows")

    # outlier removal
    df_no_out = remove_outliers_iqr_2bounds(df, OUTLIER_COLS)
    print("After outlier removal:", df_no_out.shape)

    # check required columns
    missing = [c for c in FEATURES_NUM + [TARGET] if c not in df_no_out.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # split
    X = df_no_out[FEATURES_NUM].copy()
    y = df_no_out[TARGET].astype(float).copy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Shapes:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    # pipeline + grid
    pipe = build_pipeline()
    print("Training (GridSearchCV) ...")
    grid = fit_with_grid(pipe, X_train, y_train)

    # eval
    y_pred = grid.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    print(f"Validation RMSE: {rmse:.4f}")

    # best pipeline
    best_pipe = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # save artifacts
    model_path = MODELS / "wine_model.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "target": TARGET,
        "feature_order": FEATURES_NUM,
        "wine_type": FILTER_WINE_TYPE,
        "outlier_method": f"IQR k={IQR_K} both sides",
        "grid_params": {
            "regressor__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.3],
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [2, 3, 4],
            "regressor__min_samples_leaf": [1, 5, 10],
            "regressor__max_features": [None, "sqrt", "log2"],
            "regressor__subsample": [0.6, 0.8, 1.0],
            "regressor__n_iter_no_change": [None, 5],
            "regressor__validation_fraction": [0.1],
        },
        "best_params": grid.best_params_,
        "metric": "rmse",
        "val_rmse": rmse,
        "model_version": "wine-gbr-v1",
    }
    meta_path = MODELS / "wine_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("Saved:")
    print(" -", model_path)
    print(" -", meta_path)
