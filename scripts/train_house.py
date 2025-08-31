# Train a House Price model (Belgium) with only 4 inputs and save artifacts into app/models
# - Filters to type_of_property == 'house'
# - Uses features: habitable_surface, bedrooms_count, post_code, num_facade
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# --- Paths (assume repo layout: <root>/data and <root>/app/models) ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "immoweb2022_houses-only_4features_no-outliers.csv"   # put the filtered CSV here
MODELS = ROOT / "app" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES_NUM = ["habitable_surface", "bedrooms_count", "num_facade"]
FEATURES_CAT = ["post_code"]
FEATURE_ORDER = FEATURES_NUM + FEATURES_CAT
TARGET = "price"

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\n"
                                f"Place houses_4feat.csv under {path.parent}.\n"
                                f"You can build it from all_entriess.csv by filtering type_of_property == 'house' and keeping the 4 columns + price.")
    df = pd.read_csv(path)
    # safety: coerce post_code to string for OHE, ensure numeric types for nums
    df["post_code"] = df["post_code"].astype(str).str.strip()
    for c in FEATURES_NUM + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if __name__ == "__main__":
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

    # Preprocessing
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, FEATURES_NUM),
        ("cat", cat_pipe, FEATURES_CAT),
    ])

    # Model (fast + strong for tabular)
    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=600,
        validation_fraction=0.1, random_state=42
    )

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    print("Training ...")
    pipe.fit(X_train, y_train)

    # Eval (RMSE in original € scale)
    y_pred_log = pipe.predict(X_test)
    print(X_test.head())
    print(X_test["post_code"].dtype)
    y_pred = np.expm1(y_pred_log)
    rmse = root_mean_squared_error(np.expm1(y_test), y_pred)
    print(f"RMSE ≈ €{rmse:,.0f}")

    # Save artifacts
    joblib.dump(pipe, MODELS / "house_model.joblib")
    meta = {
        "feature_order": FEATURE_ORDER,
        "categorical": FEATURES_CAT,
        "target": TARGET,
        "transform": "log1p",
        "currency": "EUR",
        "model_version": "house-4feat-v1"
    }
    with open(MODELS / "house_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved app/models/house_model.joblib & house_meta.json")
