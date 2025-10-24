# src/prep.py
from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from prefect import flow, task, get_run_logger
from prefect.assets import materialize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



@task
def predict(in_model_folder_path: Path, in_test_df_folder_path: Path, out_predictions_df_folder_path: Path):
    pipe = joblib.load(in_model_folder_path / "iris_model.joblib")

    df_test = pd.read_csv(in_test_df_folder_path / "test.csv", index_col=False)
    X_test = df_test.drop(columns=["target"])

    # METADONNEES 
    with open(in_model_folder_path / "data_model_metadata.json", "r") as f:
        model_metadata = json.load(f)

    data_metadata = {
        **model_metadata,
        "n_samples_test": len(X_test)
    }

    with open(in_model_folder_path / "data_model_metadata.json", "w") as f:
        json.dump(data_metadata, f, indent=2)

    y_pred = pipe.predict(X_test)

    # Construit le DF de sortie
    out = df_test.copy()
    out["predicted_target"] = y_pred

    # Saves probabilities if available, for roc, auc, log-loss, top-k
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)  # shape (n_samples, n_classes)
        # Tente de récupérer les noms de classes du modèle (sinon indices)
        try:
            class_labels = pipe.classes_
        except Exception:
            class_labels = list(range(proba.shape[1]))
        for j, cls in enumerate(class_labels):
            out[f"proba_{cls}"] = proba[:, j]

    out_path = out_predictions_df_folder_path / "predictions.csv"
    out.to_csv(out_path, index=False)
    return None


@flow
def score_flow(data_input_dir: Path, data_output_dir: Path, model_input_dir: Path, run_id: str) -> None:
    """
    
    """
    predict(in_model_folder_path=model_input_dir, in_test_df_folder_path=data_input_dir, out_predictions_df_folder_path=data_output_dir)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    score_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/data'),
        model_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/models'),
        data_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/data'),
        run_id="2025-09-20_23h4145-d6fc14b1"
        )
