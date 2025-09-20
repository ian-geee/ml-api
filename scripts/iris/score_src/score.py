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
    df_test_with_predictions = df_test.copy()
    X_test = df_test.drop(columns=["target"])
    y_pred = pipe.predict(X_test)

    df_test_with_predictions["predicted_target"] = y_pred
    df_test_with_predictions.to_csv(out_predictions_df_folder_path / "predictions.csv", index=False)
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
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/runs/9d2143d3-fe1a-450b-9728-e0c3a30c373c/data'),
        model_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/runs/9d2143d3-fe1a-450b-9728-e0c3a30c373c/models'),
        data_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/runs/9d2143d3-fe1a-450b-9728-e0c3a30c373c/data'),
        )
