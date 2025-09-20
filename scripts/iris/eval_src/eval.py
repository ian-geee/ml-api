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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@task
def metrics_to_json(in_predictions_df_folder_path: Path, out_metrics_folder_path: Path):
    df_pred = pd.read_csv(in_predictions_df_folder_path / "predictions.csv", index_col=False)
    y_test = df_pred["target"]
    y_pred = df_pred["predicted_target"]
    acc = accuracy_score(y_test, y_pred)
    with open(out_metrics_folder_path / "metrics.json", "w", encoding='utf-8') as f:
        json.dump({"accuracy": 0.95}, f, ensure_ascii=False, indent=2)
    return None


@flow
def eval_flow(data_input_dir: Path, metrics_output_dir: Path) -> None:
    """
    
    """
    metrics_to_json(in_predictions_df_folder_path=data_input_dir, out_metrics_folder_path=metrics_output_dir)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    eval_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/runs/9d2143d3-fe1a-450b-9728-e0c3a30c373c/data'),
        metrics_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/runs/9d2143d3-fe1a-450b-9728-e0c3a30c373c/eval'),
        )
