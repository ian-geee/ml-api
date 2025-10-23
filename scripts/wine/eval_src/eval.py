# src/prep.py
from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from prefect import flow, task, get_run_logger
from prefect.assets import materialize

from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error
)


@task
def metrics_to_json(in_predictions_df_folder_path: Path, out_metrics_folder_path: Path):
    df_pred = pd.read_csv(in_predictions_df_folder_path / "predictions.csv", index_col=False)

    y_true = df_pred["target"].to_numpy()
    y_pred = df_pred["predicted_target"].to_numpy()

    # ----- Métriques de régression -----
    metrics = {
        "rmse": root_mean_squared_error(y_true, y_pred),  # RMSE
        "r2": r2_score(y_true, y_pred),                              # R²
        "mae": mean_absolute_error(y_true, y_pred),                  # MAE
    }

    # ----- Sauvegarde JSON -----
    out_path = out_metrics_folder_path / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return None


@flow
def eval_flow(data_input_dir: Path, metrics_output_dir: Path, run_id: str) -> None:
    """
    Flow d'évaluation pour régression
    """
    metrics_to_json(
        in_predictions_df_folder_path=data_input_dir, 
        out_metrics_folder_path=metrics_output_dir
    )
    return None


if __name__ == "__main__":
    # Lancement direct en local
    eval_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/wine/testruns/00000/data'),
        metrics_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/wine/testruns/00000/eval'),
        run_id="00000"
    )