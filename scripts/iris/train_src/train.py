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
from sklearn.model_selection import StratifiedKFold

import mlflow

RANDOM_STATE = 42

@task
def load_inputs():
    return None


@task
def split_dataset(in_df_to_split_folder_path: Path, out_splitted_dfs_folder_path: Path):
    df = pd.read_csv(in_df_to_split_folder_path / "raw_data.csv", index_col=False)
    # df = pd.read_csv(data_input_dir / "iris.csv", index_col=False)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    pd.concat([X_trainval, y_trainval], axis=1).to_csv(out_splitted_dfs_folder_path / "trainval.csv", index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(out_splitted_dfs_folder_path / "train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(out_splitted_dfs_folder_path / "val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(out_splitted_dfs_folder_path / "test.csv", index=False)

    return {
        "trainval": str((out_splitted_dfs_folder_path / "trainval.csv").resolve()),
        "train": str((out_splitted_dfs_folder_path / "train.csv").resolve()),
        "val": str((out_splitted_dfs_folder_path / "val.csv").resolve()),
        "test": str((out_splitted_dfs_folder_path / "test.csv").resolve()),
    }


@task
def make_folds(in_trainval_dataframe_folder_path: Path, out_json_folds_folder_path: Path, k: int = 5):
    df = pd.read_csv(in_trainval_dataframe_folder_path / "trainval.csv")
    y = df["target"].to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    out_json_folds_folder_path = (out_json_folds_folder_path / "trainval_folds")
    out_json_folds_folder_path.mkdir(parents=True, exist_ok=True)
    for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        with (out_json_folds_folder_path / f"fold_{i}.json").open("w") as f:
            json.dump({"train_idx": tr.tolist(), "val_idx": va.tolist()}, f, indent=2)



@task
def fit_model(in_train_dataframe_folder_path: Path, out_model_folder_path: Path, run_id: str):

    # Active l’autolog AVANT fit
    mlflow.sklearn.autolog()  # ou mlflow.autolog()
    mlflow.set_experiment("iris")

    with mlflow.start_run(run_name=run_id):
        df_train = pd.read_csv(in_train_dataframe_folder_path / "train.csv", index_col=False)
        X_train = df_train.drop(columns=["target"])
        y_train = df_train["target"]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=1000))
            ])
        pipe.fit(X_train, y_train)
    


    joblib.dump(pipe, out_model_folder_path / "iris_model.joblib")
    return pipe


@flow
def train_flow(data_input_dir: Path, data_output_dir: Path, model_output_dir: Path, run_id: str) -> None:
    """
    
    """
    split_dataset(in_df_to_split_folder_path=data_input_dir, out_splitted_dfs_folder_path=data_output_dir)
    make_folds(in_trainval_dataframe_folder_path=data_input_dir, out_json_folds_folder_path=data_input_dir, k=5)
    fit_model(in_train_dataframe_folder_path=data_input_dir, out_model_folder_path=model_output_dir, run_id=run_id)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    train_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/data'),
        data_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/data'),
        model_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/models'),
        run_id="2025-09-20_23h4145-d6fc14b1"
        )