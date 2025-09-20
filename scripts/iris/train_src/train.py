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
def load_inputs():
    return None


@task
def split_dataset(data_input_dir: Path, data_output_dir: Path):
    df = pd.read_csv(r"C:/Users/joule/work_repos/ml-api/scripts/iris/data/iris.csv", index_col=False)
    # df = pd.read_csv(data_input_dir / "iris.csv", index_col=False)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    pd.concat([X_train, y_train], axis=1).to_csv(data_output_dir / "train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(data_output_dir / "val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(data_output_dir / "test.csv", index=False)

    return {
        "train": str((data_output_dir / "train.csv").resolve()),
        "val": str((data_output_dir / "val.csv").resolve()),
        "test": str((data_output_dir / "test.csv").resolve()),
    }


@task
def fit_model(data_input_dir: str, model_output_dir: str):
    df_train = pd.read_csv(data_input_dir + "/train.csv", index_col=False)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000))
        ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, model_output_dir / "iris_model.joblib")
    return pipe


@flow
def train_flow(data_input_dir: str, data_output_dir: str, model_output_dir: str) -> None:
    """
    
    """
    split_dataset(data_input_dir=Path("../data"), data_output_dir=data_output_dir)
    fit_model(data_input_dir=data_input_dir, model_output_dir=model_output_dir)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    train_flow()
