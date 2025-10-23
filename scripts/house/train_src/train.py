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
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from flaml import AutoML

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
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
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
def fit_model(in_train_dataframe_folder_path: Path, out_model_folder_path: Path, run_id: str):

    # Activer MLflow autolog
    # mlflow.sklearn.autolog() # NE PAS activer autolog avec FLAML !
    mlflow.set_experiment("regression_experiment")

    with mlflow.start_run(run_name=run_id):

        df_train = pd.read_csv(in_train_dataframe_folder_path / "trainval.csv", index_col=False)
        X_trainval = df_train.drop(columns=["target"])
        y_trainval = df_train["target"]

        y_trainval_log = np.log1p(y_trainval.clip(lower=0))

        # --- FLAML AutoML pour régression ---
        automl = AutoML()
        
        automl_settings = {
            "time_budget": 5,  # temps en secondes (ajustez selon vos besoins)
            "metric": "r2",  # ou "rmse", "mae", "mse"
            "task": "regression",  # ← IMPORTANT pour la régression
            "estimator_list": ["lgbm", "xgboost", "extra_tree"],  # modèles à tester
            "log_file_name": "flaml_regression.log",
            "seed": 42,
            "n_jobs": -1,
            "verbose": 1,

            # ← CLÉS POUR DÉSACTIVER LE LOGGING FLAML
            "log_training_metric": False,  # Pas de logging des métriques d'entraînement
            "mlflow_exp_name": None      # Désactive l'intégration MLflow de FLAML
        }

        # Entraînement automatique
        automl.fit(X_train=X_trainval, y_train=y_trainval_log, **automl_settings)

        # Logs MLflow
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_param("best_config", automl.best_config)
        mlflow.log_metric("best_loss", automl.best_loss)
        mlflow.log_metric("best_r2", automl.best_result["val_loss"])  # selon métrique

        # Récupérer le meilleur modèle
        best_model = automl.model

    # Sauvegarde du modèle
    out_path = out_model_folder_path / "house_model.joblib"
    joblib.dump(best_model, out_path)

    return best_model

@flow
def train_flow(data_input_dir: Path, data_output_dir: Path, model_output_dir: Path, run_id: str) -> None:
    """
    
    """
    split_dataset(in_df_to_split_folder_path=data_input_dir, out_splitted_dfs_folder_path=data_output_dir)
    fit_model(in_train_dataframe_folder_path=data_input_dir, out_model_folder_path=model_output_dir, run_id=run_id)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    train_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/house/testruns/00000/data'),
        data_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/house/testruns/00000/data'),
        model_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/house/testruns/00000/models'),
        run_id="00000"
        )