# src/prep.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from prefect import flow, task, get_run_logger
from prefect.assets import materialize

@task
def make_output_dir(output: str) -> str:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir.resolve())

@materialize(f"file://temp/raw_data.csv")
def load_raw_data_from_csv(in_raw_data_folder: Path, out_clean_data_folder: Path) -> pd.DataFrame:
    df = pd.read_csv(in_raw_data_folder / "immoweb2022_houses-only_4features_no-outliers.csv")
    df.rename(columns={"price": "target"}, inplace=True)
    df.to_csv(f"{out_clean_data_folder}/raw_data.csv", index=False, encoding="utf-8")
    return df


@flow
def prep_flow(in_raw_data_folder: Path, out_clean_data_folder: Path, run_id: str) -> None:
    """
    Prépare le dataset Iris et écrit iris.csv + meta.json en local, orchestré par Prefect.
    """
    load_raw_data_from_csv(in_raw_data_folder=in_raw_data_folder, out_clean_data_folder=out_clean_data_folder)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    prep_flow(in_raw_data_folder=Path('C:/Users/joule/work_repos/ml-api/scripts/house/data'),
              out_clean_data_folder=Path('C:/Users/joule/work_repos/ml-api/scripts/house/data'),
              run_id="00000"
              )
