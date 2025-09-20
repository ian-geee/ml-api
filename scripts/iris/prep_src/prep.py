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
def load_raw_data_from_csv(in_put: Path, outdir: str) -> pd.DataFrame:
    df = pd.read_csv(in_put)
    df.to_csv(f"{outdir}/raw_data.csv", index=False, encoding="utf-8")
    return df


@flow
def prep_flow(inputs: str, output: str) -> None:
    """
    Prépare le dataset Iris et écrit iris.csv + meta.json en local, orchestré par Prefect.
    """
    load_raw_data_from_csv(in_put=inputs, outdir=output)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    prep_flow(output="data/iris")
