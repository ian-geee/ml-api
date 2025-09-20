from __future__ import annotations

import os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Any, Dict

from prefect import flow, task, get_run_logger
from prefect.runtime import flow_run

from prep_src.prep import prep_flow
from train_src.train import train_flow

RANDOM_STATE = 42

IRIS_ROOT = Path(__file__).resolve().parents[0]  
REPO_ROOT = Path(__file__).resolve().parents[2] # racine = 2 niveaux au-dessus de scripts/iris

RAW_DATA_PATH = (IRIS_ROOT / "data").resolve()

# @task
# def run_step() -> None:
#     """
#     Exécute un step:
#       - charge le composant
#       - résout les placeholders
#       - lance le process
#       - retourne les outputs (chemins) pour usage downstream
#     """
#     return None

@task
def make_output_dir(output: str) -> Path:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()

@flow(name="ml_pipeline", log_prints=True)
def run_pipeline() -> None:
    """
    
    """
    logger = get_run_logger()
    run_id = flow_run.id
    logger.info(run_id)

    data_folder = make_output_dir(output="scripts/iris/runs/" + run_id + "/data")
    model_folder = make_output_dir(output="scripts/iris/runs/" + run_id + "/models")
    make_output_dir(output="scripts/iris/runs/" + run_id + "/score")
    make_output_dir(output="scripts/iris/runs/" + run_id + "/eval")


    prep_flow(in_raw_data_folder=RAW_DATA_PATH, out_clean_data_folder=data_folder)
    train_flow(data_input_dir=data_folder, data_output_dir=data_folder, model_output_dir=model_folder)
    
    return None

if __name__ == "__main__":
    run_pipeline()    
    print(IRIS_ROOT)
