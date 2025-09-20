from __future__ import annotations

import os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Any, Dict

from prefect import flow, task, get_run_logger
from prefect.runtime import flow_run

from prep_src.prep import prep_flow

IRIS_ROOT = Path(__file__).resolve().parents[0]  
REPO_ROOT = Path(__file__).resolve().parents[2] # racine = 2 niveaux au-dessus de scripts/iris
INPUT = Path(REPO_ROOT / "data" / "iris.csv").resolve()

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
def make_output_dir(output: str) -> str:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir.resolve())

@flow(name="ml_pipeline", log_prints=True)
def run_pipeline() -> None:
    """
    
    """
    logger = get_run_logger()
    run_id = flow_run.id
    logger.info(run_id)

    prep_out_string = make_output_dir(output="scripts/iris/runs/" + run_id + "/data")
    make_output_dir(output="scripts/iris/runs/" + run_id + "/models")
    make_output_dir(output="scripts/iris/runs/" + run_id + "/score")
    make_output_dir(output="scripts/iris/runs/" + run_id + "/eval")

    prep_flow(inputs=str(INPUT), output=prep_out_string)
    
    return None

if __name__ == "__main__":
    run_pipeline()    
    print(IRIS_ROOT)
