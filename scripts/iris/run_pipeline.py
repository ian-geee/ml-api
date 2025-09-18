from __future__ import annotations

import os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Any, Dict

import yaml
from prefect import flow, task, get_run_logger

PLACEHOLDER = re.compile(r"\$\{\{([^}]+)\}\}")

def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parents(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def substitute(s: str, mapping: Dict[str, str]) -> str:
    def repl(m):
        key = m.group(1).strip()
        return mapping.get(key, m.group(0))
    return PLACEHOLDER.sub(repl, s)

@task
def run_step(step_cfg: Dict[str, Any], gctx: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Exécute un step:
      - charge le composant
      - résout les placeholders
      - lance le process
      - retourne les outputs (chemins) pour usage downstream
    """
    logger = get_run_logger()

    name = step_cfg["name"]
    comp_path = Path(step_cfg["component"])
    comp = load_yaml(comp_path)

    step_with = step_cfg.get("with", {})
    step_inputs = step_with.get("inputs", {}) or {}
    step_outputs = step_with.get("outputs", {}) or {}

    # mapping disponible pour la substitution
    mapping: Dict[str, str] = {}

    # Expose steps.<name>.outputs.<key> pour le graph
    for sname, sdict in gctx.items():
        for k, v in sdict.items():
            mapping[f"steps.{sname}.outputs.{k}"] = v

    # inputs déclarés
    for k, v in step_inputs.items():
        mapping[f"inputs.{k}"] = str(v)

    # outputs déclarés (fixe le chemin final)
    for k, v in step_outputs.items():
        mapping[f"outputs.{k}"] = str(v)

    # defaults du composant (optionnel)
    for k, v in (comp.get("defaults") or {}).items():
        mapping[f"outputs.{k}"] = mapping.get(f"outputs.{k}", str(v))

    # Construire la commande
    raw_cmd = comp["command"]
    cmd = substitute(raw_cmd, mapping)

    # Prépare WD + env
    workdir = (comp_path.parent / comp["code"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Crée les dossiers des outputs
    for v in step_outputs.values():
        ensure_parents(v)

    logger.info(f"[{name}] CMD: {cmd}")
    logger.info(f"[{name}] CWD: {workdir}")

    # Lance le process : tout ce charabia sert à lancer le bon Python (celui du venv et pas un autre ? Mais wtf encore)
    parts = shlex.split(cmd)
    if parts and parts[0].lower() in {"python", "python3", "py"}:
        parts[0] = sys.executable

    proc = subprocess.run(
        parts,
        cwd=str(workdir),
        env=os.environ.copy(),
        text=True,
        capture_output=True
    )
    # Log
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.warning(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed with code {proc.returncode}")

    # Retourne les outputs pour câblage downstream
    return {k: str(v) for k, v in step_outputs.items()}

@flow(name="local_yaml_pipeline")
def run_pipeline(pipeline_yaml: str = "pipeline.yaml"):
    """
    Orchestration locale: lit pipeline.yaml, enchaîne les steps en résolvant
    ${{inputs.*}}, ${{outputs.*}} et ${{steps.<name>.outputs.*}}.
    """
    logger = get_run_logger()
    cfg = load_yaml(Path(pipeline_yaml))
    steps = cfg["steps"]

    graph_ctx: Dict[str, Dict[str, str]] = {}

    for step in steps:
        name = step["name"]
        outs = run_step(step, graph_ctx)
        graph_ctx[name] = outs
        logger.info(f"[{name}] outputs: {outs}")

    logger.info("[pipeline] DONE")
    return graph_ctx

if __name__ == "__main__":
    # Usage: python run_pipeline.py [pipeline.yaml]
    p = sys.argv[1] if len(sys.argv) > 1 else "pipeline.yaml"
    run_pipeline(pipeline_yaml=p)
