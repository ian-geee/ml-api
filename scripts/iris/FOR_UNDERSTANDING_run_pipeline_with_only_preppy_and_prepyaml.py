from __future__ import annotations

import os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Any, Dict

import yaml
from prefect import flow, task, get_run_logger

REPO_ROOT = Path(__file__).resolve().parents[2]  # racine = 2 niveaux au-dessus de scripts/iris

PLACEHOLDER = re.compile(r"\$\{\{([^}]+)\}\}")

def load_yaml(p: Path) -> Dict[str, Any]:
    """
    Opens a file.yaml content and load it into a variable as a dictionary
    """
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

    name = step_cfg["name"] # step_cfg = {'name': 'prep', 'component': 'components/prep.yaml', 'with': {'inputs': {}, 'outputs': {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}}}
    comp_path = Path(step_cfg["component"]) # comp_path = Path('components/prep.yaml')
    comp = load_yaml(comp_path) # comp = {'$schema': 'local.commandComponent', 'name': 'prepare_iris', 'display_name': 'PrepIris', 'version': 1, 'type': 'command', 'inputs': {}, 'outputs': {'csv': {...}, 'meta': {...}}, 'code': '../prep_src', 'environment': 'local/python', 'command': 'python prep.py --output ${{outputs.out_dir}}'}

    step_with = step_cfg.get("with", {}) # step_with = {'inputs': {}, 'outputs': {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}}
    step_inputs = step_with.get("inputs", {}) or {} # step_inputs = {}
    step_outputs = step_with.get("outputs", {}) or {} # step_outputs = {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}

    # 🔒 Ancrer tous les outputs sur la racine du repo (chemins ABSOLUS)
    norm_outputs = {}
    for k, v in step_outputs.items():
        p = Path(v) # p = WindowsPath('saves/temp')
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve() # REPO_ROOT = WindowsPath('C:/Users/joule/work_repos/ml-api') (défini par user en haut)
            # REPO_ROOT / p = WindowsPath('C:/Users/joule/work_repos/ml-api/saves/iris')
            # p = WindowsPath('C:/Users/joule/work_repos/ml-api/saves/iris/meta.json')
        norm_outputs[k] = p.as_posix() # p.as_posix() = 'C:/Users/joule/work_repos/ml-api/saves/temp/meta.json'
    step_outputs = norm_outputs
    # step_outputs = {'out_dir': 'C:/Users/joule/work_repos/ml-api/saves/iris', 'csv': 'C:/Users/joule/work_repos/ml-api/saves/iris/iris.csv', 'meta': 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'}

    # mapping disponible pour la substitution
    mapping: Dict[str, str] = {}

    # Expose steps.<name>.outputs.<key> pour le graph
    for sname, sdict in gctx.items(): # gctx = {}
        for k, v in sdict.items():
            mapping[f"steps.{sname}.outputs.{k}"] = v # mapping = {}

    # inputs déclarés
    for k, v in step_inputs.items(): # step_inputs = {}
        mapping[f"inputs.{k}"] = str(v) # mapping = {}

    # outputs déclarés (fixe le chemin final)
    for k, v in step_outputs.items(): # step_outputs = {'out_dir': 'C:/Users/joule/work_repos/ml-api/saves/iris', 'csv': 'C:/Users/joule/work_repos/ml-api/saves/iris/iris.csv', 'meta': 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'}
        mapping[f"outputs.{k}"] = str(v) # mapping = {'outputs.out_dir': 'C:/Users/joule/work_repos/ml-api/saves/iris', 'outputs.csv': 'C:/Users/joule/work_repos/ml-api/saves/iris/iris.csv', 'outputs.meta': 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'}
        # last k = "meta", last v = 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'

    # Construire la commande
    raw_cmd = comp["command"] # raw_cmd = 'python prep.py --output ${{outputs.out_dir}}',
    cmd = substitute(raw_cmd, mapping) # mapping = {'outputs.out_dir': 'saves/iris', 'outputs.csv': 'saves/iris/iris.csv', 'outputs.meta': 'saves/iris/meta.json'}
    # cmd = 'python prep.py --output C:/Users/joule/work_repos/ml-api/saves/iris'


    # Le workdir du script prep.py
    workdir = (comp_path.parent / comp["code"]).resolve() # L’opérateur / de pathlib.Path accepte un str à droite et retourne un nouveau path ; .resolve() vient après : il transforme ce chemin en chemin absolu
    # workdir = WindowsPath('C:/Users/joule/work_repos/ml-api/scripts/iris/prep_src')
    # comp["code"] = '../prep_src'
    # comp_path = WindowsPath('components/prep.yaml')
    # comp_path.parent = WindowsPath('components')
        
    logger.info(f"[{name}] Command: {cmd}") # logger.info équivaut à un print
    logger.info(f"[{name}] Current Working Dir: {workdir}")

    # Lance le process : 
    parts = shlex.split(cmd) # shlex coupe la commande : cmd = 'python prep.py --output C:/Users/joule/work_repos/ml-api/saves/iris' en plusieurs mots
    # Ici sert à lancer le bon Python (celui du venv et pas un autre ? Mais wtf encore) qui est sys.executable : 'C:\\Users\\joule\\work_repos\\ml-api\\.venv\\Scripts\\python.exe'
    if parts and parts[0].lower() in {"python", "python3", "py"}:
        parts[0] = sys.executable # On remplace le premier mot, càd python dans la liste des morceaux de la commande par le path correct vers l'exécutable python du venv

    proc = subprocess.run(
        parts, # parts = ['C:\\Users\\joule\\work_repos\\ml-api\\.venv\\Scripts\\python.exe', 'prep.py', '--output', 'C:/Users/joule/work_repos/ml-api/saves/iris']
        cwd=str(workdir),
        env=os.environ.copy(),
        text=True,
        capture_output=True
    )
    # Log
    if proc.stdout: # proc.stdout = 'Wrote: C:\\Users\\joule\\work_repos\\ml-api\\saves\\iris\\iris.csv\nWrote: C:\\Users\\joule\\work_repos\\ml-api\\saves\\iris\\meta.json\n'
        logger.info(proc.stdout.strip()) # print :  Wrote: C:\Users\joule\work_repos\ml-api\saves\iris\iris.csv & Wrote: C:\Users\joule\work_repos\ml-api\saves\iris\meta.json
    if proc.stderr: # proc.stderr = '' (empty string cause no error)
        logger.warning(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed with code {proc.returncode}") # name = 'prep'

    # Retourne les outputs pour câblage downstream, comme ça train.py saura où aller chercher le prepared_data et ses meta-données, qu'il remplira à son tour
    return {k: str(v) for k, v in step_outputs.items()} 

@flow(name="local_yaml_pipeline")
def run_pipeline(pipeline_yaml: str = "pipeline.yaml"):
    """
    Orchestration locale: lit pipeline.yaml, enchaîne les steps en résolvant
    ${{inputs.*}}, ${{outputs.*}} et ${{steps.<name>.outputs.*}}.
    """
    logger = get_run_logger()
    cfg = load_yaml(Path(pipeline_yaml)) # pipeline_yaml = 'C:\\Users\\joule\\work_repos\\ml-api/scripts/iris/pipeline.yaml'
    steps = cfg["steps"] # cfg = {'$schema': 'local.pipeline', 'name': 'iris_local_pipeline', 'version': 1, 'steps': [{...}]}
    # steps = cfg["steps"] = [{'name': 'prep', 'component': 'components/prep.yaml', 'with': {...}}]
    # cfg["steps"][0]["with"] = {'inputs': {}, 'outputs': {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}}

    graph_ctx: Dict[str, Dict[str, str]] = {}

    for step in steps: # 1 seul step: prep
        name = step["name"]
        outs = run_step(step, graph_ctx) # outs = {'out_dir': 'C:/Users/joule/work_repos/ml-api/saves/iris', 'csv': 'C:/Users/joule/work_repos/ml-api/saves/iris/iris.csv', 'meta': 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'}
        graph_ctx[name] = outs # graph_ctx = {'prep': {'out_dir': 'C:/Users/joule/work_repos/ml-api/saves/iris', 'csv': 'C:/Users/joule/work_repos/ml-api/saves/iris/iris.csv', 'meta': 'C:/Users/joule/work_repos/ml-api/saves/iris/meta.json'}}
        logger.info(f"[{name}] outputs: {outs}")

    logger.info("[pipeline] DONE")
    return graph_ctx

if __name__ == "__main__":
    # Usage: python run_pipeline.py [pipeline.yaml]
    p = sys.argv[1] if len(sys.argv) > 1 else "pipeline.yaml"
    run_pipeline(pipeline_yaml=p)
