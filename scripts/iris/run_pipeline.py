from __future__ import annotations

import os, re, shlex, subprocess, sys
from pathlib import Path
from typing import Any, Dict

import yaml
from prefect import flow, task, get_run_logger

PLACEHOLDER = re.compile(r"\$\{\{([^}]+)\}\}")

def load_yaml(p: Path) -> Dict[str, Any]:
    """
    Opens a file.yaml content and load it into a variable as a dictionary
    """
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parents(path: str | Path):
    """
    Assure que les dossiers parents des fichiers que nous allons sauvegarder sont bien présents, en les créant ou pas s'ils existent déjà (exist_ok = True)
    """
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

    name = step_cfg["name"] # step_cfg = {'name': 'prep', 'component': 'components/prep.yaml', 'with': {'inputs': {}, 'outputs': {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}}}
    comp_path = Path(step_cfg["component"]) # comp_path = Path('components/prep.yaml')
    comp = load_yaml(comp_path) # comp = {'$schema': 'local.commandComponent', 'name': 'prepare_iris', 'display_name': 'PrepIris', 'version': 1, 'type': 'command', 'inputs': {}, 'outputs': {'csv': {...}, 'meta': {...}}, 'code': '../prep_src', 'environment': 'local/python', 'command': 'python prep.py --output ${{outputs.out_dir}}', 'defaults': {'out_dir': 'saves/iris'}}

    step_with = step_cfg.get("with", {}) # step_with = {'inputs': {}, 'outputs': {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}}
    step_inputs = step_with.get("inputs", {}) or {} # step_inputs = {}
    step_outputs = step_with.get("outputs", {}) or {} # step_outputs = {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}

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
    for k, v in step_outputs.items(): # step_outputs = {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}
        mapping[f"outputs.{k}"] = str(v) # mapping = {'outputs.out_dir': 'saves/iris', 'outputs.csv': 'saves/iris/iris.csv', 'outputs.meta': 'saves/iris/meta.json'}
        # last k = "meta", last v = 'saves/iris/meta.json'

    # defaults du composant (optionnel)
    for k, v in (comp.get("defaults") or {}).items(): # comp.get("defaults") = {'out_dir': 'saves/iris'}
        mapping[f"outputs.{k}"] = mapping.get(f"outputs.{k}", str(v)) 
    # mapping = {'outputs.out_dir': 'saves/iris', 'outputs.csv': 'saves/iris/iris.csv', 'outputs.meta': 'saves/iris/meta.json'},
    # last k = 'out_dir',
    # last v = 'saves/iris'

    # Construire la commande
    raw_cmd = comp["command"] # raw_cmd = 'python prep.py --output ${{outputs.out_dir}}',
    cmd = substitute(raw_cmd, mapping) # mapping = {'outputs.out_dir': 'saves/iris', 'outputs.csv': 'saves/iris/iris.csv', 'outputs.meta': 'saves/iris/meta.json'}
    # cmd = 'python prep.py --output saves/iris'


    # Prépare WD + env
    workdir = (comp_path.parent / comp["code"]).resolve() # L’opérateur / de pathlib.Path accepte un str à droite et retourne un nouveau path ; .resolve() vient après : il transforme ce chemin en chemin absolu
    # workdir = WindowsPath('C:/Users/joule/work_repos/ml-api/scripts/iris/prep_src')
    # comp["code"] = '../prep_src'
    # comp_path = WindowsPath('components/prep.yaml')
    # comp_path.parent = WindowsPath('components')
    
    workdir.mkdir(parents=True, exist_ok=True) # crée le dossier s’il n’existe pas (et ses parents existent)

    # Crée les dossiers des outputs
    for v in step_outputs.values(): # step_outputs = {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}
        ensure_parents(v) # après ça, on est sur de pouvoir enregistrer les fichiers là ou nous le souhaitons
        

    logger.info(f"[{name}] CMD: {cmd}") # logger.info équivaut à un print
    logger.info(f"[{name}] CWD: {workdir}")

    # Lance le process : 
    parts = shlex.split(cmd) # shlex coupe la commande : cmd = 'python prep.py --output saves/iris' en plusieurs mots
    # Ici sert à lancer le bon Python (celui du venv et pas un autre ? Mais wtf encore) qui est sys.executable : 'C:\\Users\\joule\\work_repos\\ml-api\\.venv\\Scripts\\python.exe'
    if parts and parts[0].lower() in {"python", "python3", "py"}:
        parts[0] = sys.executable # On remplace le premier mot, càd python dans la liste des morceaux de la commande par le path correct vers l'exécutable python du venv

    proc = subprocess.run(
        parts, # parts = ['C:\\Users\\joule\\work_repos\\ml-api\\.venv\\Scripts\\python.exe', 'prep.py', '--output', 'saves/iris']
        cwd=str(workdir),
        env=os.environ.copy(),
        text=True,
        capture_output=True
    )
    # Log
    if proc.stdout: # proc.stdout = 'Wrote: C:\\Users\\joule\\work_repos\\ml-api\\scripts\\iris\\prep_src\\saves\\iris\\iris.csv\nWrote: C:\\Users\\joule\\work_repos\\ml-api\\scripts\\iris\\prep_src\\saves\\iris\\meta.json\n'
        logger.info(proc.stdout.strip()) # print :  Wrote: C:\Users\joule\work_repos\ml-api\scripts\iris\prep_src\saves\iris\iris.csv & Wrote: C:\Users\joule\work_repos\ml-api\scripts\iris\prep_src\saves\iris\meta.json
    if proc.stderr: # proc.stderr = '' (empty string cause no error)
        logger.warning(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed with code {proc.returncode}") # name = 'prep'

    # Retourne les outputs pour câblage downstream, comme ça train.py saura où aller chercher le prepared_data et ses meta-données, qu'il remplira à son tour
    return {k: str(v) for k, v in step_outputs.items()} # step_outputs = {'out_dir': 'saves/iris', 'csv': 'saves/iris/iris.csv', 'meta': 'saves/iris/meta.json'}

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
