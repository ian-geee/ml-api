from pathlib import Path

IRIS_ROOT = Path(__file__).resolve().parents[0]  
REPO_ROOT = Path(__file__).resolve().parents[2] # racine = 2 niveaux au-dessus de scripts/iris

RAW_DATA_PATH = IRIS_ROOT / "data" / "iris.csv"
print(RAW_DATA_PATH)
print(type(RAW_DATA_PATH))
print(str(RAW_DATA_PATH))