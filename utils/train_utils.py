import pandas as pd
import json, numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

def save_cv_kfolds_to_json(in_trainval_dataframe_folder_path: Path, out_json_folds_folder_path: Path, k: int = 5, random_state: int = 42):
    """
    If random_state in globally controled, saves the index of the k train / validation rows making the folds for cross-validation
    """
    df = pd.read_csv(in_trainval_dataframe_folder_path / "trainval.csv")
    y = df["target"].to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    out_json_folds_folder_path = (out_json_folds_folder_path / "trainval_folds")
    out_json_folds_folder_path.mkdir(parents=True, exist_ok=True)
    for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        with (out_json_folds_folder_path / f"fold_{i}.json").open("w") as f:
            json.dump({"train_idx": tr.tolist(), "val_idx": va.tolist()}, f, indent=2)