import pandas as pd
import json, numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42


def save_cv_kfolds(trainval_csv: Path, folds_dir: Path, k: int = 5):
    df = pd.read_csv(trainval_csv)
    y = df["target"].to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    folds_dir.mkdir(parents=True, exist_ok=True)
    for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        with (folds_dir / f"fold_{i}.json").open("w") as f:
            json.dump({"train_idx": tr.tolist(), "val_idx": va.tolist()}, f, indent=2)