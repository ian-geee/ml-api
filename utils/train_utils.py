import pandas as pd
import json, numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

def save_cv_kfolds_to_json(in_trainval_dataframe_folder_path: Path, out_json_folds_folder_path: Path, k: int = 5, random_state: int = 42) -> None: 
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
    return None



def grid_search(model, X, y, param_grid, cv, scoring, n_jobs):
    """
    Perform an exhaustive grid search and return the best model.

    Parameters
    ----------
    model : estimator
        Base estimator.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : array-like or pd.Series, shape (n_samples,)
        True labels.
    param_grid : dict
        Dictionary with parameter names (str) as keys and lists of settings.
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : str or callable, default='precision'
        Metric used to evaluate parameter combinations.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 => use all processors).

    Returns
    -------
    tuple (dict, estimator)
        Best parameter set and the refitted best estimator.
    """

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_params, best_model
    
