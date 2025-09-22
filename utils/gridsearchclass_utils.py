from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

class GridSearchRunner:
    """
    Wrapper OO autour de GridSearchCV avec:
    - fit(X, y): lance la recherche
    - fit_from_path(path): lit <path>/trainval.csv, sépare X/y, puis lance la recherche
    - Attributs: best_params_, best_estimator_, cv_results_, gs_ (l'objet GridSearchCV)
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Iterable],
        *,
        cv: int = 5,
        scoring: Union[str, callable] = "precision",
        n_jobs: int = -1,
        refit: bool = True,
        target_col: str = "target",
        return_train_score: bool = False,
    ) -> None:
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.target_col = target_col
        self.return_train_score = return_train_score

        # Remplis après fit
        self.gs_: Optional[GridSearchCV] = None
        self.best_params_: Optional[Dict[str, object]] = None
        self.best_estimator_: Optional[BaseEstimator] = None
        self.cv_results_: Optional[Dict[str, Iterable]] = None

    # ---------- API principale ----------
    def fit(self, X, y) -> Tuple[Dict[str, object], BaseEstimator]:
        self.gs_ = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            return_train_score=self.return_train_score,
        )
        self.gs_.fit(X, y)
        self.best_params_ = self.gs_.best_params_
        self.best_estimator_ = self.gs_.best_estimator_
        self.cv_results_ = self.gs_.cv_results_
        return self.best_params_, self.best_estimator_

    def fit_from_path(self, folder: Path) -> Tuple[Dict[str, object], BaseEstimator]:
        df = self._read_trainval(folder)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return self.fit(X, y)

    # ---------- Utilitaires ----------
    def save_cv_results_csv(self, outfile: Path) -> None:
        """Sauvegarde les résultats complets de la grille en CSV (très utile pour debug/comparaisons)."""
        if self.cv_results_ is None:
            raise RuntimeError("Aucun résultat: appelez fit(...) d'abord.")
        pd.DataFrame(self.cv_results_).to_csv(outfile, index=False)

    def _read_trainval(self, folder: Path) -> pd.DataFrame:
        csv_path = folder / "trainval.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
        df = pd.read_csv(csv_path)
        if self.target_col not in df.columns:
            raise ValueError(f"Colonne cible '{self.target_col}' absente de {csv_path.name}")
        return df