# src/prep.py
from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from prefect import flow, task, get_run_logger
from prefect.assets import materialize

from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, jaccard_score,
    matthews_corrcoef, cohen_kappa_score, hamming_loss, zero_one_loss,
    confusion_matrix, classification_report, log_loss, roc_auc_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize  # pour ROC/PR si besoin


def _extract_proba_matrix(df_pred: pd.DataFrame):
    # récupère les colonnes "proba_*" dans l'ordre des classes trouvées dans 'target'
    proba_cols = [c for c in df_pred.columns if c.startswith("proba_")]
    if not proba_cols:
        return None, None
    # déduis l’ordre des classes : tri par le suffixe après "proba_"
    classes = [c.replace("proba_", "") for c in proba_cols]
    # Essayez d’interpréter en numérique si possible
    try:
        classes_cast = [int(c) for c in classes]
        order = np.argsort(classes_cast)
        classes_sorted = [classes[i] for i in order]
        proba_cols_sorted = [proba_cols[i] for i in order]
    except Exception:
        classes_sorted = sorted(classes, key=str)
        proba_cols_sorted = [f"proba_{c}" for c in classes_sorted]
    proba = df_pred[proba_cols_sorted].to_numpy(dtype=float)
    return proba, classes_sorted


@task
def metrics_to_json(in_predictions_df_folder_path: Path, out_metrics_folder_path: Path):
    df_pred = pd.read_csv(in_predictions_df_folder_path / "predictions.csv", index_col=False)

    y_true = df_pred["target"].to_numpy()
    y_pred = df_pred["predicted_target"].to_numpy()

    # ----- Métriques basées sur labels (multiclass) -----
    metrics = {
        # global
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # agrégations standard
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # autres scores utiles
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "jaccard_weighted": jaccard_score(y_true, y_pred, average="weighted", zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "zero_one_loss": zero_one_loss(y_true, y_pred),
        # confusion matrix + report
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    # ----- Métriques basées sur probabilités (si dispo) -----
    proba, classes_sorted = _extract_proba_matrix(df_pred)
    if proba is not None:
        # log-loss (cross-entropy)
        try:
            metrics["log_loss"] = log_loss(y_true, proba, labels=classes_sorted)
        except Exception:
            pass

        # ROC-AUC macro/weighted en one-vs-rest (multiclass)
        try:
            # binarise y_true selon l'ordre des classes retrouvées
            y_true_bin = label_binarize(y_true, classes=[int(c) if str(c).isdigit() else c for c in classes_sorted])
            metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true_bin, proba, multi_class="ovr", average="macro")
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(y_true_bin, proba, multi_class="ovr", average="weighted")
        except Exception:
            pass

        # Top-k accuracy (utile en multiclass) : k=2 et k=3 si nb classes >= k 
        n_classes = proba.shape[1]
        if n_classes >= 2:
            try:
                metrics["top2_accuracy"] = top_k_accuracy_score(y_true, proba, k=2)
            except Exception:
                pass
        if n_classes >= 3:
            try:
                metrics["top3_accuracy"] = top_k_accuracy_score(y_true, proba, k=3)
            except Exception:
                pass

    # ----- Sauvegarde JSON -----
    out_path = out_metrics_folder_path / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return None



@flow
def eval_flow(data_input_dir: Path, metrics_output_dir: Path, run_id: str) -> None:
    """
    
    """
    metrics_to_json(in_predictions_df_folder_path=data_input_dir, out_metrics_folder_path=metrics_output_dir)
    return None


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire
    eval_flow(
        data_input_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/data'),
        metrics_output_dir=Path('C:/Users/joule/work_repos/ml-api/scripts/iris/testruns/2025-09-20_23h4145-d6fc14b1/eval'),
        run_id="2025-09-20_23h4145-d6fc14b1"
        )
