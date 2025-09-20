# src/prep.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def make_output_dir(output: str) -> str:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir.resolve())


def load_iris_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    iris = load_iris()
    X: np.ndarray = iris.data
    y: np.ndarray = iris.target
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    class_names = iris.target_names.tolist()
    return X, y, feature_names, class_names


def build_dataframe(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def write_csv(df: pd.DataFrame, out_dir: str) -> str:
    csv_path = Path(out_dir) / "iris.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path)


def write_meta(
    out_dir: str, feature_names: List[str], class_names: List[str], rows: int, cols: int
) -> str:
    meta = {
        "feature_order": feature_names,
        "classes": class_names,
        "dataset_name": "iris",
        "rows": int(rows),
        "cols": int(cols),
    }
    meta_path = Path(out_dir) / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(meta_path)


def prep_flow(output: str = "data/iris") -> dict:
    """
    Prépare le dataset Iris et écrit iris.csv + meta.json en local, orchestré par Prefect.
    """

    out_dir = make_output_dir(output)
    X, y, feature_names, class_names = load_iris_data()
    df = build_dataframe(X, y, feature_names)

    csv_path = write_csv(df, out_dir)
    meta_path = write_meta(out_dir, feature_names, class_names, df.shape[0], df.shape[1])

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")
    return {"csv": csv_path, "meta": meta_path}


if __name__ == "__main__":
    # Lancement direct en local (sans DVC/MLflow, ni serveur Prefect nécessaire)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/iris",
                        help="Dossier de sortie où écrire iris.csv et meta.json")
    args = parser.parse_args()

    prep_flow(output=args.output)
