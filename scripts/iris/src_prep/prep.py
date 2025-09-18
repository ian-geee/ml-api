# src/prep.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="Dossier de sortie où écrire iris.csv et meta.json")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charge Iris
    iris = load_iris()
    X: np.ndarray = iris.data
    y: np.ndarray = iris.target
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    classes = iris.target_names.tolist()

    # Construit un DataFrame simple (features + target)
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Écrit les fichiers
    csv_path = out_dir / "iris.csv"
    meta_path = out_dir / "meta.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    meta = {
        "feature_order": feature_names,
        "classes": classes,
        "dataset_name": "iris",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")

if __name__ == "__main__":
    main()
