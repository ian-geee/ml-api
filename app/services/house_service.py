import json, joblib, numpy as np
from pathlib import Path
import pandas as pd
import numpy as np

class HouseService:
    def __init__(self, model_dir: str):
        p = Path(model_dir)
        with open(p / "house_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.pipe = joblib.load(p / "house_model.joblib") # a la base self.model car gros débiles de dev de merde qui savent pas faire la différence entre un objet Pipe et un objet Model dans leur propre DOC
        self.feature_order = self.meta["feature_order"]

    def predict(self, payload) -> float:
        # 1) Construire un dict propre depuis le payload
        row = {f: getattr(payload, f) for f in self.feature_order}
        X = pd.DataFrame([row], columns=self.feature_order)

        # 2) Types sûrs pour faire comme à l’entraînement car ce postal_code ne me cause que des problèmes
        X["post_code"] = X["post_code"].astype(str)  # OHE attend du string
        for col in ("habitable_surface", "bedrooms_count", "num_facade"):
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # 3) Prédire
        y_hat = self.pipe.predict(X)[0] # a la base self.model car gros débiles de dev de merde qui savent pas faire la différence entre un objet Pipe et un objet Model dans leur propre DOC
        y = float(np.expm1(y_hat)) if self.meta.get("transform") == "log1p" else float(y_hat)
        return y