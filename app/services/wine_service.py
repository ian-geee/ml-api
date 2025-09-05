import json, joblib, numpy as np
from pathlib import Path
import pandas as pd
import numpy as np

class WineService:
    def __init__(self, model_dir: str):
        p = Path(model_dir)
        with open(p / "wine_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.pipe = joblib.load(p / "wine_model.joblib")
        self.feature_order = self.meta["feature_order"]

    def predict(self, payload) -> float:
        # 1) Construire un dict propre depuis le payload
        row = {f: getattr(payload, f) for f in self.feature_order}
        X = pd.DataFrame([row], columns=self.feature_order)

        # 3) Prédire
        y_hat = self.pipe.predict(X)[0]
        y = round(float(y_hat), 2)
        return y