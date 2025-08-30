import json, joblib, numpy as np
from pathlib import Path

class HouseService:
    def __init__(self, model_dir: str):
        p = Path(model_dir)
        with open(p / "house_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.model = joblib.load(p / "house_model.joblib")
        self.feature_order = self.meta["feature_order"]

    def predict(self, payload) -> float:
        x = np.array([[getattr(payload, f) for f in self.feature_order]])
        y_log = self.model.predict(x)[0]
        return float(np.expm1(y_log)) if self.meta.get("transform") == "log1p" else float(y_log)