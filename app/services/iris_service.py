from datetime import datetime
import json, joblib, numpy as np
from pathlib import Path
from typing import Dict, Any, List

class IrisService:
    def __init__(self, model_dir: str):
        p = Path(model_dir) # initialisation du path
        with open(p / "iris_model_meta.json", "r", encoding="utf-8") as f: # partie json.load des metadata du model : Dict
            self.meta: Dict[str, Any] = json.load(f)
        self.model = joblib.load(p / "iris_model.joblib") # partie joblib.load du model : Object Pipeline(steps=[('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
        self.feature_order: List[str] = self.meta["feature_order"] # attribute : List ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.class_names: List[str] = self.meta.get("classes") or [str(int(c)) for c in self.model.classes_] # attribute : List ['setosa', 'versicolor', 'virginica']

    def predict(self, payload) -> Dict[str, Any]:
        x = np.array([[getattr(payload, f) for f in self.feature_order]])
        proba = self.model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        return {
            "predicted_class_index": pred_idx,
            "predicted_class_label": self.class_names[pred_idx],
            "probabilities": { self.class_names[i]: float(p) for i, p in enumerate(proba) },
            "feature_order": self.feature_order,
        }
