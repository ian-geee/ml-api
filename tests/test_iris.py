from fastapi.testclient import TestClient
from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in {"ok", "booting"}

def test_predict_ok():
    with TestClient(app) as client:  # ← IMPORTANT : context manager qui entoure le client de test : Entrée (__enter__) : Lance les hooks startup / lifespan de FastAPI/Starlette → dans ton code, ça exécute lifespan et charge IrisService dans app.state.iris_service
        payload = {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
        r = client.post("/api/v1/iris/predict", json=payload)
        assert r.status_code == 200, f"{r.status_code} {r.text}"
        data = r.json()
        assert "predicted_class_label" in data

# Sans le with, selon les versions, le lifespan peut ne pas s’exécuter au bon moment. Résultat :
# - /health (et donc test_health) peut renvoyer "booting" (tu l’acceptais, donc test OK),
# - mais /predict (et donc test_predict) échoue en 503 (ton get_iris_service ne trouve pas app.state.iris_service).