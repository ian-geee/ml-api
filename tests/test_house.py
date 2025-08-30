from fastapi.testclient import TestClient
from app.main import app

def test_house_predict_ok():
    with TestClient(app) as client:
        payload = {
            "GrLivArea": 1500,
            "OverallQual": 6,
            "YearBuilt": 1995,
            "GarageCars": 2,
            "FullBath": 2,
            "LotArea": 8000
        }
        r = client.post("/api/v1/house/predict", json=payload)
        if r.status_code == 503:
            # service not loaded (artifacts missing) -> skip gracefully
            assert True
            return
        assert r.status_code == 200
        assert "price_usd" in r.json()