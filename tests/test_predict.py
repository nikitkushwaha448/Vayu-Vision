import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

try:
    from fastapi.testclient import TestClient
    from api.index import app
except Exception:
    app = None


def test_predict_endpoint_available():
    assert app is not None, "FastAPI app not importable"


def test_predict_response():
    client = TestClient(app)
    payload = {
        "city": "Delhi",
        "pollutant_values": {"pm25": 100.0, "pm10": 80.0, "o3": 20.0, "no2": 30.0, "so2": 5.0, "co": 0.4}
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_aqi" in data
    assert isinstance(data["predicted_aqi"], (int, float))
    assert data.get("source") in ("model", "fallback")


def test_predict_validation_missing_keys():
    client = TestClient(app)
    # missing pm10 and others
    payload = {"city": "Delhi", "pollutant_values": {"pm25": 50}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422 or resp.status_code == 400
