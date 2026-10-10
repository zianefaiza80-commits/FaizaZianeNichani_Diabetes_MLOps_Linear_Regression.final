from fastapi.testclient import TestClient
import api

client = TestClient(api.app)

class FakeModel:
    def predict(self, data):
        return [150.0]

def valid_data():
    return {
        "age": 0.04, "sex": 0.05, "bmi": 0.06, "bp": 0.07,
        "s1": 0.08, "s2": 0.09, "s3": 0.10, "s4": 0.11,
        "s5": 0.12, "s6": 0.13
    }

def test_predict_success(monkeypatch):
    monkeypatch.setattr(api, "model", FakeModel())
    response = client.post("/predict", json=valid_data())
    assert response.status_code == 200
    assert response.json()["prediction"] == 150.0

def test_invalid_input():
    response = client.post("/predict", json={"age": "not-a-number"})
    assert response.status_code == 422

def test_model_unavailable(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    response = client.get("/health")
    assert response.status_code == 503
