# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_valid():
    resp = client.post("/predict", json={"features":[5.1,3.5,1.4,0.2]})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)

def test_predict_invalid_length():
    resp = client.post("/predict", json={"features":[1,2,3]})
    assert resp.status_code == 400
