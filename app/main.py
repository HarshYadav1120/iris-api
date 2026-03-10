# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

class Features(BaseModel):
    features: list

app = FastAPI(title="Iris Predict API")
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

class Features(BaseModel):
    features: list

app = FastAPI(title="Iris Predict API")
# load model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(payload: Features):
    feats = payload.features
    if not isinstance(feats, list) or len(feats) != 4:
        raise HTTPException(status_code=400, detail="features must be a list of 4 numbers")
    try:
        arr = np.array(feats, dtype=float).reshape(1, -1)
        pred = model.predict(arr).tolist()
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
