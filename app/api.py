from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from .schemas import Wine, Rating, feature_names
from .monitoring import instrumentator

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
model = load(ROOT_DIR / "artifacts/model.joblib")

instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.get("/")
def root():
    return "Wine Quality Ratings"