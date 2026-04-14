from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import lightgbm as lgb
import time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="House Price Prediction API", version="1.0.0")

model = lgb.Booster(model_file="model_booster.txt")
logger.info("Model loaded successfully")

class HouseFeatures(BaseModel):
    area_sq_ft: float    = Field(..., gt=0)
    bedrooms: int        = Field(..., ge=0)
    bathrooms: int       = Field(..., ge=0)
    stories: int         = Field(..., ge=1)
    guestroom: int       = Field(..., ge=0, le=1)
    basement: int        = Field(..., ge=0, le=1)
    airconditioning: int = Field(..., ge=0, le=1)
    prefarea: int        = Field(..., ge=0, le=1)
    parking: int         = Field(..., ge=0)
    furnishingstatus_semi_furnished: int = Field(..., ge=0, le=1)
    furnishingstatus_unfurnished: int    = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    predicted_price: float
    latency_ms: float

class BatchRequest(BaseModel):
    houses: list[HouseFeatures]

class BatchResponse(BaseModel):
    predictions: list[float]
    count: int
    total_latency_ms: float
    avg_latency_ms: float

def to_array(houses):
    return [[
        h.area_sq_ft, h.bedrooms, h.bathrooms, h.stories,
        h.guestroom, h.basement, h.airconditioning, h.prefarea,
        h.parking, h.furnishingstatus_semi_furnished,
        h.furnishingstatus_unfurnished
    ] for h in houses]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    start = time.perf_counter()
    try:
        price = float(model.predict(to_array([house]))[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        predicted_price=round(price, 2),
        latency_ms=round(latency_ms, 3)
    )

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    start = time.perf_counter()
    try:
        prices = model.predict(to_array(request.houses)).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    total_ms = (time.perf_counter() - start) * 1000
    avg_ms = total_ms / len(request.houses)
    return BatchResponse(
        predictions=[round(p, 2) for p in prices],
        count=len(prices),
        total_latency_ms=round(total_ms, 3),
        avg_latency_ms=round(avg_ms, 3)
    )

@app.get("/model/info")
def model_info():
    return {"model_type": "LightGBM Booster", "num_features": model.num_feature()}