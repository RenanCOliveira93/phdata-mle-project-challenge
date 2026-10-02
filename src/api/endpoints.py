import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

from utils.loader import fit_imputer, load_demographics, load_features, load_model

logger = logging.getLogger(__name__)

router = APIRouter()

# House attributes that may be missing in a request and are filled by KNN imputation.
HOUSE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "sqft_above", "sqft_basement",
]

# model/ and data/ live next to the api/ package, independent of the working directory.
APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "model"
DATA_DIR = APP_DIR / "data"

# Immutable resources, loaded once at import time and shared by all requests.
model = load_model(MODEL_DIR / "model.pkl")
model_features = load_features(MODEL_DIR / "model_features.json")
demographics = load_demographics(DATA_DIR / "zipcode_demographics.csv")
imputer = fit_imputer(DATA_DIR / "kc_house_data.csv", HOUSE_FEATURES)
logger.info(
    "Loaded model (%d features), demographics for %d zipcodes, "
    "KNN imputer over %d house features",
    len(model_features), len(demographics), imputer.n_features_in_,
)


class HomeFeatures(BaseModel):
    """Prediction request. Only zipcode is required; missing house fields are imputed."""

    # Reject NaN/Infinity: missing values must be sent as null or omitted
    model_config = ConfigDict(allow_inf_nan=False)

    bedrooms: Optional[int] = Field(None, ge=0)
    bathrooms: Optional[float] = Field(None, ge=0)
    sqft_living: Optional[float] = Field(None, gt=0)
    sqft_lot: Optional[float] = Field(None, gt=0)
    floors: Optional[float] = Field(None, gt=0)
    sqft_above: Optional[float] = Field(None, ge=0)
    sqft_basement: Optional[float] = Field(None, ge=0)
    zipcode: str


class PredictionResponse(BaseModel):
    predicted_price: float


@router.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if API is ready to accept requests.
    """
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(home_features: HomeFeatures):
    demographic_info = demographics.get(home_features.zipcode)
    if demographic_info is None:
        logger.info("Rejected unknown zipcode %s", home_features.zipcode)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown zipcode: {home_features.zipcode}",
        )

    house = home_features.model_dump(exclude={"zipcode"})

    # Impute missing house attributes; complete requests skip this step
    missing = [name for name, value in house.items() if value is None]
    if missing:
        log = logger.warning if len(missing) == len(HOUSE_FEATURES) else logger.info
        log("Imputing %s for zipcode %s", missing, home_features.zipcode)
        incomplete = pd.DataFrame([house], columns=HOUSE_FEATURES, dtype=float)
        house = dict(zip(HOUSE_FEATURES, imputer.transform(incomplete)[0]))

    # Combine house attributes with demographics in the model's feature order
    input_data = pd.DataFrame([{**house, **demographic_info}], columns=model_features)
    logger.debug("Model input:\n%s", input_data)
    if input_data.isna().to_numpy().any():
        raise HTTPException(status_code=500, detail="Model input contains missing values")

    # Make prediction
    prediction = model.predict(input_data)

    return {"predicted_price": prediction[0]}
