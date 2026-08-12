import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, HTTPException, Query
try:
    from App.backend.schema import UserInput
    from App.backend.predict import predict_all
    from App.backend.recommendation import (
        recommend_places, get_all_districts, get_spots, get_nearby_amenities
    )
    from App.backend.database import save_trip_record, fetch_saved_trips
except ImportError:
    from schema import UserInput
    from predict import predict_all
    from recommendation import (
        recommend_places, get_all_districts, get_spots, get_nearby_amenities
    )
    from database import save_trip_record, fetch_saved_trips

app = FastAPI(
    title="AI Smart Tourism Recommendation API",
    version="1.0.0",
    description="Integrated Climate PyTorch LSTM, Visitor Crowd XGBoost, Budget Multi-Output Regressor & AI Spot Recommendation System"
)


def run_pipeline(user: UserInput) -> dict:
    """
    Executes master AI pipeline:
    1. Climate Prediction (PyTorch LSTM)
    2. Visitor Crowd Prediction (XGBoost)
    3. Itemized Budget Estimation (Multi-Output Regressor)
    4. Spot Recommendations & Nearby Amenities lookup
    5. Save record to DB
    """
    # 1. Run Machine Learning predictions
    predictions = predict_all(user)

    predicted_climate = predictions["predicted_climate"]
    predicted_visitors = predictions["predicted_visitors"]
    estimated_budget = predictions["estimated_budget"]
    budget_breakdown = predictions.get("budget_breakdown", {})
    season = predicted_climate["season"]

    # 2. Run Recommendation engine & get selected spot details
    recommendations = recommend_places(
        district=user.destination_district,
        category=user.category,
        season=season,
        budget=estimated_budget,
        crowd=predicted_visitors,
        transport=user.transport_mode
    )

    # Selected spot lat/lon lookup from dataset
    spot_name = user.spot_name or (recommendations[0]["spot_name"] if recommendations else None)
    if not spot_name:
        raise HTTPException(status_code=400, detail="No valid spot_name specified or found in dataset.")

    spots_found = get_spots(district=user.destination_district)
    selected_spot = next((s for s in spots_found if s["name"].lower() == spot_name.lower()), None)
    if not selected_spot:
        if spots_found:
            selected_spot = spots_found[0]
        else:
            raise HTTPException(status_code=404, detail=f"No tourist spots available in dataset for district '{user.destination_district}'.")

    # 3. Lookup Nearby Amenities
    nearby = get_nearby_amenities(
        spot_name=selected_spot["name"],
        district=user.destination_district,
        lat=selected_spot["lat"],
        lon=selected_spot["lon"]
    )

    result_payload = {
        "selected_spot": selected_spot,
        "predicted_climate": predicted_climate,
        "predicted_visitors": predicted_visitors,
        "estimated_budget": estimated_budget,
        "budget_breakdown": budget_breakdown,
        "recommendations": recommendations,
        "nearby_amenities": nearby
    }

    # 4. Save trip history
    trip_data = {
        "travel_date": user.travel_date,
        "spot_name": selected_spot["name"],
        "district": user.destination_district,
        "category": user.category,
        "season": season,
        "transport": user.transport_mode,
        "travelers": user.number_of_travelers,
        "days": user.number_of_days,
        "predicted_visitors": predicted_visitors,
        "estimated_budget": estimated_budget,
        "budget_breakdown": budget_breakdown,
        "weather_condition": predicted_climate.get("weather_condition", "Pleasant"),
        "temp_max": predicted_climate.get("temperature_max", 30.0),
        "temp_min": predicted_climate.get("temperature_min", 20.0),
        "rainfall": predicted_climate.get("rainfall_mm", 0.0),
        "recommendations": recommendations
    }
    save_trip_record(trip_data)

    return result_payload


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI Smart Tourism Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /api/predict",
            "recommend": "POST /api/recommend",
            "smart_trip": "POST /api/smart-trip",
            "districts": "GET /api/districts",
            "spots": "GET /api/spots",
            "amenities": "GET /api/amenities",
            "trips": "GET /api/trips"
        }
    }


@app.post("/api/predict")
def predict(user: UserInput):
    return run_pipeline(user)


@app.post("/api/recommend")
def recommend(user: UserInput):
    return run_pipeline(user)


@app.post("/api/smart-trip")
def smart_trip(user: UserInput):
    return run_pipeline(user)


@app.get("/api/districts")
def list_districts():
    return {"districts": get_all_districts()}


@app.get("/api/spots")
def list_spots(district: str = Query(None), category: str = Query(None)):
    return {"spots": get_spots(district=district, category=category)}


@app.get("/api/amenities")
def list_amenities(spot_name: str, district: str, lat: float = 17.3850, lon: float = 78.4867):
    return get_nearby_amenities(spot_name=spot_name, district=district, lat=lat, lon=lon)


@app.get("/api/trips")
def get_trips(limit: int = 50):
    return {"trips": fetch_saved_trips(limit=limit)}


@app.post("/api/save-trip")
def save_trip(trip_data: dict):
    success = save_trip_record(trip_data)
    return {"status": "success" if success else "failed"}

