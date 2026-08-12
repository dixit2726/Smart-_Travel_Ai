from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List, Dict, Any


class UserInput(BaseModel):
    source_location: Optional[str] = None
    destination_district: str
    travel_date: date = Field(default_factory=date.today)
    category: str
    budget: Optional[float] = None
    number_of_travelers: int = 1
    number_of_days: int = 1
    transport_mode: str
    accommodation_tier: str
    festival: Optional[str] = "None"
    spot_name: Optional[str] = None


# Alias for backward compatibility with app.py / legacy code
TravelRequest = UserInput


class ClimatePrediction(BaseModel):
    travel_date: str
    month: str
    season: str
    temperature_max: float
    temperature_min: float
    rainfall_mm: float
    weather_condition: str


class CrowdPrediction(BaseModel):
    predicted_visitors: int


class RecommendationSpot(BaseModel):
    spot_name: str
    district: str
    category: str
    season: str
    expected_crowd: int
    estimated_budget: float
    transport: str
    rating: float = 4.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    entry_fee: float = 0.0
    reviews: int = 0


class PredictionResponse(BaseModel):
    selected_spot: Optional[Dict[str, Any]] = None
    predicted_climate: ClimatePrediction
    predicted_visitors: int
    estimated_budget: float
    budget_breakdown: Optional[Dict[str, float]] = None
    recommendations: List[Dict[str, Any]]
    nearby_amenities: Optional[Dict[str, Any]] = None




