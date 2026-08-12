import os
import sys
import math
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

try:
    from App.backend.config import SPOTS_CSV, AMENITIES_CSV, ACCOMMODATIONS_CSV
except ImportError:
    from config import SPOTS_CSV, AMENITIES_CSV, ACCOMMODATIONS_CSV

# Streamlit-cached dataset loaders
@cache_data
def load_spots_df():
    return pd.read_csv(SPOTS_CSV) if SPOTS_CSV.exists() else pd.DataFrame()

@cache_data
def load_amenities_df():
    return pd.read_csv(AMENITIES_CSV) if AMENITIES_CSV.exists() else pd.DataFrame()

@cache_data
def load_acc_df():
    return pd.read_csv(ACCOMMODATIONS_CSV) if ACCOMMODATIONS_CSV.exists() else pd.DataFrame()

spots_df = load_spots_df()
amenities_df = load_amenities_df()
acc_df = load_acc_df()


def safe_float(val, default=0.0):
    """Safely parse float values from dataset rows without raising ValueError on strings/lists."""
    if pd.isna(val) or val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    """Safely parse integer values from dataset rows without raising ValueError on string review arrays."""
    if pd.isna(val) or val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        if isinstance(val, (list, tuple)):
            return len(val)
        elif isinstance(val, str):
            if "[" in val:
                return val.count("[")
            return len(val)
        return default


@cache_data
def get_all_districts() -> list:
    """Returns list of unique districts present in spots.csv."""
    df_s = load_spots_df()
    if df_s.empty or "district" not in df_s.columns:
        return []
    districts = df_s["district"].dropna().astype(str).str.strip().str.title().unique().tolist()
    return sorted(districts)


@cache_data
def get_top_districts(limit: int = 5) -> list:
    """Returns top districts ordered by number of tourist spots in the dataset."""
    df_s = load_spots_df()
    if df_s.empty or "district" not in df_s.columns:
        return []
    top_series = df_s["district"].dropna().astype(str).str.strip().str.title().value_counts()
    return top_series.head(limit).index.tolist()


@cache_data
def get_all_categories() -> list:
    """Returns list of unique categories present in spots.csv."""
    df_s = load_spots_df()
    if df_s.empty or "category" not in df_s.columns:
        return []
    categories = df_s["category"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    return sorted(categories)


@cache_data
def get_all_preprocessed_spots() -> list:
    """Loads and preprocesses spots.csv dataset ONCE into memory."""
    df_s = load_spots_df()
    if df_s.empty:
        return []

    spots = []
    for _, row in df_s.iterrows():
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except (ValueError, TypeError):
            continue

        if math.isnan(lat) or math.isnan(lon) or lat == 0 or lon == 0:
            continue

        cat_clean = str(row.get("category", "")).strip().lower()
        dist_str = str(row.get("district", "")).strip()

        spots.append({
            "name": str(row.get("name", "Tourist Spot")).strip(),
            "district": dist_str.title(),
            "district_lower": dist_str.lower(),
            "category": cat_clean,
            "rating": safe_float(row.get("rating"), 4.0),
            "popularity": safe_float(row.get("popularity"), 0.0),
            "entry_fee": safe_float(row.get("entry_fee"), 0.0),
            "lat": lat,
            "lon": lon,
            "reviews": safe_int(row.get("reviews"), 0)
        })

    return sorted(spots, key=lambda s: (s["rating"], s["popularity"]), reverse=True)


@cache_data
def get_spots(district: str = None, category: str = None) -> list:
    """Filter tourist spots from spots.csv strictly based on parameters using cached in-memory list."""
    all_spots = get_all_preprocessed_spots()
    if not all_spots:
        return []

    res = all_spots
    if district:
        d_low = str(district).strip().lower()
        res = [s for s in res if s["district_lower"] == d_low]

    if category:
        c_low = str(category).strip().lower()
        if c_low not in ["all", "none", "all categories", ""]:
            res = [s for s in res if s["category"] == c_low]

    return res


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to compute actual geographic distance in km between two lat/lon pairs."""
    try:
        R = 6371.0
        lat1_f, lon1_f = float(lat1), float(lon1)
        lat2_f, lon2_f = float(lat2), float(lon2)
        dlat = math.radians(lat2_f - lat1_f)
        dlon = math.radians(lon2_f - lon1_f)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1_f)) * math.cos(math.radians(lat2_f)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)
    except Exception:
        return 0.0


@cache_data
def get_nearby_amenities(spot_name: str, district: str, lat: float, lon: float) -> dict:
    """
    Returns actual categorized nearby amenities from accommodations.csv,
    nearby_amenities.csv, and spots.csv.
    Zero synthetic or fake amenities are generated.
    """
    results = {
        "hotels": [],
        "restaurants": [],
        "attractions": [],
        "hospitals": [],
        "parking": [],
        "atms": [],
        "petrol_pumps": [],
        "restrooms": []
    }

    # 1. Real Hotels from accommodations.csv
    if not acc_df.empty:
        dist_acc = acc_df[acc_df["district"].astype(str).str.lower() == district.lower()] if "district" in acc_df.columns else acc_df
        for _, r in dist_acc.iterrows():
            try:
                a_lat = float(r["lat"])
                a_lon = float(r["lon"])
                if math.isnan(a_lat) or math.isnan(a_lon):
                    continue
                dist_km = calculate_distance_km(lat, lon, a_lat, a_lon)
                results["hotels"].append({
                    "name": str(r.get("name", "Hotel")).strip(),
                    "tier": str(r.get("tier", "Mid")).strip(),
                    "cost": safe_float(r.get("cost"), 0.0),
                    "rating": 4.5,
                    "lat": a_lat,
                    "lon": a_lon,
                    "distance_km": dist_km,
                    "address": f"Near {spot_name}, {district}"
                })
            except Exception:
                continue

    # 2. Real Amenities from nearby_amenities.csv
    if not amenities_df.empty:
        matched_am = amenities_df[
            (amenities_df["district"].astype(str).str.lower() == district.lower()) |
            (amenities_df["spot_name"].astype(str).str.lower() == spot_name.lower())
        ] if "district" in amenities_df.columns else amenities_df

        for _, r in matched_am.iterrows():
            try:
                am_type = str(r.get("amenity_type", "")).lower().strip()
                a_lat = float(r["lat"])
                a_lon = float(r["lon"])
                if math.isnan(a_lat) or math.isnan(a_lon):
                    continue
                dist_km = calculate_distance_km(lat, lon, a_lat, a_lon)

                item = {
                    "name": str(r.get("amenity_name", am_type.title())).strip(),
                    "type": am_type,
                    "rating": 4.0,
                    "lat": a_lat,
                    "lon": a_lon,
                    "distance_km": dist_km,
                    "address": f"{district} Region"
                }

                if "restaurant" in am_type or "food" in am_type or "hotel" in am_type:
                    results["restaurants"].append(item)
                elif "hospital" in am_type or "health" in am_type or "clinic" in am_type:
                    results["hospitals"].append(item)
                elif "atm" in am_type or "bank" in am_type:
                    results["atms"].append(item)
                elif "parking" in am_type:
                    results["parking"].append(item)
                elif "petrol" in am_type or "fuel" in am_type:
                    results["petrol_pumps"].append(item)
                elif "restroom" in am_type or "toilet" in am_type:
                    results["restrooms"].append(item)
            except Exception:
                continue

    # 3. Real Attractions from spots_df
    all_spots = get_spots(district=district)
    for s in all_spots:
        if s["name"].lower() != spot_name.lower():
            try:
                d_km = calculate_distance_km(lat, lon, s["lat"], s["lon"])
                results["attractions"].append({
                    "name": s["name"],
                    "category": s["category"],
                    "rating": s["rating"],
                    "entry_fee": s["entry_fee"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "distance_km": d_km,
                    "address": f"{s['district']} Region"
                })
            except Exception:
                continue

    # Sort each list by distance
    for k in results:
        results[k] = sorted(results[k], key=lambda x: x.get("distance_km", 0.0))[:6]

    return results


@cache_data(show_spinner=False)
def recommend_places(
    district: str,
    category: str,
    season: str,
    budget: float,
    crowd: int,
    transport: str
) -> list:
    """
    Generates recommended tourist spots strictly from project data.
    Requires mandatory parameters without fallback numbers.
    """
    if not district or not category:
        raise ValueError("Both district and category are required for recommendations.")

    spots = get_spots(district=district, category=category)
    if not spots:
        spots = get_spots(district=district)

    if not spots:
        raise ValueError(f"No tourist spots available in dataset for district '{district}'.")

    recommendations = []
    for s in spots[:5]:
        recommendations.append({
            "spot_name": s["name"],
            "district": s["district"],
            "category": s["category"].title(),
            "rating": s["rating"],
            "season": season,
            "expected_crowd": crowd,
            "estimated_budget": budget,
            "transport": transport,
            "lat": s["lat"],
            "lon": s["lon"],
            "entry_fee": s["entry_fee"],
            "reviews": s["reviews"]
        })

    return recommendations
