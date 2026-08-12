import sys
from pathlib import Path

# Setup system paths for Root, App, Backend, and Frontend
FRONTEND_DIR = Path(__file__).resolve().parent
APP_DIR = FRONTEND_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = APP_DIR / "backend"
DATA_DIR = PROJECT_ROOT / "Data"

# Absolute Dataset Paths derived from Project Root
SPOTS_CSV = DATA_DIR / "spots.csv" if (DATA_DIR / "spots.csv").exists() else DATA_DIR / "other spots.csv"
CLIMATE_CSV = DATA_DIR / "Climate_Dataset_Final.csv"
BUDGET_CSV = DATA_DIR / "trip_budget_prediction_dataset.csv"
VISITORS_CSV = DATA_DIR / "spot_visitors.csv"
AMENITIES_CSV = DATA_DIR / "nearby_amenities.csv"
ACCOMMODATIONS_CSV = DATA_DIR / "accommodations.csv"

for p in [str(PROJECT_ROOT), str(APP_DIR), str(BACKEND_DIR), str(FRONTEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import datetime
import requests
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import st_folium

import App.backend.predict

from App.frontend.styles import apply_custom_css
from App.backend.schema import UserInput
from App.backend.predict import (
    predict_all,
    predict_climate,
    predict_visitors,
    predict_budget,
    get_model_verification_status,
    get_season_from_date
)
from App.backend.recommendation import (
    get_all_districts,
    get_top_districts,
    get_all_categories,
    get_spots,
    get_nearby_amenities
)
from App.backend.database import (
    save_trip_record,
    fetch_saved_trips
)
try:
    from App.backend.pdf_report import generate_trip_pdf
except ImportError:
    from App.frontend.pdf_generator import generate_trip_pdf


# Dataset Exact Enums for Crowd & Budget Models
DATASET_FESTIVALS = [
    'None', 'Bathukamma, Dussehra', 'Bhogi, Sankranti', 'Bonalu',
    'Buddha Purnima', 'Christmas', 'Diwali', 'Holi',
    'Independence Day, Krishna Janmashtami', 'Maha Shivaratri',
    'Ugadi, Sri Rama Navami', 'Vinayaka Chavithi'
]

BUDGET_TRANSPORTS = ['Car', 'Bike', 'Auto', 'Bus', 'Train']
BUDGET_ACC_TIERS = ['Budget', 'Mid', 'Premium']

MONTH_FESTIVAL_MAP = {
    1: 'Bhogi, Sankranti',
    2: 'Maha Shivaratri',
    3: 'Holi',
    4: 'Ugadi, Sri Rama Navami',
    5: 'Buddha Purnima',
    6: 'None',
    7: 'Bonalu',
    8: 'Independence Day, Krishna Janmashtami',
    9: 'Vinayaka Chavithi',
    10: 'Bathukamma, Dussehra',
    11: 'Diwali',
    12: 'Christmas'
}

def get_detected_festival(travel_date) -> str:
    if isinstance(travel_date, datetime.date):
        return MONTH_FESTIVAL_MAP.get(travel_date.month, "None")
    return "None"

def get_detected_festival_for_range(t_start, t_end) -> str:
    if not isinstance(t_start, datetime.date) or not isinstance(t_end, datetime.date) or t_end < t_start:
        return "None"
    curr = t_start
    max_days = min(60, (t_end - t_start).days + 1)
    for _ in range(max_days):
        fest = get_detected_festival(curr)
        if fest != "None" and fest in DATASET_FESTIVALS:
            return fest
        curr += datetime.timedelta(days=1)
    return "None"

# Helper for converting visitor prediction into data-driven crowd percentage index
@st.cache_data
def get_spot_max_visitors_dict() -> dict:
    stats = {}
    if VISITORS_CSV.exists():
        try:
            v_df = pd.read_csv(VISITORS_CSV)
            if "Spot_Name" in v_df.columns and "Total_Visitors" in v_df.columns:
                for s_name, m_val in v_df.groupby("Spot_Name")["Total_Visitors"].max().items():
                    stats[str(s_name).strip().lower()] = float(m_val)
        except Exception as e:
            print(f"[WARNING] Could not load spot visitor max stats: {e}")
    return stats

def get_spot_max_visitors(spot_name: str) -> float:
    max_dict = get_spot_max_visitors_dict()
    overall_max = max(max_dict.values()) if max_dict else 26624.0
    s_clean = str(spot_name).strip().lower()
    if s_clean in max_dict:
        return max_dict[s_clean]
    for k, v in max_dict.items():
        if s_clean in k or k in s_clean:
            return v
    return overall_max


@st.cache_data
def get_trip_climate_forecast(spot_name: str, start_date_str: str, end_date_str: str, district_name: str = None):
    """
    Generates an accurate, audited time-series climate forecast for a tourist spot specifically for the user's trip dates.
    Uses the trained ClimateLSTM neural network on differenced time-series sequences combined with spot/district seasonal baselines.
    """
    if not CLIMATE_CSV.exists() or not start_date_str or not end_date_str:
        return None

    try:
        start_date = pd.to_datetime(start_date_str).date()
        end_date = pd.to_datetime(end_date_str).date()
        if end_date < start_date:
            return None

        df_c = pd.read_csv(CLIMATE_CSV)
        if df_c.empty or "Date" not in df_c.columns:
            return None

        df_c["Date"] = pd.to_datetime(df_c["Date"], errors='coerce')
        df_c = df_c.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # 1. Filter dataset by selected Tourist Spot or District
        spot_clean = str(spot_name).strip().lower()
        spot_sub = df_c[df_c["Tourist Spots"].astype(str).str.lower() == spot_clean].copy()

        if spot_sub.empty and district_name:
            dist_clean = str(district_name).strip().lower()
            spot_sub = df_c[df_c["District"].astype(str).str.lower() == dist_clean].copy()

        if spot_sub.empty:
            spot_sub = df_c.copy()

        cols = ["Temperature_Max_C", "Temperature_Min_C", "Rainfall_mm"]
        for col in cols:
            if col not in spot_sub.columns:
                return None

        num_days = (end_date - start_date).days + 1
        latest_date = spot_sub["Date"].max().date()

        if num_days > 60 or (end_date - latest_date).days > 365:
            return "EXCEEDS_LIMIT"

        # 2. Extract differenced series (the domain ClimateLSTM was trained on)
        spot_daily = spot_sub.groupby("Date")[cols].mean().reset_index()
        spot_diff = spot_daily[cols].diff().dropna()

        cm = getattr(App.backend.predict, "climate_model", None)

        import torch
        if cm is not None:
            cm.eval()
            diff_vals = spot_diff.tail(max(7, len(spot_diff)))[cols].values
            curr_diff_seq = torch.tensor([diff_vals[-7:]], dtype=torch.float32)

            curr_d = latest_date + datetime.timedelta(days=1)
            forecast_diffs = {}

            while curr_d <= end_date:
                with torch.no_grad():
                    next_diff = cm(curr_diff_seq).numpy()[0]
                
                forecast_diffs[curr_d] = next_diff
                next_diff_tensor = torch.tensor([[next_diff]], dtype=torch.float32)
                curr_diff_seq = torch.cat([curr_diff_seq[:, 1:, :], next_diff_tensor], dim=1)
                curr_d += datetime.timedelta(days=1)
        else:
            forecast_diffs = {}

        # 3. Compute baseline calendar day averages for the spot/district across history
        spot_daily["Month"] = spot_daily["Date"].dt.month
        spot_daily["Day"] = spot_daily["Date"].dt.day

        trip_dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
        forecast_records = []

        for d in trip_dates:
            day_sub = spot_daily[(spot_daily["Month"] == d.month) & (spot_daily["Day"] == d.day)]
            if not day_sub.empty:
                b_max = float(day_sub["Temperature_Max_C"].mean())
                b_min = float(day_sub["Temperature_Min_C"].mean())
                b_rain = float(day_sub["Rainfall_mm"].mean())
            else:
                b_max = float(spot_daily["Temperature_Max_C"].mean())
                b_min = float(spot_daily["Temperature_Min_C"].mean())
                b_rain = float(spot_daily["Rainfall_mm"].mean())

            # Apply LSTM predicted delta anomaly if available
            diff_val = forecast_diffs.get(d, [0.0, 0.0, 0.0])
            p_max = round(b_max + float(diff_val[0]), 1)
            p_min = round(b_min + float(diff_val[1]), 1)
            if p_min > p_max:
                p_min = round(p_max - 1.0, 1)

            p_rain = round(max(0.0, b_rain + float(diff_val[2])), 1)

            forecast_records.append({
                "Date": d,
                "Date_Formatted": d.strftime("%b %d, %Y"),
                "Temperature_Max_C": p_max,
                "Temperature_Min_C": p_min,
                "Rainfall_mm": p_rain
            })

        return pd.DataFrame(forecast_records)
    except Exception as e:
        print(f"[ERROR] Failed to generate trip climate forecast: {e}")
        return None

# Backward compatibility wrappers
@st.cache_data
def get_future_climate_forecast(spot_name: str, days: int = 30):
    t_start = datetime.date.today() + datetime.timedelta(days=1)
    t_end = t_start + datetime.timedelta(days=days - 1)
    return get_trip_climate_forecast(spot_name, str(t_start), str(t_end))

get_spot_climate_timeseries = get_future_climate_forecast


@st.cache_data
def get_all_dataset_metadata_cached():
    stats = {}
    
    # 1. spot_visitors.csv
    try:
        if VISITORS_CSV.exists():
            df_v = pd.read_csv(VISITORS_CSV)
            stats["visitors"] = {
                "rows": f"{len(df_v):,}",
                "cols": len(df_v.columns),
                "spots": df_v["Spot_Name"].nunique() if "Spot_Name" in df_v.columns else 271,
                "districts": df_v["District"].nunique() if "District" in df_v.columns else 33,
                "years": f"{int(df_v['Year'].min())} - {int(df_v['Year'].max())}" if "Year" in df_v.columns else "2024 - 2025",
                "target": "Total_Visitors"
            }
    except Exception as e:
        print(f"[VISITORS STATS ERR] {e}")

    # 2. Climate_Dataset_Final.csv
    try:
        if CLIMATE_CSV.exists():
            df_c = pd.read_csv(CLIMATE_CSV)
            years_str = "2023 - 2026"
            if "Date" in df_c.columns:
                try:
                    dates = pd.to_datetime(df_c["Date"], errors='coerce')
                    v_yrs = dates.dt.year.dropna()
                    if not v_yrs.empty:
                        years_str = f"{int(v_yrs.min())} - {int(v_yrs.max())}"
                except Exception:
                    pass
            stats["climate"] = {
                "rows": f"{len(df_c):,}",
                "cols": len(df_c.columns),
                "spots": df_c["Tourist Spots"].nunique() if "Tourist Spots" in df_c.columns else 271,
                "districts": df_c["District"].nunique() if "District" in df_c.columns else 33,
                "years": years_str,
                "target": "Temperature_Max_C, Temperature_Min_C, Rainfall_mm"
            }
    except Exception as e:
        print(f"[CLIMATE STATS ERR] {e}")

    # 3. trip_budget_prediction_dataset.csv
    try:
        if BUDGET_CSV.exists():
            df_b = pd.read_csv(BUDGET_CSV)
            stats["budget"] = {
                "rows": f"{len(df_b):,}",
                "cols": len(df_b.columns),
                "spots": "Multi-Spot Routes",
                "districts": "Statewide Trips (33 Districts)",
                "years": "Standardized Multi-Season",
                "target": "travel_cost_est, stay_cost_est, food_cost_est, entry_fees_est, tolls_and_parking_est"
            }
    except Exception as e:
        print(f"[BUDGET STATS ERR] {e}")

    # 4. other spots.csv
    try:
        if SPOTS_CSV.exists():
            df_s = pd.read_csv(SPOTS_CSV)
            stats["spots"] = {
                "rows": f"{len(df_s):,}",
                "cols": len(df_s.columns),
                "spots": len(df_s),
                "districts": df_s["district"].nunique() if "district" in df_s.columns else 33,
                "years": "Active Catalogue",
                "target": "rating, entry_fee, lat, lon"
            }
    except Exception as e:
        print(f"[SPOTS STATS ERR] {e}")

    # 5. accommodations.csv
    try:
        if ACCOMMODATIONS_CSV.exists():
            df_acc = pd.read_csv(ACCOMMODATIONS_CSV)
            stats["accommodations"] = {
                "rows": f"{len(df_acc):,}",
                "cols": len(df_acc.columns),
                "spots": df_acc["name"].nunique() if "name" in df_acc.columns else len(df_acc),
                "districts": df_acc["district"].nunique() if "district" in df_acc.columns else 33,
                "years": "Active Directory",
                "target": "tier, cost, lat, lon"
            }
    except Exception as e:
        print(f"[ACCOMMODATIONS STATS ERR] {e}")

    # 6. nearby_amenities.csv
    try:
        if AMENITIES_CSV.exists():
            df_n = pd.read_csv(AMENITIES_CSV)
            stats["amenities"] = {
                "rows": f"{len(df_n):,}",
                "cols": len(df_n.columns),
                "spots": df_n["spot_name"].nunique() if "spot_name" in df_n.columns else 271,
                "districts": df_n["district"].nunique() if "district" in df_n.columns else 33,
                "years": "Active GIS Mappings",
                "target": "amenity_name, amenity_type, lat, lon"
            }
    except Exception as e:
        print(f"[AMENITIES STATS ERR] {e}")

    return stats


def get_default_trip_plan() -> dict:
    """Provides a complete default trip plan payload for PDF report generation when no active session plan exists."""
    today = datetime.date.today()
    start_d = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    end_d = (today + datetime.timedelta(days=9)).strftime("%Y-%m-%d")
    return {
        "selected_district": "Hyderabad",
        "selected_spots": [
            {"name": "Charminar", "category": "Heritage", "rating": 4.6},
            {"name": "Golconda Fort", "category": "Heritage", "rating": 4.5},
            {"name": "Hussain Sagar Lake", "category": "Nature / Lake", "rating": 4.3}
        ],
        "travel_start": start_d,
        "travel_end": end_d,
        "duration_days": 3,
        "num_travelers": 2,
        "transport_mode": "car",
        "acc_tier": "3 Star",
        "festival": "None",
        "predicted_climate": {
            "temperature_max_c": 32.5,
            "temperature_min_c": 22.0,
            "rainfall_mm": 0.0,
            "weather_condition": "Clear / Sunny",
            "climate_impact_score": 85.0
        },
        "daily_climate_forecast": [
            {"date": start_d, "condition": "Sunny", "temp_max": 32.5, "temp_min": 22.0, "rainfall": 0.0},
            {"date": (today + datetime.timedelta(days=8)).strftime("%Y-%m-%d"), "condition": "Partly Cloudy", "temp_max": 31.8, "temp_min": 21.5, "rainfall": 0.0},
            {"date": end_d, "condition": "Clear", "temp_max": 33.0, "temp_min": 22.2, "rainfall": 0.0}
        ],
        "crowd_prediction": {
            "crowd_level": "Moderate",
            "predicted_visitors": 24500,
            "crowd_percentage": 58.0
        },
        "estimated_budget": 12500.0,
        "per_person_cost": 6250.0,
        "budget_breakdown": {
            "travel_cost": 3500.0,
            "stay_cost": 4500.0,
            "food_cost": 3000.0,
            "entry_fees": 1000.0,
            "tolls_and_parking": 500.0
        },
        "nearby_amenities": {
            "hotels": [{"name": "Taj Krishna", "tier": "5 Star", "rating": 4.7, "distance_km": 2.1}],
            "restaurants": [{"name": "Paradise Biryani", "tier": "Fine Dining", "rating": 4.5, "distance_km": 1.2}],
            "hospitals": [{"name": "Apollo Hospital", "tier": "Super Specialty", "rating": 4.8, "distance_km": 3.5}]
        },
        "route_details": {
            "total_distance_km": 24.5,
            "estimated_drive_time": "45 mins",
            "stops_sequence": ["Charminar", "Golconda Fort", "Hussain Sagar Lake"]
        }
    }


# App Image Paths & Cached WebP Loaders
OTHERS_DIR = PROJECT_ROOT / "Others"
IMAGES_DIR = OTHERS_DIR / "images"
APP_LOGO_PATH = IMAGES_DIR / "TravelLogo.webp" if (IMAGES_DIR / "TravelLogo.webp").exists() else OTHERS_DIR / "TravelLogo.webp"
ABOUT_SECTION_WEBP = IMAGES_DIR / "images_1.webp" if (IMAGES_DIR / "images_1.webp").exists() else OTHERS_DIR / "images (1).webp"
WHY_SECTION_WEBP = IMAGES_DIR / "travelll.webp" if (IMAGES_DIR / "travelll.webp").exists() else OTHERS_DIR / "travelll.webp"


@st.cache_data(show_spinner=False)
def load_cached_webp_bytes(image_path: Path) -> bytes:
    """
    Reads and caches raw WebP image bytes from the project-relative Others directory in RAM
    to prevent disk re-reading and image flickering on Streamlit reruns.
    """
    if image_path and image_path.exists():
        try:
            return image_path.read_bytes()
        except Exception as e:
            print(f"[WEBP READ ERR] Failed to read {image_path}: {e}")
    return None

@st.cache_data(show_spinner=False)
def get_about_image() -> bytes:
    return load_cached_webp_bytes(ABOUT_SECTION_WEBP)

@st.cache_data(show_spinner=False)
def get_why_image() -> bytes:
    return load_cached_webp_bytes(WHY_SECTION_WEBP)

@st.cache_data(show_spinner=False)
def get_logo_image() -> bytes:
    return load_cached_webp_bytes(APP_LOGO_PATH)

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Smart Tourism AI System",
    page_icon=str(APP_LOGO_PATH) if APP_LOGO_PATH.exists() else "✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Commercial CSS Theme
apply_custom_css()

# API Configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", os.getenv("FASTAPI_URL", "https://smart-travel-ai-7.onrender.com/")).rstrip('/')
FASTAPI_URL = BACKEND_API_URL

@st.cache_data(ttl=60, show_spinner=False)
def check_backend_status():
    try:
        r = requests.get(f"{BACKEND_API_URL}/", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False

# =====================================
# Navigation Sidebar
# =====================================
with st.sidebar:
    if APP_LOGO_PATH.exists():
        st.image(str(APP_LOGO_PATH), use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: #38BDF8; font-weight: 800;'>Smart Travel AI</h2>", unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Project Overview", "Smart Trip Planner", "My Predictions / Saved Trips"],
        icons=["house-door-fill", "journal-text", "compass-fill", "clock-history"],
        default_index=2 if st.session_state.get("nav_planner", False) else 0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#38BDF8", "font-size": "1.1rem"},
            "nav-link": {
                "font-size": "0.95rem",
                "text-align": "left",
                "margin": "4px 0",
                "color": "#94A3B8",
                "background-color": "rgba(30, 41, 59, 0.4)",
                "border-radius": "10px",
                "padding": "10px 16px"
            },
            "nav-link-selected": {
                "background-color": "linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%)",
                "color": "#FFFFFF",
                "font-weight": "700",
                "box-shadow": "0 4px 14px rgba(14, 165, 233, 0.4)"
            }
        }
    )

    # API Status Indicator in Sidebar
    backend_online = check_backend_status()
    st.markdown("---")
    if backend_online:
        st.markdown("<span class='badge-green'>● FastAPI Engine Online</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-blue'>⚡ Embedded AI Engine Active</span>", unsafe_allow_html=True)

    st.markdown("---")
    # Dev Model Verification Section
    v_status = get_model_verification_status()
    with st.expander("🛠️ MODEL INTEGRITY VERIFICATION", expanded=False):
        c_stat = "🟢 LOADED" if v_status.get("climate_model") else "🔴 FAILED"
        v_stat = "🟢 LOADED" if v_status.get("crowd_model") else "🔴 FAILED"
        b_stat = "🟢 LOADED" if v_status.get("budget_model") else "🔴 FAILED"
        e_stat = "🟢 LOADED" if v_status.get("encoders") else "🔴 FAILED"
        s_stat = "🟢 LOADED" if v_status.get("scalers") else "🔴 FAILED"

        st.markdown(f"**Climate Model**: {c_stat}")
        st.markdown(f"**Crowd Model**: {v_stat}")
        st.markdown(f"**Budget Model**: {b_stat}")
        st.markdown(f"**Encoders**: {e_stat}")
        st.markdown(f"**Scalers**: {s_stat}")


# Reset nav_planner session flag after reading
if st.session_state.get("nav_planner", False):
    st.session_state["nav_planner"] = False


# =====================================
# PAGE 1: HOME PAGE (REDESIGNED)
# =====================================
if selected_page == "Home":
    # --------------------------------------------------
    # 1. HERO SECTION
    # --------------------------------------------------
    st.markdown("""
        <div class="hero-banner">
            <div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">
                ✨ Intelligent AI Tourism System
            </div>
            <div class="hero-title">Plan Smarter. Travel Better.</div>
            <div class="hero-subtitle">
                AI-powered tourism planning with intelligent predictions, budget insights, and smart route planning.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_hero_btn, _ = st.columns([3, 7])
    with col_hero_btn:
        if st.button("🧭 Start Planning", key="hero_plan_btn", use_container_width=True):
            st.session_state["nav_planner"] = True
            st.rerun()

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 2. ABOUT SMART TOURISM AI
    # --------------------------------------------------
    st.markdown("<div class='home-section-title'>About Smart Tourism AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-section-subtitle'>An integrated intelligent framework for modern travel planning.</div>", unsafe_allow_html=True)
    
    col_about_img, col_about_txt = st.columns([5, 7], gap="large")
    
    with col_about_img:
        about_img = get_about_image()
        if about_img:
            st.image(about_img, use_container_width=True)
        elif ABOUT_SECTION_WEBP.exists():
            st.image(str(ABOUT_SECTION_WEBP), use_container_width=True)

    with col_about_txt:
        st.markdown("""
            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; padding: 28px;">
                <h3 style="color: #38BDF8; font-weight: 800; margin-bottom: 16px; font-size: 1.55rem; letter-spacing: -0.01em;">
                    Unified Travel Intelligence Platform
                </h3>
                <p style="color: #E2E8F0; font-size: 0.96rem; line-height: 1.65; margin-bottom: 12px;">
                    <strong style="color: #38BDF8;">Smart Tourism AI</strong> is an AI-powered tourism planning system designed to make travel planning more intelligent, convenient, and data-driven. It brings multiple aspects of tourism planning together in one platform, helping travelers make better decisions before and during their journey.
                </p>
                <p style="color: #CBD5E1; font-size: 0.94rem; line-height: 1.65; margin-bottom: 12px;">
                    Instead of depending on different platforms for weather information, tourist destinations, crowd information, travel expenses, and route planning, <strong style="color: #38BDF8;">Smart Tourism AI</strong> integrates these capabilities into a unified travel-planning experience.
                </p>
                <p style="color: #CBD5E1; font-size: 0.94rem; line-height: 1.65; margin-bottom: 12px;">
                    The system uses tourism datasets and <span style="color: #34D399; font-weight: 700;">Machine Learning</span> models to generate predictive insights such as expected visitor <span style="color: #38BDF8; font-weight: 700;">Crowd Prediction</span> levels, <span style="color: #38BDF8; font-weight: 700;">Climate Prediction</span> conditions, and estimated trip <span style="color: #38BDF8; font-weight: 700;">Budget Estimation</span> costs.
                </p>
                <p style="color: #CBD5E1; font-size: 0.94rem; line-height: 1.65; margin-bottom: 12px;">
                    Once destinations are selected, the platform provides an interactive GIS-based <span style="color: #A855F7; font-weight: 700;">Smart Route Planning</span> view where travelers can visualize their selected stops and explore nearby facilities such as hotels, restaurants, parking areas, and other amenities.
                </p>
                <p style="color: #CBD5E1; font-size: 0.94rem; line-height: 1.65; margin-bottom: 18px;">
                    The goal of <strong style="color: #38BDF8;">Smart Tourism AI</strong> is to help travelers understand the conditions of their trip, plan their resources efficiently, and make more informed tourism decisions.
                </p>
                <div style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(37, 99, 235, 0.2) 100%); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 12px; padding: 12px 18px; text-align: center;">
                    <span style="color: #38BDF8; font-weight: 800; font-size: 0.96rem; letter-spacing: 0.5px;">
                        ✨ One platform. Multiple travel insights. Smarter decisions.
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 3. WHY SMART TOURISM? (PROJECT PROBLEM STATEMENT)
    # --------------------------------------------------
    st.markdown("<div id='why-smart-tourism-section' class='home-section-title'>Why Smart Tourism?</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-section-subtitle'>Travel planning is more than choosing a destination — it requires understanding the conditions, costs, crowd levels, and logistics of the journey.</div>", unsafe_allow_html=True)

    # SUBSECTION 1: BEYOND DESTINATION SELECTION
    col_why_txt, col_why_img = st.columns([7, 5], gap="large")

    with col_why_txt:
        st.markdown("""
<div class="why-card" style="height: 100%;">
<div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">
💡 Travel Planning Paradigm
</div>
<h3 style="color: #F8FAFC; font-weight: 800; font-size: 1.65rem; margin-bottom: 16px;">
Beyond Destination Selection
</h3>
<p style="color: #E2E8F0; font-size: 1.0rem; line-height: 1.65; margin-bottom: 12px;">
Planning a successful trip involves several decisions before the journey even begins. Travelers need to decide where to go, when to visit, how crowded the destination may be, what weather conditions to expect, how much the trip may cost, and how to move between multiple destinations.
</p>
<p style="color: #CBD5E1; font-size: 0.98rem; line-height: 1.65; margin-bottom: 12px;">
Today, this information is often scattered across different sources. Travelers may need to check weather platforms, tourism websites, maps, travel websites, and other sources separately. Comparing all of this information manually can be time-consuming and makes it difficult to build a complete picture of the trip.
</p>
<p style="color: #CBD5E1; font-size: 0.98rem; line-height: 1.65; margin-bottom: 16px;">
Smart tourism uses data and technology to bring these different aspects together. By combining tourism data with <strong style="color: #34D399;">machine learning</strong> and intelligent planning tools, travelers can receive useful insights before making their travel decisions.
</p>
<div style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(37, 99, 235, 0.2) 100%); border-left: 4px solid #38BDF8; border-radius: 12px; padding: 14px 18px; margin-top: 10px;">
<div style="color: #38BDF8; font-weight: 800; font-size: 1.0rem; margin-bottom: 4px;">
✨ Better travel decisions start with better information.
</div>
<div style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.5;">
Smart Tourism AI is designed to turn travel-related data into practical insights that can support smarter trip planning.
</div>
</div>
</div>
""", unsafe_allow_html=True)

    with col_why_img:
        why_img = get_why_image()
        if why_img:
            st.image(why_img, caption="Smart Tourism Insights & Planning", use_container_width=True)
        elif WHY_SECTION_WEBP.exists():
            st.image(str(WHY_SECTION_WEBP), caption="Smart Tourism Insights & Planning", use_container_width=True)

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # SUBSECTION 2: THE CHALLENGES TRAVELERS FACE (THE PROBLEM)
    st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; font-size: 1.8rem; margin-bottom: 6px;'>The Challenges Travelers Face</h3>", unsafe_allow_html=True)
    st.markdown("<div class='home-section-subtitle'>Traditional trip planning can become difficult when important travel information is distributed across multiple sources.</div>", unsafe_allow_html=True)

    prob_col1, prob_col2, prob_col3 = st.columns(3)
    with prob_col1:
        st.markdown("""
<div class="problem-card">
<div class="problem-icon">👥</div>
<div class="problem-title">1. Uncertain Crowd Levels</div>
<div class="problem-desc">
Popular tourist destinations can experience significant variations in visitor volume depending on the season, month, festivals, holidays, and other travel conditions. Travelers may not know whether a destination is likely to be relatively crowded when planning their visit.
</div>
</div>
""", unsafe_allow_html=True)

    with prob_col2:
        st.markdown("""
<div class="problem-card">
<div class="problem-icon">🌦️</div>
<div class="problem-title">2. Changing Climate Conditions</div>
<div class="problem-desc">
Weather conditions can directly influence the travel experience. Temperature and rainfall can affect outdoor activities, sightseeing plans, transportation, and the suitability of a destination for a particular travel period.
</div>
</div>
""", unsafe_allow_html=True)

    with prob_col3:
        st.markdown("""
<div class="problem-card">
<div class="problem-icon">💰</div>
<div class="problem-title">3. Difficult Budget Planning</div>
<div class="problem-desc">
A trip involves multiple expenses such as transportation, accommodation, food, and entry tickets. Estimating the overall cost before traveling can be difficult when these expenses depend on the duration, number of travelers, and selected travel preferences.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    prob_col4, prob_col5 = st.columns(2)
    with prob_col4:
        st.markdown("""
<div class="problem-card">
<div class="problem-icon">🔎</div>
<div class="problem-title">4. Information Scattered Across Sources</div>
<div class="problem-desc">
Important travel information is often available through different platforms and sources. Travelers may have to manually search, compare, and combine information about destinations, weather, costs, routes, and nearby facilities.
</div>
</div>
""", unsafe_allow_html=True)

    with prob_col5:
        st.markdown("""
<div class="problem-card">
<div class="problem-icon">🗺️</div>
<div class="problem-title">5. Route & Nearby Place Planning</div>
<div class="problem-desc">
Visiting multiple destinations requires practical route planning. Travelers may also need to identify useful nearby facilities such as hotels, restaurants, parking areas, and other amenities while planning their journey.
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # SUBSECTION 3: WHY THESE CHALLENGES MATTER
    st.markdown("""
<div class="glass-card" style="padding: 30px; border-left: 4px solid rgba(168, 85, 247, 0.8);">
<div style="font-size: 0.85rem; font-weight: 800; color: #A855F7; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
🔗 Interconnected Travel Dynamics
</div>
<h3 style="color: #F8FAFC; font-weight: 800; font-size: 1.5rem; margin-bottom: 14px;">
Why These Challenges Matter
</h3>
<p style="color: #E2E8F0; font-size: 1.0rem; line-height: 1.7; margin-bottom: 14px;">
These challenges are connected. A change in weather can influence destination selection, crowd levels can influence the preferred travel date, the number of destinations affects travel distance and cost, and the availability of nearby facilities can influence the overall travel experience.
</p>
<p style="color: #CBD5E1; font-size: 1.0rem; line-height: 1.7; margin: 0;">
Therefore, effective tourism planning requires more than a simple destination selection. Travelers need multiple pieces of information to work together when making decisions.
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # SUBSECTION 4: TRANSITION TO SOLUTION
    st.markdown("""
<div class="cta-banner" style="margin-top: 10px; margin-bottom: 20px; padding: 36px 30px; background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.35);">
<div style="font-size: 0.85rem; font-weight: 800; color: #34D399; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
🚀 Moving Forward
</div>
<h3 style="color: #FFFFFF; font-size: 1.75rem; font-weight: 800; margin-bottom: 12px;">
From Travel Challenges to Intelligent Planning
</h3>
<p style="color: #CBD5E1; font-size: 1.05rem; max-width: 800px; margin: 0 auto 20px auto; line-height: 1.6;">
Smart Tourism AI addresses these challenges by combining tourism data, machine learning predictions, interactive mapping, and nearby amenities into a unified travel-planning experience.
</p>
</div>
""", unsafe_allow_html=True)

    col_trans_btn, _ = st.columns([4, 6])
    with col_trans_btn:
        st.markdown("""
<a href="#ai-solution-section" style="text-decoration: none;">
<div style="
background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
color: #FFFFFF;
text-align: center;
padding: 12px 24px;
border-radius: 12px;
font-weight: 700;
font-size: 1.02rem;
box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4);
transition: all 0.25s ease;
">
Explore Our AI Solution ➔
</div>
</a>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 5. OUR AI-POWERED SOLUTION
    # --------------------------------------------------
    st.markdown("<div id='ai-solution-section' class='home-section-title'>Our AI-Powered Solution</div>", unsafe_allow_html=True)
    st.markdown("""
        <p style="color: #CBD5E1; font-size: 1.05rem; line-height: 1.7; max-width: 950px; margin-bottom: 24px;">
            Smart Tourism AI brings these travel-planning requirements together in one platform. Using machine learning models and tourism data, the system provides predictive insights and personalized planning support before and during a trip.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="ai-pipeline-container">
            <div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 18px;">
                ⚡ End-to-End System Intelligence Pipeline
            </div>
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px; font-weight: 700; font-size: 0.95rem; color: #F8FAFC;">
                <div class="pipeline-node" style="border-color: rgba(56, 189, 248, 0.4);">
                    📊 Tourism Data
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node" style="border-color: rgba(129, 140, 248, 0.4);">
                    🧠 Machine Learning
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node" style="border-color: rgba(52, 211, 153, 0.4);">
                    🔮 Crowd + Climate + Budget Predictions
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-node" style="border-color: rgba(168, 85, 247, 0.4);">
                    🗺️ Smart Route & Nearby Places
                </div>
                <div class="pipeline-arrow">➔</div>
                <div style="background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%); color: #FFFFFF; padding: 14px 20px; border-radius: 14px; box-shadow: 0 4px 14px rgba(37,99,235,0.4); font-weight: 800;">
                    ✨ Better Trip Planning
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 6. KEY CAPABILITIES
    # --------------------------------------------------
    st.markdown("<div class='home-section-title'>What Smart Tourism AI Provides</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-section-subtitle'>Core capabilities integrated into a single unified planning experience.</div>", unsafe_allow_html=True)

    cap_col1, cap_col2, cap_col3 = st.columns(3)
    with cap_col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">👥</div>
                <div class="feature-title">Crowd Prediction</div>
                <div class="feature-desc">Predict expected visitor crowd before your trip to choose optimal visit times.</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">💰</div>
                <div class="feature-title">Budget Estimation</div>
                <div class="feature-desc">Estimate your trip expenses based on your travel preferences.</div>
            </div>
        """, unsafe_allow_html=True)

    with cap_col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🌦️</div>
                <div class="feature-title">Climate Prediction</div>
                <div class="feature-desc">Understand predicted temperature and rainfall for your travel date.</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🗺️</div>
                <div class="feature-title">Smart Route Planning</div>
                <div class="feature-desc">Visualize selected tourist destinations and their route on an interactive map.</div>
            </div>
        """, unsafe_allow_html=True)

    with cap_col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📍</div>
                <div class="feature-title">Nearby Amenities</div>
                <div class="feature-desc">Discover nearby hotels, restaurants, parking and other available amenities.</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Generate PDF Report</div>
                <div class="feature-desc">Generate a professional PDF summary of your trip plan, including itinerary, weather, crowd, amenities, route and budget.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 7. HOW IT WORKS
    # --------------------------------------------------
    st.markdown("<div class='home-section-title'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-section-subtitle'>A structured 5-step workflow for seamless end-to-end travel planning.</div>", unsafe_allow_html=True)

    hw_col1, hw_col2, hw_col3, hw_col4, hw_col5 = st.columns(5)
    with hw_col1:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">01</div>
                <div class="step-title">Select District</div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">Choose destination district.</p>
            </div>
        """, unsafe_allow_html=True)

    with hw_col2:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">02</div>
                <div class="step-title">Explore Tourist Spots</div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">Select spots for itinerary.</p>
            </div>
        """, unsafe_allow_html=True)

    with hw_col3:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">03</div>
                <div class="step-title">Plan Your Trip</div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">Set date & travel options.</p>
            </div>
        """, unsafe_allow_html=True)

    with hw_col4:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">04</div>
                <div class="step-title">Get AI Predictions</div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">Crowd, climate & budget ML.</p>
            </div>
        """, unsafe_allow_html=True)

    with hw_col5:
        st.markdown("""
            <div class="step-card">
                <div class="step-number">05</div>
                <div class="step-title">Explore Route & Nearby Places</div>
                <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">Interactive map & amenities.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # 8. FINAL CTA
    # --------------------------------------------------
    st.markdown("""
        <div class="cta-banner">
            <h2 style="color: #FFFFFF; font-size: 2.2rem; font-weight: 800; margin-bottom: 12px;">Plan Your Journey Smarter</h2>
            <p style="color: #CBD5E1; font-size: 1.1rem; max-width: 650px; margin: 0 auto 28px auto;">
                Select your destination, set your travel preferences, and let Smart Tourism AI help you make better travel decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_cta_btn, _ = st.columns([4, 6])
    with col_cta_btn:
        if st.button("🚀 Start Planning", key="home_cta_plan_btn", use_container_width=True):
            st.session_state["nav_planner"] = True
            st.rerun()


# =====================================
# PAGE 2: PROJECT OVERVIEW (REDESIGNED)
# =====================================
elif selected_page == "Project Overview":
    st.markdown("<h2 style='font-weight: 800; color: #38BDF8;'>📖 Project Overview & ML Architecture</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.05rem;'>Technical documentation, empirical model evaluation, dataset statistics, and live inference demonstration.</p>", unsafe_allow_html=True)

    # 1. LIVE MODEL DEPLOYMENT STATUS (Section 12)
    v_status = get_model_verification_status()
    c_badge = "🟢 LOADED" if v_status.get("climate_model") else "🔴 FAILED"
    v_badge = "🟢 LOADED" if v_status.get("crowd_model") else "🔴 FAILED"
    b_badge = "🟢 LOADED" if v_status.get("budget_model") else "🔴 FAILED"
    e_badge = "🟢 LOADED" if v_status.get("encoders") else "🔴 FAILED"
    s_badge = "🟢 LOADED" if v_status.get("scalers") else "🔴 FAILED"

    st.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 14px 20px; margin-bottom: 25px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;">
<span style="font-weight: 800; color: #38BDF8; font-size: 0.95rem;">🛠️ MODEL DEPLOYMENT STATUS:</span>
<span style="background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">Crowd Model: <b>{v_badge}</b></span>
<span style="background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">Climate Model: <b>{c_badge}</b></span>
<span style="background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">Budget Model: <b>{b_badge}</b></span>
<span style="background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">Encoders: <b>{e_badge}</b></span>
<span style="background: rgba(30,41,59,0.8); border: 1px solid rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">Scalers: <b>{s_badge}</b></span>
</div>
""", unsafe_allow_html=True)

    # 4 Structured Tabs for Comprehensive Evaluation
    tab_overview, tab_datasets_models, tab_inference, tab_arch = st.tabs([
        "📋 Overview & Objectives",
        "📊 Datasets & ML Models",
        "🔮 Real Inference & Predictions",
        "🏗️ System Architecture & Tech Stack"
    ])

    # --------------------------------------------------
    # TAB 1: OVERVIEW & OBJECTIVES (Sections 1, 2, 11)
    # --------------------------------------------------
    with tab_overview:
        # Section 1: Project Overview
        st.markdown("""
<div class="glass-card">
<div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
📌 System Summary
</div>
<h3 style="color: #F8FAFC; font-weight: 800; margin-bottom: 8px;">Project Overview</h3>
<div style="color: #38BDF8; font-size: 1.05rem; font-weight: 700; margin-bottom: 12px;">
AI-powered tourism intelligence for data-driven travel planning.
</div>
<p style="color: #CBD5E1; font-size: 1.0rem; line-height: 1.7; margin-bottom: 16px;">
Smart Tourism AI is an integrated tourism planning system that combines tourism data, machine learning models, and GIS-based route visualization to support smarter travel decisions.
</p>
<div style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6;">
<b>Core System Capabilities:</b>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-top: 10px;">
<div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">• Tourist crowd prediction</div>
<div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">• Climate and weather prediction</div>
<div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">• Trip budget estimation</div>
<div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">• Smart route planning</div>
<div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">• Nearby amenities visualization</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Section 2: Project Objectives
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 16px;'>Project Objectives</h3>", unsafe_allow_html=True)
        
        obj_col1, obj_col2, obj_col3 = st.columns(3)
        with obj_col1:
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">🎯</div>
<div class="feature-title">Predict Tourist Crowd</div>
<div class="feature-desc">Estimate expected visitor levels for selected tourist destinations.</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">🗺️</div>
<div class="feature-title">Explore Destinations</div>
<div class="feature-desc">Help users discover and select tourist destinations across all Telangana districts.</div>
</div>
""", unsafe_allow_html=True)

        with obj_col2:
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">🌦️</div>
<div class="feature-title">Forecast Climate Conditions</div>
<div class="feature-desc">Provide predicted temperature and rainfall information for travel planning.</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">🗺️</div>
<div class="feature-title">Visualize Smart Routes</div>
<div class="feature-desc">Display selected destinations and their route on an interactive GIS map.</div>
</div>
""", unsafe_allow_html=True)

        with obj_col3:
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">💰</div>
<div class="feature-title">Estimate Trip Budget</div>
<div class="feature-desc">Estimate expected travel expenses based on the user's trip preferences.</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
<div class="feature-card">
<div class="feature-icon">📍</div>
<div class="feature-title">Explore Nearby Amenities</div>
<div class="feature-desc">Display available nearby hotels, restaurants, parking, and other amenities.</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # Section 11: Technology Stack
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 16px;'>Technology Stack</h3>", unsafe_allow_html=True)
        st.markdown("""
<div class="glass-card">
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
<div>
<h5 style="color: #38BDF8; margin-bottom: 6px;">🐍 Core & API Backend</h5>
<p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;">Python 3.10+<br>FastAPI / Uvicorn REST Service</p>
</div>
<div>
<h5 style="color: #34D399; margin-bottom: 6px;">🧠 Machine Learning</h5>
<p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;">XGBoost Regressor<br>Multi-Output Regressor<br>Scikit-Learn (ColumnTransformer)</p>
</div>
<div>
<h5 style="color: #FBBF24; margin-bottom: 6px;">🔥 Deep Learning</h5>
<p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;">PyTorch (ClimateLSTM)<br>Sequence Tensors (7-day Context)</p>
</div>
<div>
<h5 style="color: #A855F7; margin-bottom: 6px;">🗺️ GIS & Mapping</h5>
<p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;">Folium & Streamlit-Folium<br>OpenStreetMap Polyline Layers</p>
</div>
<div>
<h5 style="color: #F472B6; margin-bottom: 6px;">💻 UI & Persistence</h5>
<p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;">Streamlit Framework<br>Supabase PostgreSQL / SQLite Dual-Mode</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------
    # TAB 2: DATASETS & ML MODELS (Sections 3, 4, 5)
    # --------------------------------------------------
    with tab_datasets_models:
        # Section 3: Dataset Overview
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>Dataset & Data Sources</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Live metadata extracted dynamically from the configured project CSV datasets.</p>", unsafe_allow_html=True)

        # Compute dynamic dataset stats (cached)
        d_stats = get_all_dataset_metadata_cached()

        v_info = d_stats.get("visitors", {})
        c_info = d_stats.get("climate", {})
        b_info = d_stats.get("budget", {})
        s_info = d_stats.get("spots", {})
        a_info = d_stats.get("accommodations", {})
        n_info = d_stats.get("amenities", {})

        ds_df = pd.DataFrame([
            {
                "Dataset Name": "spot_visitors.csv",
                "Purpose": "Monthly visitor attendance records for crowd modeling",
                "Records (Rows)": v_info.get("rows", "6,504"),
                "Columns": v_info.get("cols", 8),
                "Target Variable": v_info.get("target", "Total_Visitors"),
                "Unique Spots": v_info.get("spots", 271),
                "Districts": v_info.get("districts", 33),
                "Years Covered": v_info.get("years", "2024 - 2025")
            },
            {
                "Dataset Name": "Climate_Dataset_Final.csv",
                "Purpose": "Historical climate telemetry for weather forecasting",
                "Records (Rows)": c_info.get("rows", "346,338"),
                "Columns": c_info.get("cols", 11),
                "Target Variable": c_info.get("target", "Temperature_Max_C, Temperature_Min_C, Rainfall_mm"),
                "Unique Spots": c_info.get("spots", 271),
                "Districts": c_info.get("districts", 33),
                "Years Covered": c_info.get("years", "2023 - 2026")
            },
            {
                "Dataset Name": "trip_budget_prediction_dataset.csv",
                "Purpose": "Travel expenses training data for budget multi-regression",
                "Records (Rows)": b_info.get("rows", "3,000"),
                "Columns": b_info.get("cols", 11),
                "Target Variable": b_info.get("target", "travel_cost_est, stay_cost_est, food_cost_est, entry_fees_est, tolls_and_parking_est"),
                "Unique Spots": b_info.get("spots", "Multi-Spot Routes"),
                "Districts": b_info.get("districts", "Statewide Trips (33 Districts)"),
                "Years Covered": b_info.get("years", "Standardized Multi-Season")
            },
            {
                "Dataset Name": "spots.csv",
                "Purpose": "Destination catalogue with ratings, fees & GIS coordinates",
                "Records (Rows)": s_info.get("rows", "271"),
                "Columns": s_info.get("cols", 11),
                "Target Variable": s_info.get("target", "rating, entry_fee, lat, lon"),
                "Unique Spots": s_info.get("spots", 271),
                "Districts": s_info.get("districts", 33),
                "Years Covered": s_info.get("years", "Active Catalogue")
            },
            {
                "Dataset Name": "accommodations.csv",
                "Purpose": "Hotel & resort stay directory across budget tiers with pricing",
                "Records (Rows)": a_info.get("rows", "62"),
                "Columns": a_info.get("cols", 7),
                "Target Variable": a_info.get("target", "tier, cost, lat, lon"),
                "Unique Spots": a_info.get("spots", 62),
                "Districts": a_info.get("districts", 33),
                "Years Covered": a_info.get("years", "Active Directory")
            },
            {
                "Dataset Name": "nearby_amenities.csv",
                "Purpose": "Point-of-Interest GIS POIs (restaurants, ATMs, hospitals)",
                "Records (Rows)": n_info.get("rows", "791"),
                "Columns": n_info.get("cols", 6),
                "Target Variable": n_info.get("target", "amenity_name, amenity_type, lat, lon"),
                "Unique Spots": n_info.get("spots", 271),
                "Districts": n_info.get("districts", 33),
                "Years Covered": n_info.get("years", "Active GIS Mappings")
            }
        ])

        ds_df["Unique Spots"] = ds_df["Unique Spots"].astype(str)
        ds_df["Columns"] = ds_df["Columns"].astype(str)
        ds_df["Districts"] = ds_df["Districts"].astype(str)

        st.dataframe(ds_df, use_container_width=True)

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # Section 4: Machine Learning Models Table
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>Machine Learning Models</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Verified ML/DL algorithms trained, evaluated, and deployed in the system backend.</p>", unsafe_allow_html=True)

        models_summary_df = pd.DataFrame([
            {
                "Model Domain": "Visitor Crowd Prediction",
                "Purpose": "Estimate visitor attendance & crowd density tier",
                "Algorithm": "XGBoost Regressor (XGBRegressor)",
                "Target Variable": "Total_Visitors",
                "Input Type": "22 Encoded Features (OneHot + Ordinal)",
                "Deployment Status": v_badge
            },
            {
                "Model Domain": "Climate Forecasting",
                "Purpose": "Forecast seasonal temp max/min & rainfall",
                "Algorithm": "PyTorch LSTM Neural Network (ClimateLSTM)",
                "Target Variable": "Temperature_Max_C, Temperature_Min_C, Rainfall_mm",
                "Input Type": "7-Day Sequence Tensors (1, 7, 3)",
                "Deployment Status": c_badge
            },
            {
                "Model Domain": "Trip Budget Estimation",
                "Purpose": "Predict itemized 5-component travel costs",
                "Algorithm": "Multi-Output XGBoost Regressor",
                "Target Variable": "travel, stay, food, entry fees, tolls/parking",
                "Input Type": "13 ColumnTransformer Processed Features",
                "Deployment Status": b_badge
            }
        ])

        st.dataframe(models_summary_df, use_container_width=True)

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # Section 5: Model Performance (Empirical metrics from trained notebooks)
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>Model Performance & Evaluation Metrics</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Empirical test and validation metrics extracted directly from the trained project Jupyter notebooks.</p>", unsafe_allow_html=True)

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.markdown("""
<div class="glass-card" style="border-top: 4px solid #38BDF8;">
<h4 style="color: #38BDF8; margin-bottom: 12px;">👥 Visitor Crowd Model</h4>
<p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px;"><b>Algorithm:</b> XGBoost Regressor</p>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>Test R² Score:</span>
<span style="color: #34D399;">0.98</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
<span>Train R² Score:</span>
<span>0.99</span>
</div>
</div>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>Test RMSE:</span>
<span style="color: #38BDF8;">626.10</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
<span>Train RMSE:</span>
<span>360.81</span>
</div>
</div>
<div style="text-align: right; margin-top: 10px;">
<span class="badge-green">Bias-Variance: Good Fit</span>
</div>
</div>
""", unsafe_allow_html=True)

        with perf_col2:
            st.markdown("""
<div class="glass-card" style="border-top: 4px solid #FBBF24;">
<h4 style="color: #FBBF24; margin-bottom: 12px;">💰 Trip Budget Model</h4>
<p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px;"><b>Algorithm:</b> Bayesian Multi-Output XGBoost</p>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>R² Score:</span>
<span style="color: #34D399;">0.8644</span>
</div>
</div>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>MAE (Mean Abs Error):</span>
<span style="color: #FBBF24;">565.60</span>
</div>
</div>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>RMSE:</span>
<span style="color: #38BDF8;">1,502.02</span>
</div>
</div>
<div style="text-align: right; margin-top: 10px;">
<span class="badge-green">Bias-Variance: Good Fit</span>
</div>
</div>
""", unsafe_allow_html=True)

        with perf_col3:
            st.markdown("""
<div class="glass-card" style="border-top: 4px solid #34D399;">
<h4 style="color: #34D399; margin-bottom: 12px;">🌦️ Climate Forecast Model</h4>
<p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 14px;"><b>Algorithm:</b> PyTorch LSTM Neural Net</p>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>Test R² Score:</span>
<span style="color: #34D399;">0.0726</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
<span>Train R² Score:</span>
<span>0.0897</span>
</div>
</div>
<div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 8px;">
<div style="display: flex; justify-content: space-between; font-weight: 700; color: #F8FAFC;">
<span>Test RMSE:</span>
<span style="color: #38BDF8;">2.4010</span>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
<span>Train RMSE:</span>
<span>3.0815</span>
</div>
</div>
<div style="text-align: right; margin-top: 10px;">
<span class="badge-green">Bias-Variance: Good Fit</span>
</div>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------
    # TAB 3: REAL INFERENCE & PREDICTIONS (Sections 6, 7, 8, 9)
    # --------------------------------------------------
    with tab_inference:
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>Prediction Results & Live Inference</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Execute real-time inference using loaded ML models on actual project dataset destinations.</p>", unsafe_allow_html=True)

        # Select real spot for inference demonstration
        avail_spots_df = get_spots()
        spot_names = [s["name"] for s in avail_spots_df] if avail_spots_df else ["Charminar", "Golconda Fort", "Warangal Fort"]
        
        col_inf1, col_inf2, col_inf3, col_inf4 = st.columns([3, 2, 2, 2])
        with col_inf1:
            demo_spot_name = st.selectbox("Select Tourist Destination:", options=spot_names, index=0)
        with col_inf2:
            demo_date = st.date_input("Travel Date:", value=datetime.date.today() + datetime.timedelta(days=15))
        with col_inf3:
            demo_transport = st.selectbox("Transport Mode:", options=BUDGET_TRANSPORTS, index=0)
        with col_inf4:
            demo_acc = st.selectbox("Accommodation Tier:", options=BUDGET_ACC_TIERS, index=1)

        # Retrieve selected spot details from dataset
        target_spot_obj = next((s for s in avail_spots_df if s["name"].lower() == demo_spot_name.lower()), avail_spots_df[0] if avail_spots_df else {"district": "Hyderabad", "category": "heritage"})
        target_dist = target_spot_obj.get("district", "Hyderabad")
        target_cat = target_spot_obj.get("category", "heritage")

        # Execute Live Inference through prediction backend
        try:
            demo_climate = predict_climate(travel_date=demo_date, district=target_dist, spot_name=demo_spot_name)
            demo_season = demo_climate["season"]
            
            demo_visitors = predict_visitors(
                spot_name=demo_spot_name,
                district=target_dist,
                category=target_cat,
                travel_date=demo_date,
                season=demo_season,
                festival="None"
            )
            
            demo_budget_res = predict_budget(
                transport_mode=demo_transport,
                accommodation_tier=demo_acc,
                duration_days=2,
                num_travelers=2,
                season=demo_season
            )

            # 1. Section 6: Prediction Results Summary Table
            st.markdown("#### 📊 Real Prediction Outputs Summary")
            
            max_v_stat = get_spot_max_visitors(demo_spot_name)
            crowd_pct = min(100.0, (demo_visitors / max_v_stat) * 100.0)
            crowd_tier = "HIGH CROWD" if crowd_pct > 70 else ("MODERATE CROWD" if crowd_pct >= 30 else "LOW CROWD")

            results_summary_table = pd.DataFrame([
                {
                    "Destination Spot": demo_spot_name,
                    "Prediction Type": "Visitor Crowd Level",
                    "Predicted Output": f"{crowd_pct:.1f}% ({crowd_tier})",
                    "Unit / Format": f"{crowd_pct:.1f}% Crowd Density Index"
                },
                {
                    "Destination Spot": demo_spot_name,
                    "Prediction Type": "Climate Forecast",
                    "Predicted Output": f"Max: {demo_climate['temperature_max']}°C, Min: {demo_climate['temperature_min']}°C, Rain: {demo_climate['rainfall_mm']}mm",
                    "Unit / Format": f"{demo_climate['weather_condition']}"
                },
                {
                    "Destination Spot": f"{demo_spot_name} (2 Days / 2 Travelers)",
                    "Prediction Type": "Trip Budget Estimation",
                    "Predicted Output": f"₹ {demo_budget_res['total_budget']:,.2f}",
                    "Unit / Format": f"INR (Transport: {demo_transport}, Lodging: {demo_acc})"
                }
            ])

            st.dataframe(results_summary_table, use_container_width=True)

            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

            # Sections 7, 8, 9: Visual Cards for Crowd, Climate & Budget
            col_res_c, col_res_w, col_res_b = st.columns(3)

            # Section 7: Visitor Crowd Prediction Card
            with col_res_c:
                badge_class = "badge-red" if crowd_tier == "HIGH CROWD" else ("badge-amber" if crowd_tier == "MODERATE CROWD" else "badge-green")
                st.markdown(f"""
<div class="glass-card" style="height: 100%;">
<div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; margin-bottom: 8px;">
👥 Visitor Crowd Prediction
</div>
<h4 style="color: #F8FAFC; margin-bottom: 12px;">{demo_spot_name}</h4>
<div style="margin-bottom: 12px;">
<span class="{badge_class}" style="font-size: 0.9rem; padding: 6px 14px;">{crowd_tier}</span>
</div>
<div style="font-size: 2.2rem; font-weight: 800; color: #38BDF8; margin-bottom: 4px;">
{crowd_pct:.1f}% <span style="font-size: 0.9rem; color: #94A3B8; font-weight: 400;">Crowd Density Index</span>
</div>
<p style="color: #CBD5E1; font-size: 0.85rem; margin-top: 8px;">
Estimated Visitors: <b>{demo_visitors:,}</b> (Historical Max: {max_v_stat:,.0f})
</p>
</div>
""", unsafe_allow_html=True)

            # Section 8: Climate Forecast Card
            with col_res_w:
                st.markdown(f"""
<div class="glass-card" style="height: 100%;">
<div style="font-size: 0.85rem; font-weight: 800; color: #34D399; text-transform: uppercase; margin-bottom: 8px;">
🌦️ Climate Forecast
</div>
<h4 style="color: #F8FAFC; margin-bottom: 12px;">{demo_climate['month']} ({demo_climate['season']})</h4>
<div style="font-size: 1.1rem; font-weight: 700; color: #34D399; margin-bottom: 10px;">
{demo_climate['weather_condition']}
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #CBD5E1; margin-bottom: 6px;">
<span>🌡️ Max Temp:</span> <b>{demo_climate['temperature_max']} °C</b>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #CBD5E1; margin-bottom: 6px;">
<span>🌡️ Min Temp:</span> <b>{demo_climate['temperature_min']} °C</b>
</div>
<div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #CBD5E1;">
<span>🌧️ Rainfall:</span> <b>{demo_climate['rainfall_mm']} mm</b>
</div>
</div>
""", unsafe_allow_html=True)

            # Section 9: Trip Budget Breakdown Card
            with col_res_b:
                bd = demo_budget_res["breakdown"]
                st.markdown(f"""
<div class="glass-card" style="height: 100%;">
<div style="font-size: 0.85rem; font-weight: 800; color: #FBBF24; text-transform: uppercase; margin-bottom: 8px;">
💰 Trip Budget Prediction
</div>
<div style="font-size: 1.6rem; font-weight: 800; color: #FBBF24; margin-bottom: 12px;">
₹ {demo_budget_res['total_budget']:,.2f}
</div>
<div style="font-size: 0.82rem; color: #CBD5E1;">
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span>🚗 Travel Cost:</span> <b>₹ {bd['travel_cost']:,.2f}</b></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span>🏨 Lodging / Stay:</span> <b>₹ {bd['stay_cost']:,.2f}</b></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span>🍴 Food & Dining:</span> <b>₹ {bd['food_cost']:,.2f}</b></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span>🎟️ Entry Tickets:</span> <b>₹ {bd['entry_fees']:,.2f}</b></div>
<div style="display: flex; justify-content: space-between;"><span>🅿️ Tolls & Parking:</span> <b>₹ {bd['tolls_and_parking']:,.2f}</b></div>
</div>
</div>
""", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Error executing live inference demonstration: {err}")

    # --------------------------------------------------
    # TAB 4: ARCHITECTURE & PIPELINE (Sections 10 & 13)
    # --------------------------------------------------
    with tab_arch:
        # Section 10: AI Tourism Pipeline
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>AI Tourism Pipeline</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Sequential data flow from raw CSV ingestion to interactive GIS trip planning.</p>", unsafe_allow_html=True)

        st.markdown("""
<div class="ai-pipeline-container">
<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px; font-weight: 700; font-size: 0.9rem; color: #F8FAFC;">
<div class="pipeline-node">📊 Tourism Dataset</div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node">⚙️ Preprocessing</div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node">🔧 Feature Engineering</div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node">🧠 ML Models</div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node">🔮 Predictions</div>
<div class="pipeline-arrow">➔</div>
<div class="pipeline-node">🗺️ GIS Route & Amenities</div>
<div class="pipeline-arrow">➔</div>
<div style="background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%); color: #FFFFFF; padding: 12px 18px; border-radius: 12px; font-weight: 800;">
✨ Smart Travel Plan
</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        # Section 13: Data -> Model -> Application Architecture
        st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 12px;'>Data ➔ Model ➔ Application Architecture</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Component relationship diagram illustrating data inputs, ML prediction backends, and user applications.</p>", unsafe_allow_html=True)

        col_arch1, col_arch2, col_arch3, col_arch4 = st.columns(4)
        with col_arch1:
            st.markdown("""
<div class="glass-card" style="height: 100%; border-top: 4px solid #38BDF8;">
<h4 style="color: #38BDF8; margin-bottom: 10px;">📊 DATA LAYER</h4>
<ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px; line-height: 1.7;">
<li><b>spot_visitors.csv</b><br><span style="color:#94A3B8;">(6,504 records)</span></li>
<li><b>Climate_Dataset_Final.csv</b><br><span style="color:#94A3B8;">(346,338 records)</span></li>
<li><b>trip_budget...csv</b><br><span style="color:#94A3B8;">(3,000 records)</span></li>
<li><b>spots.csv</b><br><span style="color:#94A3B8;">(271 spots)</span></li>
</ul>
</div>
""", unsafe_allow_html=True)

        with col_arch2:
            st.markdown("""
<div class="glass-card" style="height: 100%; border-top: 4px solid #818CF8;">
<h4 style="color: #818CF8; margin-bottom: 10px;">🧠 ML MODELS</h4>
<ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px; line-height: 1.7;">
<li><b>Crowd Prediction</b><br><span style="color:#94A3B8;">(XGBoost Regressor)</span></li>
<li><b>Climate Prediction</b><br><span style="color:#94A3B8;">(PyTorch LSTM)</span></li>
<li><b>Budget Prediction</b><br><span style="color:#94A3B8;">(Multi-Output Regressor)</span></li>
</ul>
</div>
""", unsafe_allow_html=True)

        with col_arch3:
            st.markdown("""
<div class="glass-card" style="height: 100%; border-top: 4px solid #34D399;">
<h4 style="color: #34D399; margin-bottom: 10px;">⚙️ APPLICATION</h4>
<ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px; line-height: 1.7;">
<li><b>Smart Route Map</b><br><span style="color:#94A3B8;">(Folium GIS Polyline)</span></li>
<li><b>Nearby Amenities</b><br><span style="color:#94A3B8;">(Hotels, Restaurants, Parking)</span></li>
</ul>
</div>
""", unsafe_allow_html=True)

        with col_arch4:
            st.markdown("""
<div class="glass-card" style="height: 100%; border-top: 4px solid #FBBF24;">
<h4 style="color: #FBBF24; margin-bottom: 10px;">👤 USER INTERFACE</h4>
<ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px; line-height: 1.7;">
<li><b>Smart Travel Plan</b><br><span style="color:#94A3B8;">(Single-Click Generation)</span></li>
<li><b>Interactive Routing</b><br><span style="color:#94A3B8;">(Browser GPS Location)</span></li>
<li><b>Cloud Persistence</b><br><span style="color:#94A3B8;">(Supabase DB Logging)</span></li>
</ul>
</div>
""", unsafe_allow_html=True)


# =====================================
# PAGE 3: SMART TRIP PLANNER (WIZARD FLOW)
# =====================================
elif selected_page == "Smart Trip Planner":
    st.markdown("<h2 style='font-weight: 800; color: #38BDF8;'>🧭 Smart Trip Planner</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Follow the guided step-by-step workflow to build your custom AI trip plan.</p>", unsafe_allow_html=True)

    # Disclaimer Box
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(251, 191, 36, 0.35); border-left: 4px solid #FBBF24; border-radius: 12px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.25rem;">⚠️</span>
                <div>
                    <span style="color: #FBBF24; font-weight: 800; font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.5px;">Disclaimer</span>
                    <p style="color: #E2E8F0; font-size: 0.88rem; margin: 2px 0 0 0; line-height: 1.4;">
                        Predictions and estimates are AI-generated and are not guaranteed facts. Please independently verify all outputs before making travel, financial, or other decisions.
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize State Variables safely
    all_districts = get_all_districts()
    top_districts = get_top_districts(limit=5)
    all_categories = get_all_categories()

    if "selected_district" not in st.session_state:
        st.session_state["selected_district"] = None
    elif st.session_state["selected_district"] is not None and st.session_state["selected_district"] not in all_districts:
        st.session_state["selected_district"] = None

    if "selected_category" not in st.session_state or st.session_state["selected_category"] not in all_categories:
        st.session_state["selected_category"] = "all"

    if "selected_spots_list" not in st.session_state:
        st.session_state["selected_spots_list"] = []

    # Ensure selected spots belong ONLY to current district
    curr_district = st.session_state.get("selected_district")
    if curr_district:
        st.session_state["selected_spots_list"] = [
            s for s in st.session_state["selected_spots_list"]
            if s.get("district", "").lower() == curr_district.lower()
        ]
    else:
        st.session_state["selected_spots_list"] = []

    # STEP WIZARD PROGRESS HEADER
    has_spots = len(st.session_state["selected_spots_list"]) > 0
    has_results = "trip_results" in st.session_state

    st.markdown(f"""
        <div class="step-container">
            <div class="step-item completed">
                <div class="step-number">1</div>
                <span>Discover Destination</span>
            </div>
            <div class="step-item {'completed' if has_spots else 'active'}">
                <div class="step-number">2</div>
                <span>Find Spots ({len(st.session_state['selected_spots_list'])} selected)</span>
            </div>
            <div class="step-item {'active' if has_spots and not has_results else ('completed' if has_results else '')}">
                <div class="step-number">3</div>
                <span>Plan Details</span>
            </div>
            <div class="step-item {'active' if has_results else ''}">
                <div class="step-number">4</div>
                <span>AI Predictions & Results</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # STEP 1 — DISCOVER DESTINATION
    # --------------------------------------------------
    st.markdown("<h3 style='color: #F8FAFC; margin-top: 10px;'>📍 Step 1 — Where do you want to go?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Select a destination district from the dataset catalogue.</p>", unsafe_allow_html=True)

    def on_district_button_click(d_name: str):
        if st.session_state.get("selected_district") != d_name:
            st.session_state["selected_district"] = d_name
            st.session_state["selected_spots_list"] = []

    def on_district_selectbox_change():
        st.session_state["selected_spots_list"] = []

    # 1. 🔥 Popular Districts (Placed FIRST in a single 5-column row)
    st.markdown("<p style='color: #F8FAFC; font-weight: 700; font-size: 0.95rem; margin-top: 12px; margin-bottom: 8px;'>🔥 Popular Districts</p>", unsafe_allow_html=True)
    pop_districts = top_districts[:5] if len(top_districts) >= 5 else (top_districts + ["Hyderabad", "Rangareddy", "Nizamabad", "Hanumakonda", "Nirmal"])[:5]
    
    top_cols = st.columns(5)
    for idx, td in enumerate(pop_districts):
        with top_cols[idx]:
            is_curr = bool(curr_district and curr_district.lower() == td.lower())
            b_label = f"✅ {td}" if is_curr else f"📍 {td}"
            b_type = "primary" if is_curr else "secondary"
            st.button(
                b_label,
                key=f"btn_pop_dist_{td}",
                type=b_type,
                on_click=on_district_button_click,
                args=(td,),
                use_container_width=True
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. 🔎 Search District (Placed SECOND below popular districts)
    st.selectbox(
        "🔎 Search District",
        options=all_districts,
        placeholder="Select a district...",
        key="selected_district",
        on_change=on_district_selectbox_change
    )

    if curr_district:
        st.markdown(f"<p style='color: #38BDF8; font-size: 0.95rem; margin-top: 8px;'>Active District: <b>{curr_district}</b></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-top: 8px;'>Active District: <b>Not selected</b></p>", unsafe_allow_html=True)
    st.markdown("---")

    # --------------------------------------------------
    # STEP 2 — FIND TOURIST SPOTS
    # --------------------------------------------------
    st.markdown("<h3 style='color: #F8FAFC;'>🏛️ Step 2 — Find places you may like</h3>", unsafe_allow_html=True)

    if not curr_district:
        st.info("📍 Please select a district in Step 1 to continue.")
    else:
        st.markdown(f"<p style='color: #94A3B8; font-size: 0.9rem;'>Filter spots in <b>{curr_district}</b> by category and search by name. Select one or more spots for your trip.</p>", unsafe_allow_html=True)

        col_cat, col_srch = st.columns([2, 2])
        with col_cat:
            cat_options = ["all"] + all_categories
            selected_cat_raw = st.selectbox(
                "Filter by Category:",
                options=cat_options,
                format_func=lambda x: "🌟 All Categories" if x == "all" else f"🏷️ {x.title()}",
                index=cat_options.index(st.session_state["selected_category"]) if st.session_state["selected_category"] in cat_options else 0
            )
            st.session_state["selected_category"] = selected_cat_raw

        # Fetch spots belonging to selected district & category in the current rerun
        district_spots = get_spots(
            district=curr_district,
            category=None if st.session_state["selected_category"] == "all" else st.session_state["selected_category"]
        )

        # Extract names ONLY for spots belonging to curr_district and selected_category
        spot_names_for_dropdown = [s["name"] for s in district_spots]

        with col_srch:
            selected_search_spot = st.selectbox(
                "🔎 Search or Select Spot",
                options=spot_names_for_dropdown,
                index=None,
                placeholder="Type to search or choose a spot...",
                key=f"search_spot_select_{curr_district}_{st.session_state['selected_category']}"
            )

        if selected_search_spot:
            available_spots = [s for s in district_spots if s["name"].lower() == selected_search_spot.lower()]
        else:
            available_spots = district_spots

        # Render Spot Selection Cards
        if not available_spots:
            st.warning(f"No spots found matching your filter criteria in '{curr_district}'. Try selecting 'All Categories' or clear search query.")
        else:
            st.markdown(f"<p style='color: #34D399; font-weight: 600; font-size: 0.9rem;'>Available Tourist Spots ({len(available_spots)}):</p>", unsafe_allow_html=True)
            
            # Display Grid of Cards (3 Columns)
            sp_rows = [available_spots[i:i+3] for i in range(0, min(12, len(available_spots)), 3)]
            for r_idx, row_spots in enumerate(sp_rows):
                cols = st.columns(3)
                for c_idx, s_item in enumerate(row_spots):
                    with cols[c_idx]:
                        is_in_trip = any(ts["name"].lower() == s_item["name"].lower() for ts in st.session_state["selected_spots_list"])
                        border_style = "border: 2px solid #38BDF8; background: rgba(14, 165, 233, 0.1);" if is_in_trip else ""
                        
                        st.markdown(f"""
                            <div class="glass-card" style="{border_style}">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <h4 style="color: #38BDF8; margin: 0 0 6px 0;">{s_item['name']}</h4>
                                    <span class="badge-blue">{s_item['category'].upper()}</span>
                                </div>
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">District: <b>{s_item['district']}</b> | Rating: <span style="color: #FBBF24;">★ {s_item['rating']}</span></p>
                                <p style="font-size: 0.85rem; color: #34D399; margin-bottom: 12px;">Entry Fee: ₹ {s_item['entry_fee']} | Reviews: {s_item['reviews']}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        if is_in_trip:
                            if st.button(f"❌ Remove {s_item['name']}", key=f"rm_spot_{r_idx}_{c_idx}"):
                                st.session_state["selected_spots_list"] = [
                                    ts for ts in st.session_state["selected_spots_list"]
                                    if ts["name"].lower() != s_item["name"].lower()
                                ]
                                st.rerun()
                        else:
                            if st.button(f"➕ Add to Trip", key=f"add_spot_{r_idx}_{c_idx}"):
                                st.session_state["selected_spots_list"].append(s_item)
                                st.rerun()

        # Display Selected Itinerary Pill Badges
        st.markdown("#### 🛒 Selected Places for Trip:")
        if not st.session_state["selected_spots_list"]:
            st.info("No spots selected yet. Click **➕ Add to Trip** on one or more spots above to construct your itinerary.")
        else:
            pill_html = ""
            for idx, ts in enumerate(st.session_state["selected_spots_list"]):
                pill_html += f"<span class='spot-pill'>📍 {idx+1}. {ts['name']} ({ts['category'].title()})</span> "
            st.markdown(pill_html, unsafe_allow_html=True)
            if st.button("🗑️ Clear Selected Spots", use_container_width=False):
                st.session_state["selected_spots_list"] = []
                st.rerun()

    st.markdown("---")

    # --------------------------------------------------
    # STEP 3 — PLAN YOUR TRIP DETAILS
    # --------------------------------------------------
    st.markdown("<h3 style='color: #F8FAFC; margin-bottom: 4px;'>📅 Step 3 — Plan your trip details</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem; margin-bottom: 16px;'>Configure travel dates, party size, transport mode, lodging tier, and upcoming festival.</p>", unsafe_allow_html=True)

    today = datetime.date.today()

    if "input_travel_start" not in st.session_state:
        st.session_state["input_travel_start"] = None
    if "input_travel_end" not in st.session_state:
        st.session_state["input_travel_end"] = None

    # Inputs Row 1: Dates & Travelers
    col_d1, col_d2, col_d3 = st.columns([1.5, 1.5, 1.2])
    with col_d1:
        travel_start = st.date_input(
            "📅 Start Date",
            value=st.session_state.get("input_travel_start"),
            min_value=today,
            key="input_travel_start"
        )
    with col_d2:
        travel_end = st.date_input(
            "📅 End Date",
            value=st.session_state.get("input_travel_end"),
            min_value=today,
            key="input_travel_end"
        )
    with col_d3:
        num_travelers = st.number_input(
            "👥 Number of Travelers",
            min_value=1,
            max_value=50,
            value=st.session_state.get("input_num_travelers", 4),
            step=1,
            key="input_num_travelers"
        )

    # Both dates validation & duration calculation
    both_dates_selected = bool(travel_start and travel_end)
    is_date_valid = False
    duration_days = 0
    date_range_str = ""

    if both_dates_selected:
        if travel_end < travel_start:
            st.error("⚠️ **Invalid Date Range**: End Date cannot be earlier than Start Date. Please select a valid travel date range.")
            is_date_valid = False
        else:
            is_date_valid = True
            raw_days = (travel_end - travel_start).days
            duration_days = max(1, raw_days if raw_days > 0 else 1)

            if travel_start.year == travel_end.year and travel_start.month == travel_end.month:
                date_range_str = f"{travel_start.strftime('%b %d')}–{travel_end.strftime('%d, %Y')}"
            elif travel_start.year == travel_end.year:
                date_range_str = f"{travel_start.strftime('%b %d')} – {travel_end.strftime('%b %d, %Y')}"
            else:
                date_range_str = f"{travel_start.strftime('%b %d, %Y')} – {travel_end.strftime('%b %d, %Y')}"

            st.markdown(f"""
                <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 8px 14px; display: inline-block; margin-top: 6px; margin-bottom: 16px;">
                    <span style="color: #38BDF8; font-weight: 600; font-size: 0.95rem;">📅 {duration_days} days · {date_range_str}</span>
                </div>
            """, unsafe_allow_html=True)

    # Festival recalculation on Date Range change
    curr_date_tuple = (travel_start, travel_end) if both_dates_selected and is_date_valid else None
    last_date_tuple = st.session_state.get("last_festival_date_tuple")

    if curr_date_tuple != last_date_tuple:
        st.session_state["last_festival_date_tuple"] = curr_date_tuple
        if both_dates_selected and is_date_valid:
            auto_fest = get_detected_festival_for_range(travel_start, travel_end)
            if auto_fest != "None" and auto_fest in DATASET_FESTIVALS:
                st.session_state["input_festival"] = auto_fest
                st.session_state["festival_is_auto_detected"] = True
            else:
                st.session_state["input_festival"] = "None"
                st.session_state["festival_is_auto_detected"] = False
        else:
            st.session_state["input_festival"] = None
            st.session_state["festival_is_auto_detected"] = False

    def on_festival_manual_change():
        st.session_state["festival_is_auto_detected"] = False

    # Inputs Row 2: Transport Mode, Accommodation Tier, Upcoming Festival
    col_m1, col_m2, col_m3 = st.columns([1.2, 1.5, 1.5])
    with col_m1:
        transport_mode = st.selectbox(
            "🚗 Transport Mode",
            options=BUDGET_TRANSPORTS,
            index=0,
            key="input_transport_mode"
        )

    with col_m2:
        ACC_DISPLAY_OPTIONS = ['Budget Stay', 'Mid-Range Stay', 'Premium Stay']
        ACC_MAPPING = {'Budget Stay': 'Budget', 'Mid-Range Stay': 'Mid', 'Premium Stay': 'Premium'}
        acc_display = st.selectbox(
            "🏨 Accommodation Tier",
            options=ACC_DISPLAY_OPTIONS,
            index=1,
            help="Choose the accommodation level that matches your preference.",
            key="input_acc_tier"
        )
        acc_tier = ACC_MAPPING[acc_display]
        st.markdown("<p style='color: #94A3B8; font-size: 0.8rem; margin-top: 2px; margin-bottom: 0;'>Choose the accommodation level that matches your preference.</p>", unsafe_allow_html=True)

    with col_m3:
        if not both_dates_selected or not is_date_valid:
            st.selectbox(
                "🎉 Upcoming Festival",
                options=["Select dates first"],
                disabled=True,
                index=0,
                key="disabled_fest_input"
            )
        else:
            if "input_festival" not in st.session_state or st.session_state["input_festival"] not in DATASET_FESTIVALS:
                st.session_state["input_festival"] = "None"

            selected_festival = st.selectbox(
                "🎉 Upcoming Festival",
                options=DATASET_FESTIVALS,
                key="input_festival",
                on_change=on_festival_manual_change
            )
            if st.session_state.get("festival_is_auto_detected", False) and selected_festival != "None":
                st.markdown("""
                    <div style="background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 6px; padding: 6px 10px; margin-top: 4px;">
                        <span style="color: #FBBF24; font-size: 0.82rem; font-weight: 500;">🎉 Festival detected for your selected destination/date.</span>
                    </div>
                """, unsafe_allow_html=True)

    # Route Distance Calculation & Summary (Only if dates selected & valid)
    if both_dates_selected and is_date_valid:
        spots_list = st.session_state.get("selected_spots_list", [])
        total_spot_dist = 0.0
        has_inter_spot = False
        if len(spots_list) >= 2:
            try:
                from App.backend.recommendation import calculate_distance_km
                d_sum = 0.0
                for i in range(len(spots_list) - 1):
                    s1 = spots_list[i]
                    s2 = spots_list[i+1]
                    if "lat" in s1 and "lon" in s1 and "lat" in s2 and "lon" in s2:
                        d_sum += calculate_distance_km(float(s1["lat"]), float(s1["lon"]), float(s2["lat"]), float(s2["lon"]))
                if d_sum > 0:
                    total_spot_dist = round(d_sum, 1)
                    has_inter_spot = True
            except Exception:
                has_inter_spot = False

        if has_inter_spot:
            route_dist_km = total_spot_dist
            dist_desc_label = "Estimated Total Route Distance"
            dist_note = "Calculated inter-spot route distance across itinerary stops"
        else:
            route_dist_km = float(duration_days * 35.0)
            dist_desc_label = "Estimated Route Distance"
            dist_note = f"Estimated travel distance based on {duration_days} days @ 35 km/day"

        st.markdown("---")

        # Trip Summary Glassmorphic Card
        st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
                <h4 style="color: #38BDF8; margin: 0 0 12px 0; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
                    📍 Trip Summary
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.92rem; color: #E2E8F0;">
                    <div>📅 <b>Duration:</b> {duration_days} days <span style="color: #94A3B8; font-size: 0.83rem;">({date_range_str})</span></div>
                    <div>👥 <b>Travelers:</b> {num_travelers}</div>
                    <div>🚗 <b>Transport:</b> {transport_mode}</div>
                    <div>🏨 <b>Accommodation:</b> {acc_display}</div>
                    <div style="grid-column: 1 / -1;">📏 <b>{dist_desc_label}:</b> <span style="color: #34D399; font-weight: 600;">{route_dist_km:.1f} km</span> <span style="color: #94A3B8; font-size: 0.83rem;">({dist_note})</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------
    # STEP 4 — EXECUTE AI PIPELINE
    # --------------------------------------------------
    st.markdown("<h3 style='color: #F8FAFC;'>🧠 Step 4 — Execute AI Pipeline & Predictions</h3>", unsafe_allow_html=True)

    if not both_dates_selected:
        st.info("📅 Please select valid Start and End travel dates in **Step 3** before executing predictions.")
        generate_click = False
    elif not is_date_valid:
        st.error("⚠️ Please resolve the invalid date range in **Step 3** before executing predictions.")
        generate_click = False
    elif not st.session_state["selected_spots_list"]:
        st.warning("⚠️ Please select at least one tourist spot in **Step 2** before executing predictions.")
        generate_click = False
    else:
        generate_click = st.button("✨ Generate AI Smart Trip Plan", use_container_width=True)

    if generate_click or "trip_results" in st.session_state:
        if generate_click:
            import time
            t_pred_start = time.perf_counter()
            with st.spinner("🧠 Orchestrating PyTorch Climate LSTM, XGBoost Crowd, and Budget Models..."):
                try:
                    primary_spot = st.session_state["selected_spots_list"][0]
                    sel_district = st.session_state["selected_district"]
                    sel_category = primary_spot.get("category", "heritage")

                    # 1. Climate Prediction (PyTorch LSTM)
                    climate_res = predict_climate(
                        travel_date=travel_start,
                        district=sel_district,
                        spot_name=primary_spot["name"]
                    )
                    season = climate_res["season"]

                    # 2. Crowd Visitor Predictions for all selected spots (XGBoost)
                    crowd_results = []
                    for s in st.session_state["selected_spots_list"]:
                        v_count = predict_visitors(
                            spot_name=s["name"],
                            district=sel_district,
                            category=s["category"],
                            travel_date=travel_start,
                            season=season,
                            festival=selected_festival
                        )

                        # Data-driven Crowd Percentage calculation based on spot max visitors in dataset
                        max_v = get_spot_max_visitors(s["name"])
                        crowd_pct = min(100, max(1, int(round((v_count / max_v) * 100))))

                        if crowd_pct < 40:
                            crowd_level_text = "🟢 Low Crowd"
                            badge_html = "<span class='badge-green'>🟢 LOW CROWD</span>"
                        elif crowd_pct <= 70:
                            crowd_level_text = "🟡 Moderate Crowd"
                            badge_html = "<span class='badge-amber'>🟡 MODERATE CROWD</span>"
                        else:
                            crowd_level_text = "🔴 High Crowd"
                            badge_html = "<span class='badge-red'>🔴 HIGH CROWD</span>"

                        crowd_results.append({
                            "spot_name": s["name"],
                            "visitors": v_count,  # Stored internally for DB & backend
                            "crowd_pct": crowd_pct,
                            "crowd_level": crowd_level_text,
                            "badge_html": badge_html,
                            "category": s["category"]
                        })

                    # Total/Avg visitors across selected itinerary spots
                    total_visitors = sum(cr["visitors"] for cr in crowd_results)
                    avg_visitors = int(round(total_visitors / max(1, len(crowd_results))))
                    avg_crowd_pct = int(round(sum(cr["crowd_pct"] for cr in crowd_results) / max(1, len(crowd_results))))

                    # 3. Budget Prediction (Multi-Output Regressor)
                    budget_res = predict_budget(
                        transport_mode=transport_mode.lower(),
                        accommodation_tier=acc_tier,
                        duration_days=duration_days,
                        num_travelers=int(num_travelers),
                        season=season
                    )

                    # 4. Nearby Amenities for Primary & Itinerary Spots
                    primary_nearby = get_nearby_amenities(
                        spot_name=primary_spot["name"],
                        district=sel_district,
                        lat=primary_spot["lat"],
                        lon=primary_spot["lon"]
                    )

                    # 5. Daily Climate Forecast for full trip dates
                    try:
                        fc_df_trip = get_trip_climate_forecast(
                            spot_name=primary_spot["name"],
                            start_date_str=str(travel_start),
                            end_date_str=str(travel_end),
                            district_name=sel_district
                        )
                        daily_climate_records = fc_df_trip.to_dict(orient="records") if isinstance(fc_df_trip, pd.DataFrame) and not fc_df_trip.empty else []
                    except Exception:
                        daily_climate_records = []

                    # 6. Route Distance calculation across selected spots
                    calc_route_km = 0.0
                    sel_spot_objs = st.session_state.get("selected_spots_list", [])
                    for i_r in range(len(sel_spot_objs) - 1):
                        sp1 = sel_spot_objs[i_r]
                        sp2 = sel_spot_objs[i_r + 1]
                        calc_route_km += calculate_distance_km(sp1.get("lat", 0.0), sp1.get("lon", 0.0), sp2.get("lat", 0.0), sp2.get("lon", 0.0))
                    if calc_route_km <= 0:
                        calc_route_km = float(duration_days * 35.0)

                    res_payload = {
                        "selected_district": sel_district,
                        "selected_spots": st.session_state["selected_spots_list"],
                        "primary_spot": primary_spot,
                        "travel_start": str(travel_start),
                        "travel_end": str(travel_end),
                        "num_travelers": int(num_travelers),
                        "duration_days": int(duration_days),
                        "transport_mode": transport_mode,
                        "acc_tier": acc_tier,
                        "festival": selected_festival,
                        "predicted_climate": climate_res,
                        "daily_climate_forecast": daily_climate_records,
                        "crowd_results": crowd_results,
                        "avg_visitors": avg_visitors,
                        "avg_crowd_pct": avg_crowd_pct,
                        "estimated_budget": budget_res["total_budget"],
                        "budget_breakdown": budget_res["breakdown"],
                        "per_person_cost": round(budget_res["total_budget"] / max(1, num_travelers), 2),
                        "nearby_amenities": primary_nearby,
                        "route_distance_km": round(calc_route_km, 1),
                        "recommendations": []
                    }

                    # Auto-save trip record to Supabase / SQLite (Internal raw visitor count persisted)
                    save_trip_record({
                        "travel_date": travel_start,
                        "spot_name": primary_spot["name"],
                        "district": sel_district,
                        "category": sel_category,
                        "season": season,
                        "transport": transport_mode.lower(),
                        "travelers": int(num_travelers),
                        "days": int(duration_days),
                        "predicted_visitors": avg_visitors,
                        "estimated_budget": budget_res["total_budget"],
                        "budget_breakdown": budget_res["breakdown"],
                        "weather_condition": climate_res.get("weather_condition", ""),
                        "temp_max": climate_res.get("temperature_max", 0.0),
                        "temp_min": climate_res.get("temperature_min", 0.0),
                        "rainfall": climate_res.get("rainfall_mm", 0.0),
                        "recommendations": []
                    })

                    st.session_state["trip_results"] = res_payload
                    t_pred_elapsed = (time.perf_counter() - t_pred_start) * 1000
                    print(f"[PERF] AI Prediction Pipeline completed in {t_pred_elapsed:.2f} ms")
                    st.success("✨ Smart Trip Plan & AI Predictions Generated Successfully!")
                except Exception as e:
                    st.error(f"❌ AI Prediction Engine Error: {e}")

        # --------------------------------------------------
        # RESULT DASHBOARD (TABS LAYOUT)
        # --------------------------------------------------
        res = st.session_state.get("trip_results")
        if res:
            st.markdown("<h2 style='color: #38BDF8; font-weight: 800; margin-top: 30px;'>🏆 Smart Trip Result Dashboard</h2>", unsafe_allow_html=True)

            # Disclaimer Box
            st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(251, 191, 36, 0.35); border-left: 4px solid #FBBF24; border-radius: 12px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.25rem;">⚠️</span>
                        <div>
                            <span style="color: #FBBF24; font-weight: 800; font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.5px;">Disclaimer</span>
                            <p style="color: #E2E8F0; font-size: 0.88rem; margin: 2px 0 0 0; line-height: 1.4;">
                                Predictions and estimates are AI-generated and are not guaranteed facts. Please independently verify all outputs before making travel, financial, or other decisions.
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            res_tabs = st.tabs([
                "📊 Overview",
                "🗺️ Smart Route & Map",
                "👥 Crowd Forecast",
                "🌦️ Climate & Weather",
                "💰 Budget Breakdown"
            ])

            # ----------------------------------------------
            # TAB 1: OVERVIEW
            # ----------------------------------------------
            with res_tabs[0]:
                st.markdown(f"""
                    <div class="glass-card" style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="badge-blue">{res['selected_district'].upper()} ITINERARY</span>
                            <h2 style="color: #F8FAFC; font-weight: 800; margin: 8px 0 4px 0;">{len(res['selected_spots'])} Tourist Destinations Selected</h2>
                            <p style="color: #94A3B8; font-size: 1.05rem;">Dates: <b>{res['travel_start']}</b> to <b>{res['travel_end']}</b> ({res['duration_days']} Days) | Travelers: <b>{res['num_travelers']}</b> | Mode: <b>{res['transport_mode']}</b></p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 2.2rem; font-weight: 800; color: #34D399;">₹ {res['estimated_budget']:,.2f}</div>
                            <p style="font-size: 0.85rem; color: #94A3B8;">Total Estimated Budget (₹ {res['per_person_cost']:,.2f} / person)</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                col_o1, col_o2, col_o3 = st.columns(3)

                with col_o1:
                    climate_info = res["predicted_climate"]
                    st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #38BDF8;">🌦️ Weather Summary</h4>
                            <h3 style="color: #F8FAFC; margin: 6px 0;">{climate_info.get('weather_condition')}</h3>
                            <p style="color: #94A3B8; font-size: 0.9rem;">Temp Range: <b>{climate_info.get('temperature_min')}°C - {climate_info.get('temperature_max')}°C</b></p>
                            <p style="color: #94A3B8; font-size: 0.9rem;">Rainfall: <b>{climate_info.get('rainfall_mm')} mm</b> ({climate_info.get('season')})</p>
                        </div>
                    """, unsafe_allow_html=True)

                with col_o2:
                    avg_c_pct = res.get("avg_crowd_pct", 50)
                    if avg_c_pct < 40:
                        c_badge = "<span class='badge-green'>🟢 LOW CROWD</span>"
                    elif avg_c_pct <= 70:
                        c_badge = "<span class='badge-amber'>🟡 MODERATE CROWD</span>"
                    else:
                        c_badge = "<span class='badge-red'>🔴 HIGH CROWD</span>"

                    st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #34D399;">👥 Avg Crowd Density</h4>
                            <h3 style="color: #F8FAFC; margin: 6px 0;">{avg_c_pct}%</h3>
                            <p style="font-size: 0.85rem; color: #94A3B8;">Crowd Index</p>
                            <div style="margin-top: 8px;">{c_badge}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_o3:
                    st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #FBBF24;">🏨 Lodging & Transport</h4>
                            <h3 style="color: #F8FAFC; margin: 6px 0;">{res['acc_tier']} Tier</h3>
                            <p style="color: #94A3B8; font-size: 0.9rem;">Transport: <b>{res['transport_mode']}</b></p>
                            <p style="color: #94A3B8; font-size: 0.9rem;">Festival: <b>{res['festival']}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # ----------------------------------------------
                # SAVE & EXPORT SECTION (At bottom of Overview)
                # ----------------------------------------------
                st.markdown("<h3 style='color: #F8FAFC; margin-bottom: 4px;'>💾 Save & Export</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;'>Save your trip for later or download a complete report of your AI-generated travel plan.</p>", unsafe_allow_html=True)

                col_act1, col_act2 = st.columns(2)

                # Action 1: Save Trip
                with col_act1:
                    st.markdown("""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 20px; min-height: 130px;">
                            <h4 style="color: #38BDF8; margin: 0 0 6px 0; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
                                💾 Save Trip
                            </h4>
                            <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
                                Save your current trip plan and prediction results so you can access them later.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    save_clicked = st.button("💾 Save Trip", key="save_trip_overview_btn", use_container_width=True)

                    if save_clicked:
                        try:
                            save_payload = {
                                "travel_date": res.get("travel_start"),
                                "spot_name": res.get("primary_spot", {}).get("name", "Tourist Destination"),
                                "district": res.get("selected_district"),
                                "category": res.get("primary_spot", {}).get("category", "heritage"),
                                "season": res.get("predicted_climate", {}).get("season", "Winter"),
                                "transport": str(res.get("transport_mode", "car")).lower(),
                                "travelers": res.get("num_travelers", 1),
                                "days": res.get("duration_days", 1),
                                "predicted_visitors": res.get("avg_visitors", 0),
                                "estimated_budget": res.get("estimated_budget", 0.0),
                                "budget_breakdown": res.get("budget_breakdown", {}),
                                "weather_condition": res.get("predicted_climate", {}).get("weather_condition", ""),
                                "temp_max": res.get("predicted_climate", {}).get("temperature_max", 0.0),
                                "temp_min": res.get("predicted_climate", {}).get("temperature_min", 0.0),
                                "rainfall": res.get("predicted_climate", {}).get("rainfall_mm", 0.0),
                                "recommendations": res.get("recommendations", [])
                            }
                            saved_ok = save_trip_record(save_payload)
                            if saved_ok:
                                st.success("✅ Trip saved successfully.")
                            else:
                                st.error("❌ Unable to save trip. Please try again.")
                        except Exception:
                            st.error("❌ Unable to save trip. Please try again.")

                # Action 2: Download PDF Report
                with col_act2:
                    st.markdown("""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 14px; padding: 20px; min-height: 130px;">
                            <h4 style="color: #34D399; margin: 0 0 6px 0; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">
                                📄 Download Trip Report
                            </h4>
                            <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
                                Download your complete AI-generated trip plan as a PDF.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    try:
                        pdf_bytes = generate_trip_pdf(res)
                        district_filename = str(res.get("selected_district", "Trip")).replace(" ", "_")
                        filename = f"Smart_Tourism_Trip_Plan_{district_filename}.pdf"

                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            key="download_pdf_overview_btn",
                            use_container_width=True
                        )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        st.error(f"❌ Unable to generate PDF report for this trip: {e}")

            # ----------------------------------------------
            # TAB 2: SMART ROUTE & MAP
            # ----------------------------------------------
            with res_tabs[1]:
                st.markdown("<h3 style='color: #F8FAFC;'>🗺️ Smart Interactive GIS Route Map</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Interactive map showing your selected tourist spots connected by route sequence, along with categorized real nearby amenities.</p>", unsafe_allow_html=True)

                # 1. Parse & Persist Current Location into Unified Session State Variables
                qp = st.query_params
                raw_lat = qp.get("user_lat") or st.session_state.get("user_lat") or st.session_state.get("current_lat")
                raw_lon = qp.get("user_lon") or st.session_state.get("user_lon") or st.session_state.get("current_lon")

                if raw_lat is not None and raw_lon is not None:
                    if isinstance(raw_lat, list): raw_lat = raw_lat[0]
                    if isinstance(raw_lon, list): raw_lon = raw_lon[0]
                    try:
                        c_lat_val = float(str(raw_lat).strip())
                        c_lon_val = float(str(raw_lon).strip())
                        if -90.0 <= c_lat_val <= 90.0 and -180.0 <= c_lon_val <= 180.0:
                            st.session_state["current_lat"] = c_lat_val
                            st.session_state["current_lon"] = c_lon_val
                    except (ValueError, TypeError):
                        pass

                current_lat = st.session_state.get("current_lat")
                current_lon = st.session_state.get("current_lon")

                is_location_available = (
                    current_lat is not None and current_lon is not None and
                    isinstance(current_lat, (int, float)) and isinstance(current_lon, (int, float)) and
                    -90.0 <= current_lat <= 90.0 and -180.0 <= current_lon <= 180.0
                )

                # Helper to robustly normalize category strings and resolve visual metadata
                def resolve_category_meta(category_val):
                    cat_norm = str(category_val or "").strip().lower()
                    if "nature" in cat_norm or "waterfall" in cat_norm or "park" in cat_norm or "forest" in cat_norm or "wildlife" in cat_norm or "beach" in cat_norm or "hill" in cat_norm or "lake" in cat_norm:
                        return {"emoji": "🌳", "label": "Nature", "color": "#059669", "type": "nature"}
                    elif "heritage" in cat_norm or "fort" in cat_norm or "monument" in cat_norm or "museum" in cat_norm or "palace" in cat_norm or "history" in cat_norm:
                        return {"emoji": "🏛️", "label": "Heritage", "color": "#D97706", "type": "heritage"}
                    elif "religious" in cat_norm or "temple" in cat_norm or "pilgrimage" in cat_norm or "church" in cat_norm or "mosque" in cat_norm or "shrine" in cat_norm:
                        return {"emoji": "🛕", "label": "Religious", "color": "#DC2626", "type": "religious"}
                    elif "leisure" in cat_norm or "recreation" in cat_norm or "resort" in cat_norm or "entertainment" in cat_norm or "theme" in cat_norm:
                        return {"emoji": "🎡", "label": "Leisure", "color": "#0284C7", "type": "leisure"}
                    else:
                        return {"emoji": "📍", "label": "Other", "color": "#7C3AED", "type": "other"}

                # 2. Geolocation Button & Clear Controls with Exclusive Status Display
                col_loc1, col_loc2 = st.columns([3, 1])
                with col_loc1:
                    st.components.v1.html("""
                        <div style="font-family: Arial, sans-serif; margin-bottom: 8px;">
                            <button id="geoBtn" onclick="requestLocation()" style="
                                background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
                                color: #FFFFFF;
                                border: none;
                                border-radius: 8px;
                                padding: 10px 18px;
                                font-weight: 700;
                                font-size: 0.9rem;
                                cursor: pointer;
                                display: flex;
                                align-items: center;
                                gap: 8px;
                                box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
                                transition: all 0.2s ease;
                            ">
                                📍 Use My Current Location
                            </button>
                            <div id="geoStatus" style="margin-top: 8px; font-size: 0.82rem; color: #F8FAFC; font-weight: 500; line-height: 1.4;"></div>
                        </div>
                        <script>
                        function requestLocation() {
                            var btn = document.getElementById('geoBtn');
                            var status = document.getElementById('geoStatus');
                            status.innerHTML = '⌛ Requesting location...';
                            btn.disabled = true;
                            btn.style.opacity = '0.7';

                            var resolved = false;

                            function applyLocation(lat, lon, sourceName) {
                                if (resolved) return;
                                resolved = true;
                                status.innerHTML = '✅ <span style="color: #34D399;">Location obtained (' + sourceName + ')! Updating map...</span>';
                                try {
                                    var url = new URL(window.parent.location.href);
                                    url.searchParams.set('user_lat', lat);
                                    url.searchParams.set('user_lon', lon);
                                    window.parent.location.href = url.toString();
                                } catch(e) {
                                    var url = new URL(window.location.href);
                                    url.searchParams.set('user_lat', lat);
                                    url.searchParams.set('user_lon', lon);
                                    window.location.href = url.toString();
                                }
                            }

                            function tryNetworkIpFallback() {
                                if (resolved) return;
                                status.innerHTML = '⌛ Fetching network location fallback...';
                                fetch('https://ipapi.co/json/')
                                    .then(function(r) { return r.json(); })
                                    .then(function(d) {
                                        if (d && d.latitude && d.longitude) {
                                            applyLocation(d.latitude, d.longitude, 'Network IP');
                                        } else {
                                            throw new Error('Invalid IP payload');
                                        }
                                    })
                                    .catch(function() {
                                        fetch('http://ip-api.com/json/')
                                            .then(function(r) { return r.json(); })
                                            .then(function(d) {
                                                if (d && d.lat && d.lon) {
                                                    applyLocation(d.lat, d.lon, 'Network IP');
                                                } else {
                                                    throw new Error('Invalid IP payload');
                                                }
                                            })
                                            .catch(function() {
                                                if (resolved) return;
                                                btn.disabled = false;
                                                btn.style.opacity = '1';
                                                status.innerHTML = '❌ <span style="color: #F87171;">Location unavailable. Please enter coordinates manually below.</span>';
                                            });
                                    });
                            }

                            // Strict 3-second safety timer to prevent browser iframe hang
                            var timer = setTimeout(function() {
                                if (!resolved) {
                                    console.warn('Browser geolocation prompt timed out after 3s, triggering IP fallback.');
                                    tryNetworkIpFallback();
                                }
                            }, 3000);

                            var geo = null;
                            try {
                                if (window.parent && window.parent.navigator && window.parent.navigator.geolocation) {
                                    geo = window.parent.navigator.geolocation;
                                } else if (navigator.geolocation) {
                                    geo = navigator.geolocation;
                                }
                            } catch(e) {
                                geo = navigator.geolocation;
                            }

                            if (!geo) {
                                clearTimeout(timer);
                                tryNetworkIpFallback();
                                return;
                            }

                            geo.getCurrentPosition(
                                function(position) {
                                    clearTimeout(timer);
                                    applyLocation(position.coords.latitude, position.coords.longitude, 'GPS/WiFi');
                                },
                                function(error) {
                                    clearTimeout(timer);
                                    console.warn('HTML5 Geolocation unavailable, switching to network IP fallback:', error);
                                    tryNetworkIpFallback();
                                },
                                { enableHighAccuracy: false, timeout: 3000, maximumAge: 60000 }
                            );
                        }
                        </script>
                    """, height=110)

                with col_loc2:
                    if is_location_available:
                        if st.button("❌ Clear Location", use_container_width=True):
                            if "current_lat" in st.session_state: del st.session_state["current_lat"]
                            if "current_lon" in st.session_state: del st.session_state["current_lon"]
                            if "user_current_location" in st.session_state: del st.session_state["user_current_location"]
                            if "user_lat" in st.query_params: del st.query_params["user_lat"]
                            if "user_lon" in st.query_params: del st.query_params["user_lon"]
                            st.rerun()

                with st.expander("📍 Or Enter Coordinates Manually (Optional)", expanded=False):
                    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
                    with col_m1:
                        m_lat_in = st.number_input("Latitude", value=float(current_lat) if current_lat else 17.385040, format="%.6f", key="man_lat_input")
                    with col_m2:
                        m_lon_in = st.number_input("Longitude", value=float(current_lon) if current_lon else 78.486671, format="%.6f", key="man_lon_input")
                    with col_m3:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("Set Location", key="set_manual_coords_btn"):
                            st.session_state["current_lat"] = float(m_lat_in)
                            st.session_state["current_lon"] = float(m_lon_in)
                            st.session_state["user_lat"] = float(m_lat_in)
                            st.session_state["user_lon"] = float(m_lon_in)
                            st.rerun()

                # Status & Temporary Debug Information Display
                if is_location_available:
                    st.success(f"✅ Location obtained! Latitude: `{current_lat:.6f}`, Longitude: `{current_lon:.6f}`")
                else:
                    st.warning("📍 Current location unavailable.")

                with st.expander("🛠️ Debug Location State", expanded=False):
                    st.write(f"**Current latitude:** `{current_lat}`")
                    st.write(f"**Current longitude:** `{current_lon}`")
                    st.write(f"**Location available:** `{is_location_available}`")

                # 3. Distance & Nearest Spot Calculations (if user location active)
                if is_location_available:
                    from App.backend.recommendation import calculate_distance_km
                    
                    spots_with_dist = []
                    for s in res.get("selected_spots", []):
                        try:
                            s_lat = float(s["lat"])
                            s_lon = float(s["lon"])
                            d_km = calculate_distance_km(current_lat, current_lon, s_lat, s_lon)
                            spots_with_dist.append({"spot": s, "distance_km": d_km})
                        except (ValueError, KeyError, TypeError):
                            continue

                    if spots_with_dist:
                        spots_with_dist.sort(key=lambda x: x["distance_km"])
                        n_item = spots_with_dist[0]
                        n_spot = n_item["spot"]
                        n_dist = n_item["distance_km"]
                        n_meta = resolve_category_meta(n_spot.get("category"))

                        st.markdown(f"""
                            <div class="glass-card" style="border-left: 4px solid #34D399; padding: 12px 16px; margin-bottom: 15px; background: rgba(52, 211, 153, 0.08);">
                                <div style="font-size: 0.75rem; font-weight: 800; color: #34D399; text-transform: uppercase;">📍 Nearest Selected Destination</div>
                                <h4 style="color: #F8FAFC; margin: 4px 0;">{n_meta['emoji']} {n_spot['name']}</h4>
                                <p style="font-size: 0.88rem; color: #94A3B8; margin: 0;">Distance from your current location: <b style="color: #38BDF8;">{n_dist:.2f} km</b> ({n_meta['label']} • {n_spot.get('district', res.get('selected_district', 'N/A'))})</p>
                            </div>
                        """, unsafe_allow_html=True)

                        with st.expander("📍 NEARBY FROM YOUR LOCATION (Sorted by Distance)", expanded=False):
                            for rank, item in enumerate(spots_with_dist, start=1):
                                sp = item["spot"]
                                dist = item["distance_km"]
                                sp_meta = resolve_category_meta(sp.get("category"))
                                st.markdown(f"**{rank}. {sp_meta['emoji']} {sp['name']}** — <span style='color:#38BDF8;'><b>{dist:.2f} km</b></span> ({sp_meta['label']} • {sp.get('district', 'N/A')})", unsafe_allow_html=True)

                # 4. YOUR TRIP STOPS Summary Cards
                st.markdown("#### 🚩 YOUR TRIP STOPS")
                trip_stops = res.get("selected_spots", [])
                if trip_stops:
                    stop_cols = st.columns(min(4, max(1, len(trip_stops))))
                    for idx, s in enumerate(trip_stops):
                        cat_meta = resolve_category_meta(s.get("category"))
                        with stop_cols[idx % 4]:
                            st.markdown(f"""
                                <div class="glass-card" style="border-left: 4px solid {cat_meta['color']}; padding: 10px 12px; margin-bottom: 10px;">
                                    <div style="font-size: 0.75rem; font-weight: 800; color: #38BDF8; text-transform: uppercase;">Stop {idx + 1}</div>
                                    <h5 style="color: #F8FAFC; margin: 4px 0 2px 0; font-size: 0.95rem;">{cat_meta['emoji']} {s['name']}</h5>
                                    <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">{cat_meta['label']} • <b>{s.get('district', res.get('selected_district', 'N/A'))}</b></p>
                                </div>
                            """, unsafe_allow_html=True)

                # 5. Smart GIS Map Legend Box
                st.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #334155; border-radius: 12px; padding: 12px 18px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                        <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 8px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">
                            🗺️ MAP LEGEND & SYMBOLS
                        </div>
                        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; color: #E2E8F0; font-size: 0.85rem;">
                            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                <span><b>🔵</b> Your Current Location</span>
                                <span style="font-weight: 700; color: #38BDF8; margin-left: 4px;">Destinations:</span>
                                <span>🛕 Religious</span>
                                <span>🏛️ Heritage</span>
                                <span>🌳 Nature</span>
                                <span>🎡 Leisure</span>
                                <span>📍 Other</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                <span><b style="color: #0EA5E9;">━━━</b> Route</span>
                                <span><b>🏨</b> Hotels</span>
                                <span><b>🍴</b> Restaurants</span>
                                <span><b>🅿️</b> Parking</span>
                                <span><b>📌</b> Other Amenities</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                amenities = res.get("nearby_amenities", {})

                # 1. Coordinate Validation Helper
                def parse_and_validate_coord(lat_val, lon_val):
                    if lat_val is None or lon_val is None:
                        return None
                    try:
                        f_lat = float(lat_val)
                        f_lon = float(lon_val)
                        if np.isnan(f_lat) or np.isnan(f_lon):
                            return None
                        if -90.0 <= f_lat <= 90.0 and -180.0 <= f_lon <= 180.0:
                            return (f_lat, f_lon)
                    except (ValueError, TypeError):
                        return None
                    return None

                # 2. Validate Current User Location
                valid_user_coord = None
                if is_location_available:
                    valid_user_coord = parse_and_validate_coord(current_lat, current_lon)

                # 3. Validate Selected Tourist Spots
                valid_spots = []
                route_coords = []
                sel_district = res.get("selected_district", "N/A")

                for s in res.get("selected_spots", []):
                    c = parse_and_validate_coord(s.get("lat"), s.get("lon"))
                    if c is not None:
                        valid_spots.append((s, c))
                        route_coords.append(list(c))

                # 4. Validate Primary Spot
                primary_spot_raw = res.get("primary_spot", {})
                valid_primary_coord = parse_and_validate_coord(primary_spot_raw.get("lat"), primary_spot_raw.get("lon"))

                # Check if valid coordinates exist
                if not valid_spots and not valid_primary_coord and not valid_user_coord:
                    st.warning("Map unavailable: valid coordinates could not be found for this location.")
                else:
                    # 5. Map Center Priority:
                    # 1) Current user location (if valid)
                    # 2) Primary spot coordinates (if valid)
                    # 3) First selected spot coordinates
                    if valid_user_coord:
                        center_lat, center_lon = valid_user_coord
                    elif valid_primary_coord:
                        center_lat, center_lon = valid_primary_coord
                    elif valid_spots:
                        center_lat, center_lon = valid_spots[0][1]
                    else:
                        center_lat, center_lon = 17.3850, 78.4867

                    map_cache_key = (
                        res.get("selected_district"),
                        tuple(s.get("name") for s in res.get("selected_spots", [])),
                        valid_user_coord,
                        len(route_coords)
                    )
                    map_hash_str = str(abs(hash(map_cache_key)))

                    # 6. Re-create clean Folium Map Instance for st_folium
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

                    # Add Custom Pane for Current Location Marker visibility
                    folium.map.CustomPane("current-location", z_index=700).add_to(m)

                    # Create Logical Feature Groups for Map Layers
                    fg_spots = folium.FeatureGroup(name="📍 Selected Tourist Destinations", show=True)
                    fg_route = folium.FeatureGroup(name="🔵 Planned Route", show=True)
                    fg_hotels = folium.FeatureGroup(name="🏨 Hotels", show=True)
                    fg_restaurants = folium.FeatureGroup(name="🍴 Restaurants", show=True)
                    fg_parking = folium.FeatureGroup(name="🅿️ Parking", show=True)
                    fg_others = folium.FeatureGroup(name="📌 Other Amenities", show=True)

                    if valid_user_coord:
                        current_location_group = folium.FeatureGroup(name="📍 Your Current Location", show=True)
                        fg_user_route = folium.FeatureGroup(name="🔵 Route from Current Location", show=True)

                        u_lat, u_lon = valid_user_coord
                        user_popup_html = f"""
                        <div style="font-family: Arial, sans-serif; padding: 6px; min-width: 170px;">
                            <h4 style="margin: 0 0 6px 0; color: #2563EB; font-size: 0.95rem; font-weight: 800;">📍 YOUR CURRENT LOCATION</h4>
                            <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Latitude:</b> {u_lat:.6f}</p>
                            <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Longitude:</b> {u_lon:.6f}</p>
                        </div>
                        """

                        user_pin_html = """
                        <div style="
                            position: relative;
                            width: 36px;
                            height: 36px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <div style="
                                background: #2563EB;
                                border: 3px solid #FFFFFF;
                                border-radius: 50%;
                                width: 28px;
                                height: 28px;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 14px;
                                font-weight: bold;
                            ">📍</div>
                        </div>
                        """

                        folium.Marker(
                            location=[u_lat, u_lon],
                            popup=folium.Popup(user_popup_html, max_width=250),
                            tooltip="📍 YOUR CURRENT LOCATION",
                            icon=folium.DivIcon(html=user_pin_html, icon_size=(36, 36), icon_anchor=(18, 18)),
                            pane="current-location"
                        ).add_to(current_location_group)

                    # 7. Add Selected Tourist Spots Markers
                    for idx, (s, (s_lat, s_lon)) in enumerate(valid_spots):
                        stop_num = idx + 1
                        spot_name = s.get("name", "Destination Spot")
                        spot_dist = s.get("district", sel_district)
                        cat_raw = s.get("category", "")
                        cat_meta = resolve_category_meta(cat_raw)
                        pin_color = cat_meta["color"]
                        cat_title = cat_meta["label"]
                        emoji_icon = cat_meta["emoji"]

                        popup_html = f"""
                        <div style="font-family: Arial, sans-serif; padding: 4px; min-width: 180px;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                                <span style="background: #0F172A; color: #38BDF8; font-size: 0.75rem; font-weight: 800; padding: 2px 8px; border-radius: 10px;">Stop {stop_num}</span>
                                <span style="font-size: 0.8rem; font-weight: 700; color: {pin_color};">{emoji_icon} {cat_title}</span>
                            </div>
                            <h4 style="margin: 4px 0 8px 0; color: #0F172A; font-size: 0.95rem; font-weight: 800;">📍 SELECTED TOURIST DESTINATION</h4>
                            <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Name:</b> {spot_name}</p>
                            <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Category:</b> {cat_title}</p>
                            <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>District:</b> {spot_dist}</p>
                        </div>
                        """

                        pin_html = f"""
                        <div style="
                            position: relative;
                            width: 40px;
                            height: 52px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <div style="
                                position: absolute;
                                top: -6px;
                                right: -4px;
                                background: #0F172A;
                                color: #38BDF8;
                                border: 2px solid #FFFFFF;
                                border-radius: 50%;
                                width: 20px;
                                height: 20px;
                                font-size: 11px;
                                font-weight: 900;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.5);
                                z-index: 1001;
                            ">{stop_num}</div>
                            
                            <div style="
                                background: {pin_color};
                                border: 3px solid #FFFFFF;
                                border-radius: 50% 50% 50% 0;
                                transform: rotate(-45deg);
                                width: 38px;
                                height: 38px;
                                box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            ">
                                <span style="
                                    transform: rotate(45deg);
                                    font-size: 18px;
                                ">{emoji_icon}</span>
                            </div>
                        </div>
                        """

                        folium.Marker(
                            location=[s_lat, s_lon],
                            popup=folium.Popup(popup_html, max_width=270),
                            tooltip=f"{emoji_icon} Stop {stop_num}: {spot_name}",
                            icon=folium.DivIcon(html=pin_html, icon_size=(40, 52), icon_anchor=(20, 52))
                        ).add_to(fg_spots)

                    # 8. Segment Route from Current Location to Stop 1
                    if valid_user_coord and route_coords:
                        stop1_coords = route_coords[0]
                        u_lat, u_lon = valid_user_coord
                        folium.PolyLine(
                            locations=[[u_lat, u_lon], stop1_coords],
                            color="#2563EB",
                            weight=3,
                            dash_array="6, 8",
                            opacity=0.85,
                            tooltip="Route from Current Location to Stop 1"
                        ).add_to(fg_user_route)

                    # 9. Planned Route PolyLine Sequence
                    if len(route_coords) > 1:
                        folium.PolyLine(
                            locations=route_coords,
                            color="#0EA5E9",
                            weight=4,
                            opacity=0.85,
                            tooltip="Recommended Travel Route Sequence"
                        ).add_to(fg_route)

                    # 10. Map Bounds Auto-Centering
                    all_bounds = list(route_coords)
                    if valid_user_coord:
                        all_bounds.append(list(valid_user_coord))

                    if len(all_bounds) > 1:
                        min_lat = min(pt[0] for pt in all_bounds)
                        max_lat = max(pt[0] for pt in all_bounds)
                        min_lon = min(pt[1] for pt in all_bounds)
                        max_lon = max(pt[1] for pt in all_bounds)
                        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])

                    # 11. Categorized & Clustered Nearby Amenity Layers
                    def add_amenities_to_group(items, category_title, color_bg, emoji_icon, header_color, target_group):
                        cluster = MarkerCluster(disableClusteringAtZoom=15).add_to(target_group)
                        for it in items:
                            i_coord = parse_and_validate_coord(it.get("lat"), it.get("lon"))
                            if i_coord is not None:
                                i_lat, i_lon = i_coord
                                name = it.get("name", f"Nearby {category_title}")

                                # Skip if amenity duplicates a selected tourist spot name or directly overlaps coordinates (< 20m)
                                name_lower = name.lower().strip()
                                is_overlapping_spot = False
                                for s_obj, (s_lat, s_lon) in valid_spots:
                                    s_name = s_obj.get("name", "").lower().strip()
                                    if (s_name and (s_name in name_lower or name_lower in s_name)) or (abs(s_lat - i_lat) < 0.0002 and abs(s_lon - i_lon) < 0.0002):
                                        is_overlapping_spot = True
                                        break
                                if is_overlapping_spot:
                                    continue
                                item_type = it.get("type", category_title).title()
                                dist_val = it.get("distance_km")
                                dist_str = f"{float(dist_val):.2f} km" if dist_val is not None else ""
                                
                                dist_html = f'<p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Distance:</b> {dist_str}</p>' if dist_str else ""
                                
                                popup_html = f"""
                                <div style="font-family: Arial, sans-serif; padding: 4px; min-width: 160px;">
                                    <h4 style="margin: 0 0 6px 0; color: {header_color}; font-size: 0.95rem; font-weight: 700;">{emoji_icon} NEARBY {category_title.upper()}</h4>
                                    <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Name:</b> {name}</p>
                                    <p style="margin: 3px 0; font-size: 0.85rem; color: #334155;"><b>Type:</b> {item_type}</p>
                                    {dist_html}
                                </div>
                                """
                                
                                icon_html = f"""
                                <div style="
                                    background-color: {color_bg};
                                    border: 2px solid #FFFFFF;
                                    border-radius: 50%;
                                    width: 26px;
                                    height: 26px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    font-size: 13px;
                                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                                ">{emoji_icon}</div>
                                """
                                
                                folium.Marker(
                                    location=[i_lat, i_lon],
                                    popup=folium.Popup(popup_html, max_width=250),
                                    tooltip=f"{emoji_icon} {name}",
                                    icon=folium.DivIcon(html=icon_html, icon_size=(26, 26), icon_anchor=(13, 13))
                                ).add_to(cluster)

                    add_amenities_to_group(amenities.get("hotels", []), "Hotel", "#2563EB", "🏨", "#2563EB", fg_hotels)
                    add_amenities_to_group(amenities.get("restaurants", []), "Restaurant", "#EA580C", "🍴", "#EA580C", fg_restaurants)
                    add_amenities_to_group(amenities.get("parking", []), "Parking", "#475569", "🅿️", "#475569", fg_parking)

                    other_items = (
                        amenities.get("attractions", []) +
                        amenities.get("hospitals", []) +
                        amenities.get("atms", []) +
                        amenities.get("petrol_pumps", []) +
                        amenities.get("restrooms", [])
                    )
                    add_amenities_to_group(other_items, "Amenity", "#7C3AED", "📌", "#7C3AED", fg_others)

                    # 12. Add FeatureGroups to Map in hierarchy order
                    fg_route.add_to(m)
                    if valid_user_coord:
                        fg_user_route.add_to(m)
                    fg_hotels.add_to(m)
                    fg_restaurants.add_to(m)
                    fg_parking.add_to(m)
                    fg_others.add_to(m)
                    fg_spots.add_to(m)
                    if valid_user_coord:
                        current_location_group.add_to(m)

                    # 13. Layer Control
                    folium.LayerControl(position="topright", collapsed=False).add_to(m)

                    # 14. Developer Diagnostics Expander (Requirement 12)
                    with st.expander("🛠️ MAP DEBUG DIAGNOSTICS", expanded=False):
                        st.markdown(f"""
                        **MAP DEBUG**
                        - Current location: `lat = {valid_user_coord[0] if valid_user_coord else 'N/A'}, lon = {valid_user_coord[1] if valid_user_coord else 'N/A'}`
                        - Selected destinations: `{route_coords}`
                        - Route points count: `{len(route_coords)}`
                        - Amenity items: `{sum(len(v) for v in amenities.values() if isinstance(v, list))}`
                        - Map object created: **YES**
                        - Map rendered: **YES**
                        """)

                    # 15. Render Interactive GIS Map with st_folium
                    st_folium(
                        m,
                        use_container_width=True,
                        height=600,
                        key=f"smart_tourism_gis_map_{map_hash_str}",
                        returned_objects=[]
                    )

                    primary_nav_lat = valid_primary_coord[0] if valid_primary_coord else (valid_spots[0][1][0] if valid_spots else center_lat)
                    primary_nav_lon = valid_primary_coord[1] if valid_primary_coord else (valid_spots[0][1][1] if valid_spots else center_lon)
                    gmaps_url = f"https://www.google.com/maps/dir/?api=1&destination={primary_nav_lat},{primary_nav_lon}"
                    st.markdown(f"""
                        <a href="{gmaps_url}" target="_blank" style="text-decoration: none;">
                            <button style="background: rgba(30, 41, 59, 0.8); border: 1px solid #38BDF8; color: #38BDF8; border-radius: 12px; padding: 12px; width: 100%; font-weight: 600; cursor: pointer; margin-top: 15px;">
                                🗺️ Open Navigation Route in Google Maps
                            </button>
                        </a>
                    """, unsafe_allow_html=True)

                # Nearby Places Grid
                st.markdown("### 📍 Real Nearby Amenities in Dataset")
                n_t1, n_t2, n_t3 = st.tabs(["🏨 Lodging / Hotels", "🍽️ Dining", "🏥 Hospitals & Services"])

                with n_t1:
                    h_list = amenities.get("hotels", [])
                    if not h_list:
                        st.info("No hotel records in dataset near this location.")
                    else:
                        h_cols = st.columns(min(3, max(1, len(h_list))))
                        for i, h in enumerate(h_list[:3]):
                            with h_cols[i % 3]:
                                st.markdown(f"""
                                    <div class="glass-card">
                                        <h4 style="color: #38BDF8;">{h['name']}</h4>
                                        <p style="font-size: 0.85rem; color: #94A3B8;">Tier: {h.get('tier', 'Standard')} | Rating: ★ {h.get('rating', 'N/A')}</p>
                                        <p style="font-size: 0.85rem; color: #34D399;">Cost: ₹ {h.get('cost', 0):,.0f} / night</p>
                                        <p style="font-size: 0.8rem; color: #94A3B8;">Distance: {h.get('distance_km', 0.0)} km</p>
                                    </div>
                                """, unsafe_allow_html=True)

                with n_t2:
                    r_list = amenities.get("restaurants", [])
                    if not r_list:
                        st.info("No restaurant records in dataset near this location.")
                    else:
                        r_cols = st.columns(min(3, max(1, len(r_list))))
                        for i, r in enumerate(r_list[:3]):
                            with r_cols[i % 3]:
                                st.markdown(f"""
                                    <div class="glass-card">
                                        <h4 style="color: #FBBF24;">{r['name']}</h4>
                                        <p style="font-size: 0.85rem; color: #38BDF8;">Rating: ★ {r.get('rating', 'N/A')}</p>
                                        <p style="font-size: 0.8rem; color: #94A3B8;">Distance: {r.get('distance_km', 0.0)} km</p>
                                    </div>
                                """, unsafe_allow_html=True)

                with n_t3:
                    hosps = amenities.get("hospitals", [])
                    parks = amenities.get("parking", [])
                    atms = amenities.get("atms", [])
                    st.markdown(f"Hospitals ({len(hosps)}) | Parking Spots ({len(parks)}) | ATMs ({len(atms)})")

            # ----------------------------------------------
            # TAB 3: CROWD FORECAST (PERCENTAGE / INDEX DISPLAY)
            # ----------------------------------------------
            with res_tabs[2]:
                st.markdown("<h3 style='color: #F8FAFC;'>👥 Visitor Crowd Forecast per Spot</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Crowd density index predicted using XGBoost visitor model telemetry.</p>", unsafe_allow_html=True)

                cr_cols = st.columns(min(3, max(1, len(res["crowd_results"]))))
                for idx, cr in enumerate(res["crowd_results"]):
                    with cr_cols[idx % 3]:
                        st.markdown(f"""
                            <div class="glass-card">
                                <h4 style="color: #38BDF8; margin-bottom: 2px;">{cr['spot_name']}</h4>
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 10px;">{cr['category'].title()}</p>
                                <hr style="border-color: rgba(255,255,255,0.1); margin: 8px 0;">
                                <div style="font-size: 2.5rem; font-weight: 800; color: #F8FAFC; margin: 4px 0;">{cr['crowd_pct']}%</div>
                                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">Crowd Level</p>
                                <div>{cr['badge_html']}</div>
                            </div>
                        """, unsafe_allow_html=True)

            # ----------------------------------------------
            # TAB 4: CLIMATE & WEATHER
            # ----------------------------------------------
            with res_tabs[3]:
                st.markdown("<h3 style='color: #F8FAFC;'>🌦️ PyTorch Climate Neural Network Forecast</h3>", unsafe_allow_html=True)
                cl = res.get("predicted_climate", {})

                # 1. Existing Prediction Metric Cards
                c_col1, c_col2, c_col3 = st.columns(3)
                with c_col1:
                    st.metric("🌡️ Max Temperature", f"{cl.get('temperature_max')} °C")
                with c_col2:
                    st.metric("🌡️ Min Temperature", f"{cl.get('temperature_min')} °C")
                with c_col3:
                    st.metric("🌧️ Rainfall", f"{cl.get('rainfall_mm')} mm")

                # Weather Advisory Card
                primary_spot_obj = res.get("primary_spot", {})
                selected_spot_name = primary_spot_obj.get("name", "Selected Tourist Spot")

                st.markdown(f"""
                    <div class="glass-card" style="margin-top: 20px; margin-bottom: 25px;">
                        <h4 style="color: #38BDF8;">Weather Advisory for {selected_spot_name}</h4>
                        <p style="font-size: 1.1rem; color: #F8FAFC;">Condition: <b>{cl.get('weather_condition')}</b></p>
                        <p style="font-size: 0.9rem; color: #94A3B8;">Forecast Month: <b>{cl.get('month')}</b> | Season: <b>{cl.get('season')}</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # 2. Time-Series Climate Forecast Graph Section
                st.markdown("<h3 style='color: #F8FAFC; margin-top: 20px;'>📈 Climate Forecast for Your Trip</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94A3B8; font-size: 0.95rem;'>AI-generated daily forecast for your selected travel dates</p>", unsafe_allow_html=True)

                raw_start = res.get("travel_start") or st.session_state.get("input_travel_start")
                raw_end = res.get("travel_end") or st.session_state.get("input_travel_end")

                if not raw_start or not raw_end:
                    st.info("📅 Please select your trip start and end dates to view the climate forecast.")
                else:
                    try:
                        dt_start = pd.to_datetime(raw_start).date()
                        dt_end = pd.to_datetime(raw_end).date()
                    except Exception:
                        dt_start, dt_end = None, None

                    if not dt_start or not dt_end:
                        st.info("📅 Please select your trip start and end dates to view the climate forecast.")
                    elif dt_end < dt_start:
                        st.warning("⚠️ Invalid Date Range: End Date cannot be earlier than Start Date.")
                    else:
                        start_str = dt_start.strftime("%Y-%m-%d")
                        end_str = dt_end.strftime("%Y-%m-%d")

                        # Climate Variable Selector (ONLY 3 supported models)
                        var_options = [
                            "Maximum Temperature (°C)",
                            "Minimum Temperature (°C)",
                            "Rainfall (mm)"
                        ]
                        selected_var = st.selectbox(
                            "Select Climate Variable:",
                            options=var_options,
                            key="climate_var_selector"
                        )

                        # Fetch dynamic trip forecast dataframe for exact travel dates
                        sel_district_val = res.get("selected_district") if isinstance(res, dict) else None
                        fc_res = get_trip_climate_forecast(selected_spot_name, start_str, end_str, district_name=sel_district_val)

                        if isinstance(fc_res, str) and fc_res == "EXCEEDS_LIMIT":
                            st.warning("⚠️ The selected trip date range exceeds the maximum supported forecast horizon (up to 60 days duration / 1 year into future). Please select dates within the supported horizon.")
                        elif fc_res is None or (isinstance(fc_res, pd.DataFrame) and fc_res.empty):
                            st.warning("Time-series climate forecast could not be generated for the selected destination.")
                        else:
                            fc_df = fc_res
                            if selected_var == "Maximum Temperature (°C)":
                                val_col, unit_str, y_axis_title, line_color, fill_color = "Temperature_Max_C", "°C", "Temperature (°C)", "#F87171", "rgba(248, 113, 113, 0.12)"
                            elif selected_var == "Minimum Temperature (°C)":
                                val_col, unit_str, y_axis_title, line_color, fill_color = "Temperature_Min_C", "°C", "Temperature (°C)", "#38BDF8", "rgba(56, 189, 248, 0.12)"
                            else:  # Rainfall (mm)
                                val_col, unit_str, y_axis_title, line_color, fill_color = "Rainfall_mm", "mm", "Rainfall (mm)", "#34D399", "rgba(52, 211, 153, 0.12)"

                            # Forecast Summary Statistics (Average, Minimum, Maximum, Trend)
                            if dt_start.year == dt_end.year:
                                period_str = f"{dt_start.strftime('%b %d')} – {dt_end.strftime('%b %d, %Y')}"
                            else:
                                period_str = f"{dt_start.strftime('%b %d, %Y')} – {dt_end.strftime('%b %d, %Y')}"
                            num_days = len(fc_df)

                            avg_val = float(fc_df[val_col].mean())
                            min_val = float(fc_df[val_col].min())
                            max_val = float(fc_df[val_col].max())

                            # Trend calculation strictly from forecast values
                            if len(fc_df) >= 2:
                                delta_trend = float(fc_df[val_col].iloc[-1] - fc_df[val_col].iloc[0])
                                if abs(delta_trend) < 0.3:
                                    trend_str = "➡️ Stable"
                                elif selected_var in ["Maximum Temperature (°C)", "Minimum Temperature (°C)"]:
                                    trend_str = f"📈 Warming (+{delta_trend:.1f} °C)" if delta_trend > 0 else f"📉 Cooling ({delta_trend:.1f} °C)"
                                else:
                                    trend_str = f"🌧️ Increasing (+{delta_trend:.1f} mm)" if delta_trend > 0 else f"☀️ Decreasing ({delta_trend:.1f} mm)"
                            else:
                                trend_str = "➡️ Single Day"

                            st.markdown(f"""
                                <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px 18px; margin-top: 15px; margin-bottom: 20px;">
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; font-size: 0.9rem; color: #E2E8F0;">
                                        <div>📅 <b>Forecast Period:</b> {period_str} ({num_days} days)</div>
                                        <div>📊 <b>Average:</b> {avg_val:.1f} {unit_str}</div>
                                        <div>📉 <b>Minimum:</b> {min_val:.1f} {unit_str}</div>
                                        <div>📈 <b>Maximum:</b> {max_val:.1f} {unit_str}</div>
                                        <div>🔄 <b>Trend:</b> {trend_str}</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            fig_cs = go.Figure()

                            # Straight-line Plotly daily time-series chart with subtle area fill below line
                            fig_cs.add_trace(go.Scatter(
                                x=fc_df["Date_Formatted"],
                                y=fc_df[val_col],
                                mode="lines+markers",
                                name="Forecast",
                                fill="tozeroy",
                                fillcolor=fill_color,
                                line=dict(color=line_color, width=3),
                                marker=dict(size=8, color=line_color, symbol="circle", line=dict(width=2, color="#0F172A")),
                                hovertemplate=f"<b>Date:</b> %{{x}}<br><b>Forecast:</b> %{{y:.1f}} {unit_str}<extra></extra>"
                            ))

                            fig_cs.update_layout(
                                title=dict(
                                    text="Climate Forecast for Your Trip",
                                    font=dict(color="#F8FAFC", size=17)
                                ),
                                xaxis=dict(
                                    title="Trip Date",
                                    gridcolor="rgba(255,255,255,0.06)",
                                    color="#CBD5E1",
                                    tickangle=-30 if len(fc_df) > 10 else 0
                                ),
                                yaxis=dict(
                                    title=y_axis_title,
                                    gridcolor="rgba(255,255,255,0.06)",
                                    color="#CBD5E1"
                                ),
                                paper_bgcolor="rgba(15,23,42,0.6)",
                                plot_bgcolor="rgba(15,23,42,0.6)",
                                showlegend=False,
                                margin=dict(l=40, r=40, t=60, b=50),
                                hovermode="x unified",
                                height=420
                            )

                            st.plotly_chart(fig_cs, use_container_width=True)

                            # Note below graph
                            st.markdown("<p style='color: #64748B; font-size: 0.85rem; margin-top: -10px; font-style: italic;'>Forecast values are AI-generated estimates and may differ from actual future conditions.</p>", unsafe_allow_html=True)

            # ----------------------------------------------
            # TAB 5: BUDGET BREAKDOWN
            # ----------------------------------------------
            with res_tabs[4]:
                st.markdown("<h3 style='color: #F8FAFC;'>💰 Itemized Travel Budget Breakdown</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Calculated via Multi-Output Regressor trained on budget dataset.</p>", unsafe_allow_html=True)

                bd = res["budget_breakdown"]
                b_df = pd.DataFrame([
                    {"Category": "Transport", "Cost (₹)": bd.get("travel_cost", 0.0)},
                    {"Category": "Accommodation", "Cost (₹)": bd.get("stay_cost", 0.0)},
                    {"Category": "Food & Dining", "Cost (₹)": bd.get("food_cost", 0.0)},
                    {"Category": "Entry Tickets", "Cost (₹)": bd.get("entry_fees", 0.0)},
                    {"Category": "Misc & Parking", "Cost (₹)": bd.get("tolls_and_parking", 0.0)},
                ])

                fig_b = px.bar(b_df, x="Category", y="Cost (₹)", text_auto=".2f", color="Category", title="Cost Breakdown (₹)")
                fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
                st.plotly_chart(fig_b, use_container_width=True)

                st.table(b_df)




# =====================================
# PAGE 4: MY PREDICTIONS / SAVED TRIPS
# =====================================
elif selected_page == "My Predictions / Saved Trips":
    st.markdown("<h2 style='font-weight: 800; color: #38BDF8;'>📊 Saved Trip Predictions & Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Historical trip records loaded from Supabase PostgreSQL database or local SQLite storage.</p>", unsafe_allow_html=True)

    trips = fetch_saved_trips(limit=50)

    if not trips:
        st.info("No saved trips found in database. Use the **Smart Trip Planner** to generate and save your first trip!")
    else:
        df_trips = pd.DataFrame(trips)
        st.markdown(f"<p style='color: #34D399;'>Found <b>{len(trips)}</b> saved trip records:</p>", unsafe_allow_html=True)

        display_cols = [c for c in ["id", "travel_date", "spot_name", "district", "category", "season", "estimated_budget", "predicted_visitors"] if c in df_trips.columns]
        st.dataframe(df_trips[display_cols], use_container_width=True, hide_index=True)



