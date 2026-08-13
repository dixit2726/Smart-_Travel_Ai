import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

PKL_DIR = APP_DIR / "Pickles" if (APP_DIR / "Pickles").exists() else APP_DIR / "Pkl"
DATA_DIR = PROJECT_ROOT / "Data"
OTHERS_DIR = PROJECT_ROOT / "Others"
DB_PATH = OTHERS_DIR / "smart_tourism.db" if (OTHERS_DIR / "smart_tourism.db").exists() else BASE_DIR / "smart_tourism.db"

CLIMATE_CSV = DATA_DIR / "Climate_Dataset_Final.csv"
SPOTS_CSV = DATA_DIR / "spots.csv" if (DATA_DIR / "spots.csv").exists() else DATA_DIR / "other spots.csv"
AMENITIES_CSV = DATA_DIR / "nearby_amenities.csv"
ACCOMMODATIONS_CSV = DATA_DIR / "accommodations.csv"
VISITORS_CSV = DATA_DIR / "spot_visitors.csv"
BUDGET_CSV = DATA_DIR / "trip_budget_prediction_dataset.csv"


load_dotenv(BASE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Fallback to Streamlit Cloud secrets if running on Streamlit Community Cloud
if not SUPABASE_URL or not SUPABASE_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            SUPABASE_URL = SUPABASE_URL or st.secrets.get("SUPABASE_URL", "") or st.secrets.get("supabase", {}).get("SUPABASE_URL", "")
            SUPABASE_KEY = SUPABASE_KEY or st.secrets.get("SUPABASE_KEY", "") or st.secrets.get("supabase", {}).get("SUPABASE_KEY", "")
    except Exception:
        pass