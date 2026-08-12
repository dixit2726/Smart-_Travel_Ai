import sqlite3
import json
import datetime
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from supabase import create_client
try:
    from App.backend.config import SUPABASE_URL, SUPABASE_KEY, BASE_DIR, DB_PATH
except ImportError:
    from config import SUPABASE_URL, SUPABASE_KEY, BASE_DIR, DB_PATH

try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

# 1. Supabase Initialization (Cached Resource)
@cache_resource
def get_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL.startswith("http"):
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[OK] Supabase Connected Successfully!")
            return client
        except Exception as e:
            print(f"[WARNING] Supabase Connection Warning: {e}")
            return None
    else:
        print("[INFO] Supabase credentials not configured. Local SQLite persistence enabled.")
        return None

supabase = get_supabase_client()

# 2. Local SQLite Initialization
OTHERS_DIR = PROJECT_ROOT / "Others"
if not ('DB_PATH' in locals() or 'DB_PATH' in globals()) or not DB_PATH.exists():
    DB_PATH = OTHERS_DIR / "smart_tourism.db" if (OTHERS_DIR / "smart_tourism.db").exists() else BASE_DIR / "smart_tourism.db"


def init_sqlite():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                travel_date TEXT,
                spot_name TEXT,
                district TEXT,
                category TEXT,
                season TEXT,
                transport TEXT,
                travelers INTEGER,
                days INTEGER,
                predicted_visitors INTEGER,
                estimated_budget REAL,
                budget_breakdown TEXT,
                weather_condition TEXT,
                temp_max REAL,
                temp_min REAL,
                rainfall REAL,
                recommendations TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARNING] SQLite Init Error: {e}")

init_sqlite()


def save_trip_record(trip_data: dict) -> bool:
    """Saves trip record to Supabase if available, and always logs to local SQLite."""
    required_keys = ["spot_name", "district", "category", "season", "transport", "travelers", "days", "predicted_visitors", "estimated_budget"]
    for k in required_keys:
        if k not in trip_data or trip_data[k] is None:
            raise ValueError(f"Cannot save trip record: missing required field '{k}'")

    saved_supabase = False
    
    # Format data for DB directly from trip_data
    record = {
        "travel_date": str(trip_data.get("travel_date") or datetime.date.today()),
        "spot_name": str(trip_data["spot_name"]),
        "district": str(trip_data["district"]),
        "category": str(trip_data["category"]),
        "season": str(trip_data["season"]),
        "transport": str(trip_data["transport"]),
        "travelers": int(trip_data["travelers"]),
        "days": int(trip_data["days"]),
        "predicted_visitors": int(trip_data["predicted_visitors"]),
        "estimated_budget": float(trip_data["estimated_budget"]),
        "budget_breakdown": json.dumps(trip_data.get("budget_breakdown") or {}),
        "weather_condition": str(trip_data.get("weather_condition") or ""),
        "temp_max": float(trip_data.get("temp_max") or 0.0),
        "temp_min": float(trip_data.get("temp_min") or 0.0),
        "rainfall": float(trip_data.get("rainfall") or 0.0),
        "recommendations": json.dumps(trip_data.get("recommendations") or [])
    }

    # Save to Supabase
    if supabase is not None:
        for table_name in ["Predictions", "saved_trips"]:
            try:
                supabase.table(table_name).insert(record).execute()
                saved_supabase = True
                print(f"[SUCCESS] Saved trip record to Supabase ('{table_name}' table)!")
                break
            except Exception as e:
                print(f"[INFO] Supabase insert to '{table_name}' skipped: {e}")

    # Always save to SQLite fallback
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO saved_trips (
                travel_date, spot_name, district, category, season, transport,
                travelers, days, predicted_visitors, estimated_budget, budget_breakdown,
                weather_condition, temp_max, temp_min, rainfall, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["travel_date"], record["spot_name"], record["district"], record["category"],
            record["season"], record["transport"], record["travelers"], record["days"],
            record["predicted_visitors"], record["estimated_budget"], record["budget_breakdown"],
            record["weather_condition"], record["temp_max"], record["temp_min"], record["rainfall"],
            record["recommendations"]
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[WARNING] SQLite save error: {e}")
        return saved_supabase


def fetch_saved_trips(limit: int = 50) -> list:
    """Fetches saved trips from Supabase or local SQLite."""
    if supabase is not None:
        for table_name in ["Predictions", "saved_trips"]:
            try:
                res = supabase.table(table_name).select("*").order("id", desc=True).limit(limit).execute()
                if res.data:
                    return res.data
            except Exception as e:
                pass

    # SQLite fallback
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM saved_trips ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["budget_breakdown"] = json.loads(item.get("budget_breakdown") or "{}")
            except Exception:
                pass
            try:
                item["recommendations"] = json.loads(item.get("recommendations") or "[]")
            except Exception:
                pass
            result.append(item)
        conn.close()
        return result
    except Exception as e:
        print(f"[WARNING] Error reading SQLite trips: {e}")
        return []

