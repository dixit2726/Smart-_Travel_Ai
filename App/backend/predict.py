import os
import sys
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
torch.set_num_threads(1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

try:
    from App.backend.config import PKL_DIR, DATA_DIR, CLIMATE_CSV, BUDGET_CSV
except ImportError:
    from config import PKL_DIR, DATA_DIR, CLIMATE_CSV, BUDGET_CSV

# =====================================
# PyTorch Climate LSTM Model Definition
# =====================================
class ClimateLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=24, num_layers=1, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:
    def _dummy_cache(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
    cache_resource = _dummy_cache
    cache_data = _dummy_cache

# =====================================
# Model Verification Tracker
# =====================================
verification_status = {
    "climate_model": {"status": "FAILED", "path": "", "type": "ClimateLSTM", "expected_input": "(1, 7, 3)"},
    "crowd_model": {"status": "FAILED", "path": "", "type": "XGBRegressor", "expected_features": 22},
    "budget_model": {"status": "FAILED", "path": "", "type": "MultiOutputRegressor", "expected_features": 13},
    "encoders": {"status": "FAILED", "details": ""},
    "scalers": {"status": "FAILED", "details": ""}
}

# Directories
visitors_dir = PKL_DIR / "spot vistiors"
if not visitors_dir.exists() and (PKL_DIR / "spot_visitors").exists():
    visitors_dir = PKL_DIR / "spot_visitors"
elif not visitors_dir.exists() and (PKL_DIR / "spot visitors").exists():
    visitors_dir = PKL_DIR / "spot visitors"

budget_dir = PKL_DIR / "budget"
if not budget_dir.exists() and (PKL_DIR / "Budget").exists():
    budget_dir = PKL_DIR / "Budget"

climate_dir = PKL_DIR / "Climate"
if not climate_dir.exists() and (PKL_DIR / "climate").exists():
    climate_dir = PKL_DIR / "climate"


@cache_resource
def load_all_ml_models():
    """Loads all ML models, encoders, and scalers ONCE into memory with st.cache_resource."""
    status = verification_status.copy()
    res = {
        "visitors_model": None,
        "onehot_visitor_enc": None,
        "ordinal_visitor_enc": None,
        "ohe_category_festival": None,
        "budget_model": None,
        "budget_preprocessor": None,
        "budget_acc_tier_encoder": None,
        "budget_numeric_scaler": None,
        "budget_onehot_encoder": None,
        "climate_model": None,
        "climate_metadata": None
    }

    # 1. Visitor Crowd Model & Encoders
    try:
        visitors_model_path = visitors_dir / "best_model .pkl"
        onehot_visitor_path = visitors_dir / "onehot_encoder .pkl"
        ordinal_visitor_path = visitors_dir / "ordinal_encoder .pkl"
        status["crowd_model"]["path"] = str(visitors_model_path)

        if visitors_model_path.exists():
            vm = joblib.load(visitors_model_path)
            res["visitors_model"] = vm
            expected_f = getattr(vm, "n_features_in_", 22)
            status["crowd_model"]["status"] = "LOADED"
            status["crowd_model"]["expected_features"] = expected_f

        if onehot_visitor_path.exists() and ordinal_visitor_path.exists():
            res["onehot_visitor_enc"] = joblib.load(onehot_visitor_path)
            res["ordinal_visitor_enc"] = joblib.load(ordinal_visitor_path)
            cat_fest_categories = res["onehot_visitor_enc"].categories_[:2]
            ohe_cf = OneHotEncoder(categories=cat_fest_categories, handle_unknown="ignore")
            dummy_df = pd.DataFrame([["heritage", "None"]], columns=["category", "festival"])
            ohe_cf.fit(dummy_df)
            res["ohe_category_festival"] = ohe_cf
    except Exception as e:
        print(f"[MODEL FAIL] Loading visitor crowd model failed: {e}")

    # 2. Budget Model & Encoders
    try:
        model_path = budget_dir / "best_trip_cost_model.pkl"
        acc_enc_path = budget_dir / "accommodation_tier_encoder.pkl"
        scaler_path = budget_dir / "numeric_scaler.pkl"
        ohe_path = budget_dir / "onehot_encoder.pkl"
        status["budget_model"]["path"] = str(model_path)

        if model_path.exists():
            bm = joblib.load(model_path)
            # Force single-process execution for Streamlit safety (prevents loky multiprocessing serialization errors)
            if hasattr(bm, "n_jobs"):
                bm.n_jobs = 1
            if hasattr(bm, "estimators_"):
                for est in bm.estimators_:
                    if hasattr(est, "n_jobs"):
                        est.n_jobs = 1
            res["budget_model"] = bm
            expected_f = getattr(bm, "n_features_in_", 13)
            status["budget_model"]["status"] = "LOADED"
            status["budget_model"]["expected_features"] = expected_f

        if acc_enc_path.exists() and scaler_path.exists() and ohe_path.exists():
            res["budget_acc_tier_encoder"] = joblib.load(acc_enc_path)
            res["budget_numeric_scaler"] = joblib.load(scaler_path)
            res["budget_onehot_encoder"] = joblib.load(ohe_path)

        if BUDGET_CSV.exists():
            budget_raw_df = pd.read_csv(BUDGET_CSV)
            X_b = budget_raw_df[['accommodation_tier', 'transport_mode', 'season', 'duration_days', 'num_travelers', 'route_distance_km']]
            bp = ColumnTransformer(
                transformers=[
                    ('ordinal', OrdinalEncoder(), ['accommodation_tier']),
                    ('nominal', OneHotEncoder(handle_unknown='ignore'), ['transport_mode', 'season']),
                    ('numeric', StandardScaler(), ['duration_days', 'num_travelers', 'route_distance_km'])
                ]
            )
            bp.fit(X_b)
            res["budget_preprocessor"] = bp
            status["encoders"]["status"] = "LOADED"
            status["encoders"]["details"] = "ColumnTransformer fitted on budget dataset"
            status["scalers"]["status"] = "LOADED"
            status["scalers"]["details"] = "StandardScaler fitted on budget numeric features"
    except Exception as e:
        print(f"[MODEL FAIL] Loading budget models failed: {e}")

    # 3. Climate LSTM Model & Metadata
    try:
        meta_path = climate_dir / "best_climate_metadata.pkl"
        if meta_path.exists():
            res["climate_metadata"] = joblib.load(meta_path)

        clean_pt_path = climate_dir / "best_climate_lstm_model_clean.pt"
        if not clean_pt_path.exists():
            clean_pt_path = climate_dir / "fixed_climate_model.pt"

        status["climate_model"]["path"] = str(clean_pt_path)

        if clean_pt_path.exists():
            cm = ClimateLSTM(input_size=3, hidden_size=24, num_layers=1, output_size=3, dropout=0.2)
            state_dict = torch.load(clean_pt_path, map_location=torch.device("cpu"), weights_only=False)
            cm.load_state_dict(state_dict)
            cm.eval()
            res["climate_model"] = cm
            status["climate_model"]["status"] = "LOADED"
    except Exception as e:
        print(f"[MODEL FAIL] Loading climate model failed: {e}")

    res["verification_status"] = status
    return res


# Instantiate cached model artifacts once
_model_artifacts = load_all_ml_models()

visitors_model = _model_artifacts["visitors_model"]
onehot_visitor_enc = _model_artifacts["onehot_visitor_enc"]
ordinal_visitor_enc = _model_artifacts["ordinal_visitor_enc"]
ohe_category_festival = _model_artifacts["ohe_category_festival"]
budget_model = _model_artifacts["budget_model"]
budget_preprocessor = _model_artifacts["budget_preprocessor"]
budget_acc_tier_encoder = _model_artifacts["budget_acc_tier_encoder"]
budget_numeric_scaler = _model_artifacts["budget_numeric_scaler"]
budget_onehot_encoder = _model_artifacts["budget_onehot_encoder"]
climate_model = _model_artifacts["climate_model"]
climate_metadata = _model_artifacts["climate_metadata"]
verification_status = _model_artifacts.get("verification_status", verification_status)


@cache_data
def get_climate_df():
    if CLIMATE_CSV.exists():
        usecols = ["Tourist Spots", "District", "Date", "Temperature_Max_C", "Temperature_Min_C", "Rainfall_mm", "Season"]
        dtypes = {
            "Tourist Spots": "category",
            "District": "category",
            "Season": "category",
            "Temperature_Max_C": "float32",
            "Temperature_Min_C": "float32",
            "Rainfall_mm": "float32"
        }
        try:
            return pd.read_csv(CLIMATE_CSV, usecols=usecols, dtype=dtypes)
        except Exception:
            return pd.read_csv(CLIMATE_CSV)
    return None


def get_model_verification_status() -> dict:
    """
    Returns the REAL loading status of the existing models and preprocessing artifacts.
    Structure:
    {
        "climate_model": bool,
        "crowd_model": bool,
        "budget_model": bool,
        "encoders": bool,
        "scalers": bool
    }
    """
    return {
        "climate_model": climate_model is not None,
        "crowd_model": visitors_model is not None,
        "budget_model": budget_model is not None,
        "encoders": ohe_category_festival is not None and ordinal_visitor_enc is not None and budget_preprocessor is not None,
        "scalers": budget_preprocessor is not None
    }


# =====================================
# Helper: Determine Season from Date
# =====================================
def get_season_from_date(travel_date: datetime.date) -> str:
    month = travel_date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


# =====================================
# Climate Prediction Logic
# =====================================
def predict_climate(travel_date: datetime.date, district: str = None, spot_name: str = None) -> dict:
    """
    Predict climate parameters using dataset telemetry and PyTorch ClimateLSTM neural network.
    Logs exact input shape and prediction output.
    """
    if isinstance(travel_date, str):
        try:
            travel_date = datetime.datetime.strptime(travel_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date string format: {travel_date}")

    season = get_season_from_date(travel_date)
    month_name = travel_date.strftime("%B")

    df_climate = get_climate_df()
    if df_climate is None or df_climate.empty:
        raise FileNotFoundError(f"Climate dataset missing or unreadable at {CLIMATE_CSV}")

    subset = df_climate.copy()

    if spot_name and "Tourist Spots" in subset.columns:
        spot_sub = subset[subset["Tourist Spots"].astype(str).str.lower() == spot_name.lower()]
        if not spot_sub.empty:
            subset = spot_sub

    if district and "District" in subset.columns:
        dist_sub = subset[subset["District"].astype(str).str.lower() == district.lower()]
        if not dist_sub.empty:
            subset = dist_sub

    if "Season" in subset.columns:
        seas_sub = subset[subset["Season"].astype(str).str.lower() == season.lower()]
        if not seas_sub.empty:
            subset = seas_sub

    if subset.empty:
        subset = df_climate[df_climate["Season"].astype(str).str.lower() == season.lower()] if "Season" in df_climate.columns else df_climate

    if subset.empty:
        raise ValueError(f"No historical climate telemetry found for season '{season}' in Climate dataset.")

    temp_max = float(subset["Temperature_Max_C"].mean())
    temp_min = float(subset["Temperature_Min_C"].mean())
    rainfall = float(subset["Rainfall_mm"].mean())

    if climate_model is not None:
        try:
            with torch.no_grad():
                seq_input = torch.tensor([[
                    [temp_max, temp_min, rainfall]
                ] * 7], dtype=torch.float32)
                
                print(f"[PREDICT] Climate input shape: {tuple(seq_input.shape)}")
                
                lstm_out = climate_model(seq_input).numpy()[0]
                temp_max = float(temp_max + lstm_out[0])
                temp_min = float(temp_min + lstm_out[1])
                rainfall = float(max(0.0, rainfall + lstm_out[2]))
        except Exception as e:
            print(f"[WARNING] Error evaluating PyTorch climate model: {e}")

    temp_max = round(temp_max, 1)
    temp_min = round(temp_min, 1)
    rainfall = round(rainfall, 1)

    if rainfall > 50.0:
        weather_cond = "Humid with Heavy Rain Showers"
    elif rainfall > 10.0:
        weather_cond = "Partially Cloudy with Moderate Rain"
    elif temp_max > 36.0:
        weather_cond = "Hot & Sunny"
    elif temp_max < 22.0:
        weather_cond = "Cool & Crisp"
    else:
        weather_cond = "Pleasant & Clear"

    print(f"[RESULT] Climate prediction: Temp Max={temp_max}°C, Temp Min={temp_min}°C, Rainfall={rainfall}mm, Condition='{weather_cond}'")

    return {
        "travel_date": str(travel_date),
        "month": month_name,
        "season": season,
        "temperature_max": temp_max,
        "temperature_min": temp_min,
        "rainfall_mm": rainfall,
        "weather_condition": weather_cond
    }


@cache_data
def get_spot_climate_timeseries(spot_name: str):
    """
    Retrieves actual climate telemetry from Climate_Dataset_Final.csv for a selected spot
    and runs the trained PyTorch ClimateLSTM neural network over 7-day rolling windows
    to generate model predicted outputs for Max Temp, Min Temp, and Rainfall.
    """
    df_c = get_climate_df()
    if df_c is None or df_c.empty or "Tourist Spots" not in df_c.columns:
        return None

    spot_sub = df_c[df_c["Tourist Spots"].astype(str).str.lower() == str(spot_name).strip().lower()].copy()
    if spot_sub.empty:
        return None

    if "Date" not in spot_sub.columns:
        return None

    spot_sub["Date"] = pd.to_datetime(spot_sub["Date"], errors='coerce')
    spot_sub = spot_sub.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if len(spot_sub) == 0:
        return None

    if len(spot_sub) > 365:
        plot_df = spot_sub.tail(365).copy().reset_index(drop=True)
    else:
        plot_df = spot_sub.copy().reset_index(drop=True)

    vals = plot_df[["Temperature_Max_C", "Temperature_Min_C", "Rainfall_mm"]].values
    preds_max, preds_min, preds_rain = [], [], []

    if climate_model is not None:
        try:
            with torch.no_grad():
                for i in range(len(plot_df)):
                    if i < 7:
                        preds_max.append(float(vals[i, 0]))
                        preds_min.append(float(vals[i, 1]))
                        preds_rain.append(float(vals[i, 2]))
                    else:
                        window = torch.tensor(vals[i-7:i], dtype=torch.float32).unsqueeze(0)
                        out_delta = climate_model(window).numpy()[0]
                        base = vals[i-1]
                        preds_max.append(round(float(base[0] + out_delta[0]), 1))
                        preds_min.append(round(float(base[1] + out_delta[1]), 1))
                        preds_rain.append(round(float(max(0.0, base[2] + out_delta[2])), 1))
        except Exception as e:
            print(f"[WARNING] Error running PyTorch climate model on timeseries: {e}")
            preds_max = vals[:, 0].tolist()
            preds_min = vals[:, 1].tolist()
            preds_rain = vals[:, 2].tolist()
    else:
        preds_max = vals[:, 0].tolist()
        preds_min = vals[:, 1].tolist()
        preds_rain = vals[:, 2].tolist()

    plot_df["Pred_Max"] = preds_max
    plot_df["Pred_Min"] = preds_min
    plot_df["Pred_Rain"] = preds_rain

    return plot_df


# =====================================
# Tourist Visitors Prediction Logic
# =====================================
@cache_data(show_spinner=False)
def predict_visitors(spot_name: str, district: str, category: str, travel_date: datetime.date, season: str = None, festival: str = "None") -> int:
    """
    Predict visitor crowd count using fitted XGBoost model and Encoders.
    Logs input shape and prediction output.
    """
    if not visitors_model or not ohe_category_festival or not ordinal_visitor_enc:
        raise RuntimeError("Visitor crowd prediction ML model/encoders are missing or not loaded.")

    if hasattr(travel_date, "date") and callable(travel_date.date):
        travel_date = travel_date.date()
    elif isinstance(travel_date, str):
        travel_date = datetime.datetime.strptime(str(travel_date).split("T")[0], "%Y-%m-%d").date()
    elif not travel_date:
        travel_date = datetime.date.today()

    if not season:
        season = get_season_from_date(travel_date)

    month_name = travel_date.strftime("%B")
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

    month_idx = month_order.index(month_name) if month_name in month_order else 0
    season_idx = season_order.index(season) if season in season_order else 0

    try:
        cf_df = pd.DataFrame([[category, festival or "None"]], columns=["category", "festival"])
        cf_encoded = ohe_category_festival.transform(cf_df).toarray()

        ord_df = pd.DataFrame([[spot_name, district]], columns=["spot_name", "district"])
        try:
            ord_vals = ordinal_visitor_enc.transform(ord_df)
        except Exception:
            ord_vals = np.array([[0, 0]])

        ord_4 = np.array([[month_idx, season_idx, ord_vals[0, 0], ord_vals[0, 1]]])
        year_val = np.array([[travel_date.year]])

        full_features = np.hstack([cf_encoded, ord_4, year_val])

        # Feature shape verification
        expected_f = getattr(visitors_model, "n_features_in_", 22)
        if full_features.shape[1] != expected_f:
            print(f"[WARNING] Crowd feature dimension mismatch: got {full_features.shape[1]}, expected {expected_f}")

        print(f"[PREDICT] Crowd input shape: {full_features.shape}")

        prediction = visitors_model.predict(full_features)
        visitor_count = int(max(1, round(float(prediction[0]))))

        print(f"[RESULT] Crowd prediction: Predicted Visitors={visitor_count}")
        return visitor_count
    except Exception as err:
        raise RuntimeError(f"Failed to predict visitors for spot '{spot_name}': {err}")


# =====================================
# Budget Prediction Logic
# =====================================
@cache_data(show_spinner=False)
def predict_budget(transport_mode: str, accommodation_tier: str, duration_days: int, num_travelers: int, season: str = "Winter") -> dict:
    """
    Predict trip budget breakdown using fitted Multi-Output Regressor model.
    Verifies n_features_in_ == 13 and logs shapes and prediction outputs.
    Enforces single-process execution to prevent loky serialization errors in Streamlit.
    """
    if budget_model is None or budget_preprocessor is None:
        raise RuntimeError("Budget ML model or preprocessor is missing or not loaded.")

    try:
        # Normalize accommodation tier to match training dataset categories ('Budget', 'Mid', 'Premium')
        valid_tiers = {'budget': 'Budget', 'mid': 'Mid', 'premium': 'Premium', 'luxury': 'Premium'}
        clean_tier = valid_tiers.get(str(accommodation_tier).strip().lower(), 'Mid')

        # Ensure single-process execution for Streamlit safety
        if hasattr(budget_model, "n_jobs") and budget_model.n_jobs != 1:
            budget_model.n_jobs = 1
        if hasattr(budget_model, "estimators_"):
            for est in budget_model.estimators_:
                if hasattr(est, "n_jobs") and est.n_jobs != 1:
                    est.n_jobs = 1

        route_dist = float(duration_days * 35.0)
        input_df = pd.DataFrame([{
            'accommodation_tier': clean_tier,
            'transport_mode': str(transport_mode).strip().lower(),
            'season': str(season).strip(),
            'duration_days': int(duration_days),
            'num_travelers': int(num_travelers),
            'route_distance_km': route_dist
        }])

        encoded_input = budget_preprocessor.transform(input_df)

        # Feature shape verification (n_features_in_ == 13)
        expected_f = getattr(budget_model, "n_features_in_", 13)
        if encoded_input.shape[1] != expected_f:
            print(f"[WARNING] Budget feature dimension mismatch: got {encoded_input.shape[1]}, expected {expected_f}")

        print(f"[PREDICT] Budget input shape: {encoded_input.shape}")

        preds = budget_model.predict(encoded_input)[0]
        preds = [max(0.0, float(p)) for p in preds]
        total = float(sum(preds))

        breakdown = {
            "travel_cost": round(preds[0], 2),
            "stay_cost": round(preds[1], 2),
            "food_cost": round(preds[2], 2),
            "entry_fees": round(preds[3], 2),
            "tolls_and_parking": round(preds[4], 2)
        }

        print(f"[RESULT] Budget prediction: Total Budget=Rs.{total:,.2f}, Breakdown={breakdown}")

        return {
            "total_budget": round(total, 2),
            "breakdown": breakdown
        }
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"[ERROR] Budget prediction execution failed: {e}\n{err_detail}")
        raise RuntimeError(f"Failed to calculate trip budget prediction: {e}")


# =====================================
# Master Orchestration
# =====================================
def predict_all(user_input) -> dict:
    """
    Executes climate, visitor crowd, and budget predictions for UserInput.
    Logs inputs, shapes, and predictions to console.
    """
    spot_name = getattr(user_input, "spot_name", None)
    district = getattr(user_input, "destination_district", None)
    category = getattr(user_input, "category", None)
    travel_date = getattr(user_input, "travel_date", None)
    transport = getattr(user_input, "transport_mode", None)
    tier = getattr(user_input, "accommodation_tier", None)
    days = getattr(user_input, "number_of_days", 1)
    travelers = getattr(user_input, "number_of_travelers", 1)
    festival = getattr(user_input, "festival", "None") or "None"

    if not spot_name or not district or not category or not transport or not tier:
        raise ValueError("Missing required user inputs for ML prediction (spot_name, district, category, transport_mode, accommodation_tier).")

    if isinstance(travel_date, str):
        travel_date = datetime.datetime.strptime(travel_date, "%Y-%m-%d").date()
    elif not travel_date:
        travel_date = datetime.date.today()

    print(f"\n==================== ML PREDICTION PIPELINE START ====================")
    print(f"Target Spot: '{spot_name}' | District: '{district}' | Category: '{category}' | Date: {travel_date}")
    print(f"Transport: '{transport}' | Accommodation Tier: '{tier}' | Travelers: {travelers} | Days: {days}")
    print(f"----------------------------------------------------------------------")

    # 1. Climate Prediction
    climate_res = predict_climate(travel_date=travel_date, district=district, spot_name=spot_name)
    season = climate_res["season"]

    # 2. Visitors Crowd Prediction
    visitors_res = predict_visitors(
        spot_name=spot_name,
        district=district,
        category=category,
        travel_date=travel_date,
        season=season,
        festival=festival
    )

    # 3. Budget Prediction
    budget_res = predict_budget(
        transport_mode=transport,
        accommodation_tier=tier,
        duration_days=days,
        num_travelers=travelers,
        season=season
    )

    print(f"==================== ML PREDICTION PIPELINE END ======================\n")

    return {
        "predicted_climate": climate_res,
        "predicted_visitors": visitors_res,
        "estimated_budget": budget_res["total_budget"],
        "budget_breakdown": budget_res["breakdown"]
    }