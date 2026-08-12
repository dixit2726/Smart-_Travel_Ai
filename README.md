# Smart Travel AI 🧳🤖

An AI-powered Smart Tourism Recommendation & Planning Platform combining PyTorch deep learning, XGBoost footfall forecasting, multi-output trip budget estimation, and intelligent spot recommendations.

---

## 📁 Project Architecture

```text
ML_project/
│
├── App/
│   ├── __init__.py
│   │
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI REST API endpoints
│   │   ├── schema.py          # Input/output Pydantic schemas
│   │   ├── predict.py         # PyTorch, XGBoost & Scikit-Learn pipelines
│   │   ├── recommendation.py  # Spot & amenity recommendation engine
│   │   ├── database.py        # Supabase & SQLite persistence manager
│   │   └── pdf_report.py      # PDF trip summary report generator
│   │
│   ├── frontend/
│   │   ├── app.py             # Interactive Streamlit Web Application
│   │   ├── styles.py          # Custom CSS glassmorphism styling
│   │   └── pdf_generator.py   # Legacy import compatibility shim
│   │
│   ├── Pickles/               # Machine learning model artifacts & scalers
│   │   ├── Climate/           # PyTorch LSTM climate forecast models
│   │   ├── budget/            # Trip cost regressor & encoders
│   │   └── spot vistiors/     # Crowd footfall predictor & encoders
│   │
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # App documentation
│
├── Data/                      # CSV Datasets
│   ├── Climate_Dataset_Final.csv
│   ├── accommodations.csv
│   ├── nearby_amenities.csv
│   ├── spot_visitors.csv
│   ├── spots.csv
│   └── trip_budget_prediction_dataset.csv
│
├── Notebooks/                 # EDA & Training Jupyter Notebooks
│   ├── Climate_Forecast_EDA.ipynb
│   ├── Crowd_predication .ipynb
│   └── eda_trip_budget_prediction.ipynb
│
├── Others/                    # Application Assets & Supporting Files
│   ├── images/
│   │   ├── TravelLogo.webp
│   │   ├── travelll.webp
│   │   └── images_1.webp
│   ├── smart_tourism.db       # Primary SQLite Database
│   ├── deployment_notes.md    # Operating & deployment instructions
│   ├── package.json
│   └── package-lock.json
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `pip` package manager

### Installation

```bash
# Clone the repository and navigate to root directory
cd ML_project

# Install dependencies
pip install -r App/requirements.txt
```

---

## 🖥️ Running the Application

### 1. Launch Streamlit Web UI
```bash
streamlit run App/frontend/app.py
```

### 2. Launch FastAPI REST Server
```bash
uvicorn App.backend.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive Swagger API documentation.

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check & service status |
| `POST` | `/api/predict` | Runs full ML predictions (climate, crowd, budget) |
| `POST` | `/api/recommend` | Recommends spots based on user preferences |
| `POST` | `/api/smart-trip` | End-to-end trip planning & recommendation pipeline |
| `GET` | `/api/districts` | Retrieves list of supported destination districts |
| `GET` | `/api/spots` | Retrieves tourist spots filtered by district & category |
| `GET` | `/api/amenities` | Retrieves nearby hotels, restaurants, and hospitals |
| `GET` | `/api/trips` | Fetches saved trip history records |
| `POST` | `/api/save-trip` | Saves trip itinerary record to database |

---

## 📄 License & Notes
Designed for Smart Tourism AI planning. All datasets and model weights are preserved in `Data/` and `App/Pickles/`.
