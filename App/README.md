# Smart Travel AI Application (`App/`)

This directory contains the core full-stack AI application, ML model artifacts, FastAPI backend, and Streamlit frontend.

## Structure

```text
App/
├── __init__.py
│
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI REST API server & endpoints
│   ├── schema.py        # Pydantic input/output request schemas
│   ├── predict.py       # ML Pipeline (PyTorch Climate LSTM, Crowd XGBoost, Budget Regressor)
│   ├── recommendation.py# Spot & Amenity Recommendation Engine
│   ├── database.py      # Supabase & SQLite persistence layer
│   └── pdf_report.py    # ReportLab PDF summary generator
│
├── frontend/
│   ├── app.py           # Streamlit Web UI application
│   ├── styles.py        # Glassmorphic CSS styling system
│   └── pdf_generator.py # Legacy import shim for PDF generator
│
├── Pickles/
│   ├── Climate/         # PyTorch LSTM climate forecast models & metadata
│   ├── budget/          # Multi-Output budget regressor & encoders/scalers
│   └── spot vistiors/   # XGBoost crowd footfall predictor & encoders
│
├── requirements.txt     # Python package dependencies
└── README.md            # App component documentation
```

## Running the Application

### 1. Run Frontend (Streamlit)
```bash
streamlit run App/frontend/app.py
```

### 2. Run Backend (FastAPI)
```bash
uvicorn App.backend.main:app --reload --port 8000
```
