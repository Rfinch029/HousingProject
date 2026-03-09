# HousingProject

Small housing-price forecasting project using:
- a custom NumPy LSTM (`Model.py`)
- per-house monthly features in CSV (`Preprocessing/Output/final/monthly_aggregated_data_test.csv`)
- a Flask UI (`app.py`) for latitude/longitude prediction

## Quick Start (Windows)

```powershell
cd C:\Users\seanr\PythonTutorial\HousingProject
.\venv\Scripts\python.exe -m pip install flask numpy pandas matplotlib requests requests-cache retry-requests openmeteo-requests
.\venv\Scripts\python.exe .\app.py
```

Open: `http://127.0.0.1:5000/`

## How Prediction Works

- The app trains model artifacts once (if no saved artifact exists).
- Artifacts are saved to:
  `Output/model/predictor_artifacts.pkl`
- Later app restarts load this file instead of retraining.
- Each form submit predicts next-month price from submitted lat/lon using the closest known house state + lag features.

## CLI Training (optional)

```powershell
.\venv\Scripts\python.exe .\Train.py
```

This prints training/validation metrics and allows interactive lat/lon predictions in terminal.

## Main Files

- `Train.py`: training, metrics, artifact save/load, prediction helpers
- `Model.py`: LSTM implementation (forward, step, backward)
- `app.py`: Flask routes and web inference
- `Templates/index.html`: input form
- `Templates/sucess.html`: prediction result page

## Notes

- To force retraining for Flask, delete:
  `Output/model/predictor_artifacts.pkl`
- `data_store.json` logs submitted coordinates and prediction metadata.
