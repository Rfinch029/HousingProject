# HousingProject

A Python project for exploring **housing price forecasting with weather and location data**, using a custom NumPy-based LSTM and a small Flask web interface for collecting latitude/longitude inputs.

## Overview

This repository combines three main pieces:

1. **Data preprocessing**
   - Loads housing price data from CSV files.
   - Builds full property addresses from Zillow-style fields.
   - Geocodes addresses with the Google Maps Geocoding API.
   - Pulls historical weather data from Open-Meteo.
   - Aggregates weather by month and merges it with price and coordinate data.

2. **Model training**
   - Implements a custom **LSTM from scratch in NumPy**.
   - Trains on the generated monthly dataset.
   - Produces next-month price predictions from a feature vector containing weather + coordinates.

3. **Web app**
   - A lightweight Flask app that collects latitude/longitude values from a form.
   - Saves submitted inputs to a local JSON file.

## Repository Structure

```text
HousingProject/
├── Main.py
├── Model.py
├── Train.py
├── Preprocessing.py
├── app.py
├── data_store.json
├── requirements.txt
├── test.py
├── Templates/
│   ├── index.html
│   └── sucess.html
└── Preprocessing/
    └── ...
```

## What Each File Does

### `Preprocessing.py` / `Main.py`
These scripts handle dataset preparation.

They:
- load a CSV of housing price data,
- create a `FullAddress` field from region/city/state,
- fetch coordinates for each address,
- request historical daily weather data,
- aggregate weather to monthly averages,
- reshape housing prices into a time-series format,
- save the final merged dataset for training.

### `Model.py`
Contains a handwritten **LSTM implementation** using NumPy, including:
- parameter initialization,
- forward pass,
- single-step inference,
- backward pass through time.

### `Train.py`
Loads the processed dataset and trains the custom LSTM.

It also:
- computes MSE loss,
- plots the training loss curve,
- prints final predictions on the training set,
- accepts latitude/longitude input from the terminal for next-month price prediction.

### `app.py`
Runs a Flask app that:
- serves a form at `/`,
- stores submitted latitude/longitude pairs in `data_store.json`,
- shows a success page,
- exposes saved entries at `/data`.

## Data Flow

```text
Raw housing CSV
   ↓
Address construction
   ↓
Geocoding (Google Maps API)
   ↓
Historical weather fetch (Open-Meteo)
   ↓
Monthly aggregation + merge with prices
   ↓
Training dataset CSV
   ↓
Custom LSTM training
   ↓
Price prediction
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rfinch029/HousingProject.git
cd HousingProject
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

The current `requirements.txt` only includes Flask, so you will likely need to install the project’s data and modeling dependencies manually as well.

```bash
pip install -r requirements.txt
pip install numpy pandas matplotlib requests requests-cache retry-requests openmeteo-requests
```

## Running the Project

### Run the Flask app

```bash
python app.py
```

Then open the local URL shown in your terminal.

### Run preprocessing

```bash
python Preprocessing.py
```

or

```bash
python Main.py
```

Use this step to generate the processed monthly dataset before training.

### Train the model

```bash
python Train.py
```

This will:
- load the processed CSV,
- train the LSTM,
- show a loss curve,
- prompt for latitude and longitude to generate a prediction.

## Expected Inputs and Outputs

### Inputs
- A housing dataset CSV under the preprocessing directory.
- Address fields such as region, city, and state.
- Historical date range for weather collection.
- Latitude and longitude for interactive prediction.

### Outputs
- Geocoded address CSV
- Weather data CSV
- Monthly aggregated final dataset CSV
- Stored form submissions in `data_store.json`
- Terminal prediction output from the trained model

## Notes and Current Limitations

This repository appears to be an in-progress academic or experimental project, so a few parts are still rough around the edges:

- `requirements.txt` is incomplete for the full pipeline.
- API usage is currently hard-coded in the preprocessing scripts.
- The Flask app stores coordinates, but it does **not** currently call the trained model.
- The success template filename is spelled `sucess.html`, and the route depends on that exact filename.
- Several paths are hard-coded for local preprocessing files.
- The training script expects the processed CSV to already exist.

## Suggested Improvements

- Move API keys into environment variables.
- Expand `requirements.txt` to include all dependencies.
- Connect the Flask app to the trained model so predictions can be made through the web UI.
- Add data validation and error handling for missing files and API failures.
- Save trained model parameters for reuse.
- Add example datasets and screenshots.
- Rename `sucess.html` to `success.html` for consistency.

## Tech Stack

- **Python**
- **Flask**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Requests / requests-cache**
- **Open-Meteo API**
- **Google Maps Geocoding API**

## Project Status

Experimental / work in progress.

This project is a good foundation for a fuller end-to-end housing forecasting application, but it currently reads as a prototype rather than a production-ready system.

---

## Source Summary

This README is based on the current repository contents, including:
- preprocessing scripts for address, weather, and monthly aggregation,
- a custom NumPy LSTM implementation,
- a training script for next-month prediction,
- a Flask app with HTML templates and JSON storage.
