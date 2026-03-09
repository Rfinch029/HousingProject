from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
from Train import (
    train_model_artifacts,
    predict_from_artifacts,
    save_model_artifacts,
    load_model_artifacts,
)

app = Flask(__name__, template_folder="Templates")

DATA_FILE = "data_store.json"
MODEL_ARTIFACT_PATH = os.path.join("Output", "model", "predictor_artifacts.pkl")
LEGACY_MODEL_ARTIFACT_PATH = os.path.join("Preprocessing", "Output", "model", "predictor_artifacts.pkl")
PREDICTOR_CACHE = None

# Ensure the data file exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)


def save_data(data):
    """ Save user inputs to a JSON file """
    with open(DATA_FILE, "r") as f:
        existing_data = json.load(f)

    existing_data.append(data)

    with open(DATA_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)


def get_predictor():
    global PREDICTOR_CACHE
    if PREDICTOR_CACHE is None:
        if os.path.exists(MODEL_ARTIFACT_PATH):
            try:
                PREDICTOR_CACHE = load_model_artifacts(MODEL_ARTIFACT_PATH)
                print(f"Loaded model artifacts from {MODEL_ARTIFACT_PATH}")
            except Exception as exc:
                print(f"Failed to load saved artifacts ({exc}); retraining.")
                PREDICTOR_CACHE = train_model_artifacts(
                    num_epochs=100,
                    verbose=False,
                    plot_loss=False
                )
                save_model_artifacts(PREDICTOR_CACHE, MODEL_ARTIFACT_PATH)
                print(f"Saved model artifacts to {MODEL_ARTIFACT_PATH}")
        elif os.path.exists(LEGACY_MODEL_ARTIFACT_PATH):
            try:
                PREDICTOR_CACHE = load_model_artifacts(LEGACY_MODEL_ARTIFACT_PATH)
                print(f"Loaded legacy model artifacts from {LEGACY_MODEL_ARTIFACT_PATH}")
                save_model_artifacts(PREDICTOR_CACHE, MODEL_ARTIFACT_PATH)
                print(f"Migrated model artifacts to {MODEL_ARTIFACT_PATH}")
            except Exception as exc:
                print(f"Failed to load legacy artifacts ({exc}); retraining.")
                PREDICTOR_CACHE = train_model_artifacts(
                    num_epochs=100,
                    verbose=False,
                    plot_loss=False
                )
                save_model_artifacts(PREDICTOR_CACHE, MODEL_ARTIFACT_PATH)
                print(f"Saved model artifacts to {MODEL_ARTIFACT_PATH}")
        else:
            # Train once and persist for future app restarts.
            PREDICTOR_CACHE = train_model_artifacts(
                num_epochs=100,
                verbose=False,
                plot_loss=False
            )
            save_model_artifacts(PREDICTOR_CACHE, MODEL_ARTIFACT_PATH)
            print(f"Saved model artifacts to {MODEL_ARTIFACT_PATH}")
    return PREDICTOR_CACHE


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        if latitude and longitude:
            try:
                lat = float(latitude)
                lon = float(longitude)
            except ValueError:
                return render_template("index.html", error="Please enter valid numeric coordinates.")

            predictor = get_predictor()
            result = predict_from_artifacts(predictor, lat, lon)

            save_data({
                "latitude": lat,
                "longitude": lon,
                "predicted_price": result["predicted_price"],
                "predicted_residual": result["predicted_residual"],
                "closest_latitude": result["closest_latitude"],
                "closest_longitude": result["closest_longitude"],
                "last_known_price": result["last_known_price"],
                "last_month": result["last_month"]
            })
            return render_template("sucess.html", result=result)

    return render_template("index.html")


@app.route("/success")
def success():
    return render_template("sucess.html", result=None)


@app.route("/data")
def get_data():
    """ Endpoint to view stored data """
    with open(DATA_FILE, "r") as f:
        stored_data = json.load(f)
    return jsonify(stored_data)


if __name__ == "__main__":
    app.run(debug=True)
