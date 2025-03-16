from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os

app = Flask(__name__)

DATA_FILE = "data_store.json"

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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        address = request.form.get("address")
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")

        if address and temperature and humidity:
            user_data = {
                "address": address,
                "temperature": temperature,
                "humidity": humidity
            }
            save_data(user_data)
            return redirect(url_for("success"))

    return render_template("index.html")


@app.route("/success")
def success():
    return render_template("success.html")


@app.route("/data")
def get_data():
    """ Endpoint to view stored data """
    with open(DATA_FILE, "r") as f:
        stored_data = json.load(f)
    return jsonify(stored_data)


if __name__ == "__main__":
    app.run(debug=True)
