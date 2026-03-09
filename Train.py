import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from Model import LSTM


def load_csv_dataset(file_path):
    """
    Loads, cleans, and sorts the dataset for per-house time-series training.
    Also creates lag-price features per house.
    """
    df = pd.read_csv(file_path)

    required_columns = {"month_year", "Latitude", "Longitude", "price"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required column(s): {missing_str}")

    df["month_year"] = pd.to_datetime(df["month_year"].astype(str), format="%Y-%m", errors="coerce")

    numeric_columns = [col for col in df.columns if col != "month_year"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["month_year", "Latitude", "Longitude", "price"]).copy()
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    # Use rounded coordinates as house identity.
    df["house_id"] = df["Latitude"].round(6).astype(str) + "|" + df["Longitude"].round(6).astype(str)
    df = df.sort_values(["house_id", "month_year"]).reset_index(drop=True)

    # Lag features are often the strongest signal for short-horizon housing forecasts.
    df["price_lag_1"] = df.groupby("house_id")["price"].shift(1)
    df["price_lag_3"] = df.groupby("house_id")["price"].shift(3)
    df["price_lag_12"] = df.groupby("house_id")["price"].shift(12)

    # Require lag-1; backfill larger lags from shorter lag so we keep early rows.
    df = df.dropna(subset=["price_lag_1"]).copy()
    df["price_lag_3"] = df["price_lag_3"].fillna(df["price_lag_1"])
    df["price_lag_12"] = df["price_lag_12"].fillna(df["price_lag_3"])

    # Residual target: predict change from previous month.
    df["target_residual"] = df["price"] - df["price_lag_1"]

    feature_columns = [col for col in df.columns if col not in {"month_year", "price", "house_id", "target_residual"}]
    return df, feature_columns


def fit_standard_scaler(values):
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standard_scale(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def standard_inverse(values, mean, std):
    return (values * std + mean).astype(np.float32)


def transform_target(values, use_log_target):
    if use_log_target:
        return np.log1p(values).astype(np.float32)
    return values.astype(np.float32)


def inverse_transform_target(values, use_log_target):
    if use_log_target:
        return np.expm1(values).astype(np.float32)
    return values.astype(np.float32)


def compute_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def clip_gradients(gradients, clip_value):
    """Elementwise gradient clipping."""
    return np.clip(gradients, -clip_value, clip_value)


def compute_regression_metrics(targets, predictions):
    y_true = targets.reshape(-1)
    y_pred = predictions.reshape(-1)

    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
            "r2": float("nan"),
            "within_10pct": float("nan"),
        }

    errors = y_pred - y_true
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    non_zero_mask = np.abs(y_true) > 1e-8
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(errors[non_zero_mask]) / np.abs(y_true[non_zero_mask])) * 100
    else:
        mape = float("nan")

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    within_10pct = np.mean(np.abs(errors) <= (0.10 * np.abs(y_true))) * 100

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "within_10pct": float(within_10pct),
    }


def print_regression_metrics(title, metrics):
    print(f"\n--- {title} ---")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R^2: {metrics['r2']:.4f}")
    print(f"Within 10%: {metrics['within_10pct']:.2f}%")


def predict_next_month(model, w_out, b_out, h_last, c_last, x_next):
    h_next, c_next = model.step_forward(h_last, c_last, x_next)
    prediction = np.dot(h_next.squeeze(), w_out) + b_out
    return prediction, h_next, c_next


def build_house_sequences(
    df,
    feature_columns,
    target_column,
    x_mean,
    x_std,
    y_mean,
    y_std,
    use_log_target,
    train_last_month,
):
    sequences = []
    grouped = df.groupby("house_id", sort=False)

    for house_id, group in grouped:
        group = group.sort_values("month_year")
        if len(group) < 1:
            continue

        x_raw = group[feature_columns].values.astype(np.float32)
        price_raw = group[["price"]].values.astype(np.float32)
        lag1_raw = group[["price_lag_1"]].values.astype(np.float32)
        y_raw = group[[target_column]].values.astype(np.float32)
        months = group["month_year"].to_numpy()

        x_scaled = standard_scale(x_raw, x_mean, x_std)
        y_transformed = transform_target(y_raw, use_log_target)
        y_scaled = standard_scale(y_transformed, y_mean, y_std)

        train_row_count = int(np.sum(months <= train_last_month))
        last_feature_values = group[feature_columns].iloc[-1].to_dict()

        sequences.append(
            {
                "house_id": house_id,
                "latitude": float(group["Latitude"].iloc[0]),
                "longitude": float(group["Longitude"].iloc[0]),
                "months": months,
                "x_scaled": x_scaled,
                "y_scaled": y_scaled,
                "target_residual_raw": y_raw,
                "price_raw": price_raw,
                "lag1_raw": lag1_raw,
                "train_row_count": train_row_count,
                "last_feature_values": last_feature_values,
            }
        )

    return sequences


def evaluate_metrics_and_baseline(model, w_out, b_out, sequences, train_last_month, y_mean, y_std, use_log_target):
    hidden_dim = w_out.shape[0]

    model_train_pred_scaled, model_train_true_price = [], []
    model_val_pred_scaled, model_val_true_price = [], []
    model_train_lag1, model_val_lag1 = [], []

    baseline_train_pred_raw, baseline_train_true_price = [], []
    baseline_val_pred_raw, baseline_val_true_price = [], []

    for seq in sequences:
        x_seq = seq["x_scaled"]
        price_raw = seq["price_raw"].reshape(-1)
        lag1_raw = seq["lag1_raw"].reshape(-1)
        months = seq["months"]

        h = np.zeros((hidden_dim, 1), dtype=np.float32)
        c = np.zeros((hidden_dim, 1), dtype=np.float32)

        for t in range(len(x_seq)):
            x_t = x_seq[t].reshape(-1, 1)
            h, c = model.step_forward(h, c, x_t)

            pred_scaled = float((h.T @ w_out + b_out).squeeze())
            target_true_price = float(price_raw[t])
            lag1_value = float(lag1_raw[t])
            target_month = months[t]

            if target_month <= train_last_month:
                model_train_pred_scaled.append(pred_scaled)
                model_train_true_price.append(target_true_price)
                model_train_lag1.append(lag1_value)
            else:
                model_val_pred_scaled.append(pred_scaled)
                model_val_true_price.append(target_true_price)
                model_val_lag1.append(lag1_value)

            # Naive baseline: predict current month as previous month.
            baseline_pred = lag1_value
            if target_month <= train_last_month:
                baseline_train_pred_raw.append(baseline_pred)
                baseline_train_true_price.append(target_true_price)
            else:
                baseline_val_pred_raw.append(baseline_pred)
                baseline_val_true_price.append(target_true_price)

    model_train_pred_scaled = np.array(model_train_pred_scaled, dtype=np.float32).reshape(-1, 1)
    model_val_pred_scaled = np.array(model_val_pred_scaled, dtype=np.float32).reshape(-1, 1)
    model_train_true_price = np.array(model_train_true_price, dtype=np.float32).reshape(-1, 1)
    model_val_true_price = np.array(model_val_true_price, dtype=np.float32).reshape(-1, 1)
    model_train_lag1 = np.array(model_train_lag1, dtype=np.float32).reshape(-1, 1)
    model_val_lag1 = np.array(model_val_lag1, dtype=np.float32).reshape(-1, 1)

    # Model predicts residual; convert back to dollar-space residual then add lag-1 to recover price.
    model_train_pred_transformed = standard_inverse(model_train_pred_scaled, y_mean, y_std)
    model_val_pred_transformed = standard_inverse(model_val_pred_scaled, y_mean, y_std)
    model_train_pred_residual = inverse_transform_target(model_train_pred_transformed, use_log_target)
    model_val_pred_residual = inverse_transform_target(model_val_pred_transformed, use_log_target)
    model_train_pred_price = model_train_lag1 + model_train_pred_residual
    model_val_pred_price = model_val_lag1 + model_val_pred_residual

    model_train_metrics = compute_regression_metrics(model_train_true_price, model_train_pred_price)
    model_val_metrics = compute_regression_metrics(model_val_true_price, model_val_pred_price)

    baseline_train_pred_raw = np.array(baseline_train_pred_raw, dtype=np.float32).reshape(-1, 1)
    baseline_train_true_price = np.array(baseline_train_true_price, dtype=np.float32).reshape(-1, 1)
    baseline_val_pred_raw = np.array(baseline_val_pred_raw, dtype=np.float32).reshape(-1, 1)
    baseline_val_true_price = np.array(baseline_val_true_price, dtype=np.float32).reshape(-1, 1)

    baseline_train_metrics = compute_regression_metrics(baseline_train_true_price, baseline_train_pred_raw)
    baseline_val_metrics = compute_regression_metrics(baseline_val_true_price, baseline_val_pred_raw)

    return model_train_metrics, model_val_metrics, baseline_train_metrics, baseline_val_metrics


def build_house_state_map(model, sequences):
    states = {}
    for seq in sequences:
        h_seq = model.forward(seq["x_scaled"])
        h_last = h_seq[-1].copy()
        _, c_last, _, _, _, _, _ = model.cache[-1]

        price_history = seq["price_raw"].reshape(-1)
        lag_1 = float(price_history[-1])
        lag_3 = float(price_history[-3]) if len(price_history) >= 3 else lag_1
        lag_12 = float(price_history[-12]) if len(price_history) >= 12 else lag_3

        states[seq["house_id"]] = {
            "h_last": h_last,
            "c_last": c_last.copy(),
            "latitude": seq["latitude"],
            "longitude": seq["longitude"],
            "last_price": float(price_history[-1]),
            "last_month": pd.Timestamp(seq["months"][-1]).strftime("%Y-%m"),
            "price_lag_1": lag_1,
            "price_lag_3": lag_3,
            "price_lag_12": lag_12,
            "last_feature_values": seq["last_feature_values"],
        }
    return states


def find_closest_house_state(state_map, lat, lon):
    best_id = None
    best_dist = None

    for house_id, state in state_map.items():
        d2 = (state["latitude"] - lat) ** 2 + (state["longitude"] - lon) ** 2
        if best_dist is None or d2 < best_dist:
            best_dist = d2
            best_id = house_id

    return best_id, state_map[best_id]


def build_inference_feature_row(feature_columns, lat, lon, closest_state, weather_defaults):
    row = []
    for feature in feature_columns:
        if feature == "Latitude":
            value = lat
        elif feature == "Longitude":
            value = lon
        elif feature == "price_lag_1":
            value = closest_state["price_lag_1"]
        elif feature == "price_lag_3":
            value = closest_state["price_lag_3"]
        elif feature == "price_lag_12":
            value = closest_state["price_lag_12"]
        elif feature in weather_defaults:
            value = weather_defaults[feature]
        else:
            value = closest_state["last_feature_values"].get(feature, 0.0)
        row.append(float(value))

    return np.array(row, dtype=np.float32).reshape(1, -1)


def get_weather_defaults():
    return {
        "avg_temperature": 10.0,
        "avg_precipitation": 50.0,
        "avg_rain": 0.0,
        "avg_snowfall": 0.0,
        "avg_wind_speed": 5.0,
        "avg_wind_gusts": 8.0,
        "temperature_2m_mean": 10.0,
        "precipitation_sum": 50.0,
        "rain_sum": 0.0,
        "snowfall_sum": 0.0,
        "wind_speed_10m_max": 5.0,
        "wind_gusts_10m_max": 8.0,
    }


def train_model_artifacts(
    dataset_path="Preprocessing/Output/final/monthly_aggregated_data_test.csv",
    train_fraction=0.8,
    hidden_dim=16,
    num_epochs=100,
    learning_rate=0.001,
    gradient_clip_value=1.0,
    verbose=True,
    plot_loss=False,
):
    # Residuals can be negative, so keep target linear and standardized.
    use_log_target = False

    df, feature_columns = load_csv_dataset(dataset_path)
    unique_months = np.array(sorted(df["month_year"].unique()))
    if len(unique_months) < 2:
        raise ValueError("Need at least 2 unique months for train/validation split.")

    split_idx = max(1, int(train_fraction * len(unique_months)))
    if split_idx >= len(unique_months):
        split_idx = len(unique_months) - 1
    train_last_month = unique_months[split_idx - 1]

    train_mask = df["month_year"] <= train_last_month
    val_mask = df["month_year"] > train_last_month

    x_train_raw_for_scaler = df.loc[train_mask, feature_columns].values.astype(np.float32)
    y_train_raw_for_scaler = df.loc[train_mask, ["target_residual"]].values.astype(np.float32)

    x_mean, x_std = fit_standard_scaler(x_train_raw_for_scaler)
    y_train_transformed_for_scaler = transform_target(y_train_raw_for_scaler, use_log_target)
    y_mean, y_std = fit_standard_scaler(y_train_transformed_for_scaler)

    sequences = build_house_sequences(
        df=df,
        feature_columns=feature_columns,
        target_column="target_residual",
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        use_log_target=use_log_target,
        train_last_month=train_last_month,
    )
    if not sequences:
        raise ValueError("No valid house sequences found for training.")

    input_dim = len(feature_columns)
    model = LSTM(input_dim, hidden_dim)
    w_out = np.random.randn(hidden_dim, 1) * np.sqrt(1 / hidden_dim)
    b_out = np.zeros((1, 1))

    loss_history = []
    train_rows = int(train_mask.sum())
    val_rows = int(val_mask.sum())

    if verbose:
        print(f"Total rows: {len(df)} | Train: {train_rows} | Validation: {val_rows}")
        print(f"Total houses: {len(sequences)}")
        print(f"Num features: {input_dim}")
        print(f"Train months end at: {pd.Timestamp(train_last_month).strftime('%Y-%m')}")
        print("Target: residual (price - price_lag_1)")
        print(f"Target transform: {'log1p + standardize' if use_log_target else 'standardize only'}")

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_row_count = 0

        for seq_idx in np.random.permutation(len(sequences)):
            seq = sequences[seq_idx]
            row_count = seq["train_row_count"]
            if row_count < 1:
                continue

            x_seq = seq["x_scaled"][:row_count]
            y_seq = seq["y_scaled"][:row_count]

            h_seq = model.forward(x_seq)
            h_seq_squeezed = h_seq.squeeze(axis=2)
            predictions = np.dot(h_seq_squeezed, w_out) + b_out

            loss = compute_loss(predictions, y_seq)
            epoch_loss_sum += float(loss) * row_count
            epoch_row_count += row_count

            dpred = 2 * (predictions - y_seq) / row_count
            d_w_out = h_seq_squeezed.T @ dpred
            d_b_out = np.sum(dpred, axis=0, keepdims=True)
            dh_seq_grad = (dpred @ w_out.T).reshape(row_count, hidden_dim, 1)

            lstm_grads = model.backward(dh_seq_grad)

            d_w_out = clip_gradients(d_w_out, gradient_clip_value)
            d_b_out = clip_gradients(d_b_out, gradient_clip_value)
            for key in lstm_grads:
                lstm_grads[key] = clip_gradients(lstm_grads[key], gradient_clip_value)

            w_out -= learning_rate * d_w_out
            b_out -= learning_rate * d_b_out

            model.W_f -= learning_rate * lstm_grads["W_f"]
            model.b_f -= learning_rate * lstm_grads["b_f"]
            model.W_i -= learning_rate * lstm_grads["W_i"]
            model.b_i -= learning_rate * lstm_grads["b_i"]
            model.W_c -= learning_rate * lstm_grads["W_c"]
            model.b_c -= learning_rate * lstm_grads["b_c"]
            model.W_o -= learning_rate * lstm_grads["W_o"]
            model.b_o -= learning_rate * lstm_grads["b_o"]

        epoch_loss = epoch_loss_sum / max(1, epoch_row_count)
        loss_history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

    (
        model_train_metrics,
        model_val_metrics,
        baseline_train_metrics,
        baseline_val_metrics,
    ) = evaluate_metrics_and_baseline(
        model=model,
        w_out=w_out,
        b_out=b_out,
        sequences=sequences,
        train_last_month=train_last_month,
        y_mean=y_mean,
        y_std=y_std,
        use_log_target=use_log_target,
    )

    if verbose:
        print_regression_metrics("Training Metrics (Model, per-house)", model_train_metrics)
        print_regression_metrics("Validation Metrics (Model, per-house)", model_val_metrics)
        print_regression_metrics("Training Metrics (Naive Baseline: prev-month price)", baseline_train_metrics)
        print_regression_metrics("Validation Metrics (Naive Baseline: prev-month price)", baseline_val_metrics)
        if baseline_val_metrics["mae"] > 0:
            mae_ratio = model_val_metrics["mae"] / baseline_val_metrics["mae"]
            print(f"\nValidation MAE ratio (model / baseline): {mae_ratio:.2f}x")

    if plot_loss:
        plt.figure()
        plt.plot(range(1, num_epochs + 1), loss_history, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.show()

    weather_defaults = get_weather_defaults()
    house_state_map = build_house_state_map(model, sequences)

    return {
        "model": model,
        "w_out": w_out,
        "b_out": b_out,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_columns": feature_columns,
        "use_log_target": use_log_target,
        "house_state_map": house_state_map,
        "weather_defaults": weather_defaults,
        "loss_history": loss_history,
        "metrics": {
            "model_train": model_train_metrics,
            "model_val": model_val_metrics,
            "baseline_train": baseline_train_metrics,
            "baseline_val": baseline_val_metrics,
        },
    }


def predict_from_artifacts(artifacts, lat, lon):
    _, closest_state = find_closest_house_state(artifacts["house_state_map"], lat, lon)

    new_input_raw = build_inference_feature_row(
        feature_columns=artifacts["feature_columns"],
        lat=lat,
        lon=lon,
        closest_state=closest_state,
        weather_defaults=artifacts["weather_defaults"],
    )
    new_input_scaled = standard_scale(new_input_raw, artifacts["x_mean"], artifacts["x_std"]).reshape(-1, 1)

    prediction_scaled, _, _ = predict_next_month(
        artifacts["model"],
        artifacts["w_out"],
        artifacts["b_out"],
        closest_state["h_last"].copy(),
        closest_state["c_last"].copy(),
        new_input_scaled,
    )

    prediction_transformed = standard_inverse(
        np.array(prediction_scaled, dtype=np.float32).reshape(1, 1),
        artifacts["y_mean"],
        artifacts["y_std"],
    )
    prediction_residual = float(
        inverse_transform_target(prediction_transformed, artifacts["use_log_target"]).squeeze()
    )
    prediction_value = closest_state["price_lag_1"] + prediction_residual

    return {
        "input_latitude": float(lat),
        "input_longitude": float(lon),
        "closest_latitude": float(closest_state["latitude"]),
        "closest_longitude": float(closest_state["longitude"]),
        "last_known_price": float(closest_state["last_price"]),
        "last_month": closest_state["last_month"],
        "lag_1": float(closest_state["price_lag_1"]),
        "lag_3": float(closest_state["price_lag_3"]),
        "lag_12": float(closest_state["price_lag_12"]),
        "predicted_residual": float(prediction_residual),
        "predicted_price": float(prediction_value),
    }


def save_model_artifacts(artifacts, output_path):
    """Saves trained artifacts to disk for reuse by the Flask app."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_artifacts(output_path):
    """Loads previously saved artifacts."""
    with open(output_path, "rb") as f:
        return pickle.load(f)


def train():
    artifacts = train_model_artifacts(verbose=True, plot_loss=True)

    print("\n--- Next Month Prediction ---")
    print("Enter the house coordinates (latitude, longitude) to get the next month's predicted price.")
    print("Type 'q' at any prompt to quit.")

    while True:
        lat_str = input("Enter the latitude of the house: ").strip()
        if lat_str.lower() == "q":
            break

        lon_str = input("Enter the longitude of the house: ").strip()
        if lon_str.lower() == "q":
            break

        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            print("Invalid input. Please enter numeric values for latitude and longitude.")
            continue

        result = predict_from_artifacts(artifacts, lat, lon)
        print(
            f"\nUsing closest historical house at "
            f"({result['closest_latitude']:.6f}, {result['closest_longitude']:.6f}) "
            f"with last known price ${result['last_known_price']:,.2f} "
            f"in {result['last_month']}."
        )
        print(
            f"Using lag features: "
            f"lag_1=${result['lag_1']:,.2f}, "
            f"lag_3=${result['lag_3']:,.2f}, "
            f"lag_12=${result['lag_12']:,.2f}"
        )
        print(f"Predicted residual change: ${result['predicted_residual']:,.2f}")
        print(
            f"Predicted Price for Next Month at "
            f"({result['input_latitude']}, {result['input_longitude']}): "
            f"${result['predicted_price']:,.2f}\n"
        )


if __name__ == "__main__":
    train()
