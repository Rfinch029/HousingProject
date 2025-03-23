import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import LSTM  # Ensure your LSTM class defines forward(), step_forward(), and backward()


def load_csv_dataset(file_path):
    """
    Loads the CSV dataset, drops non-numeric columns (like 'month_year'),
    and separates input features from the target ('price').
    """
    df = pd.read_csv(file_path)
    # Drop non-numeric column 'month_year' if present.
    if 'month_year' in df.columns:
        df = df.drop(columns=['month_year'])
    if 'price' not in df.columns:
        raise ValueError("Dataset must contain a 'price' column as the target.")

    # Use all columns except 'price' as input features.
    input_columns = [col for col in df.columns if col != 'price']
    x = df[input_columns].values.astype(np.float32)
    target = df[['price']].values.astype(np.float32)
    return x, target


def compute_loss(predictions, targets):
    """Computes Mean Squared Error (MSE) loss."""
    return np.mean((predictions - targets) ** 2)


def predict_next_month(model, W_out, b_out, h_last, c_last, x_next):
    """
    Predicts the next month's price given the last hidden/cell states and new input features.

    Args:
        model (LSTM): The trained LSTM instance.
        W_out (np.ndarray): Output layer weight matrix of shape (hidden_dim, 1).
        b_out (np.ndarray): Output layer bias of shape (1, 1).
        h_last (np.ndarray): Final hidden state from the training sequence (shape: (hidden_dim, 1)).
        c_last (np.ndarray): Final cell state from the training sequence (shape: (hidden_dim, 1)).
        x_next (np.ndarray): New input features for the next month (shape: (input_dim, 1)).

    Returns:
        prediction (float): Predicted price for the next month.
        h_next, c_next: Updated hidden and cell states (not used for subsequent queries here).
    """
    h_next, c_next = model.step_forward(h_last, c_last, x_next)
    prediction = np.dot(h_next.squeeze(), W_out) + b_out
    return prediction, h_next, c_next


def train():
    # Set the dataset path.
    dataset_path = "Preprocessing/Output/final/monthly_aggregated_data_test.csv"
    x, target = load_csv_dataset(dataset_path)

    T = x.shape[0]  # Number of training samples (time steps)
    input_dim = x.shape[1]  # Number of features (e.g., 8)
    hidden_dim = 5  # Chosen hidden dimension

    # Initialize the LSTM model.
    model = LSTM(input_dim, hidden_dim)

    # Initialize output layer parameters (mapping LSTM hidden state to a scalar).
    W_out = np.random.randn(hidden_dim, 1) * np.sqrt(1 / hidden_dim)
    b_out = np.zeros((1, 1))

    num_epochs = 100
    learning_rate = 0.003
    loss_history = []

    # Training loop.
    for epoch in range(num_epochs):
        # Full sequence forward pass.
        h_seq = model.forward(x)  # Expected shape: (T, hidden_dim, 1)
        h_seq_squeezed = h_seq.squeeze(axis=2)  # Now shape: (T, hidden_dim)
        predictions = np.dot(h_seq_squeezed, W_out) + b_out  # Shape: (T, 1)

        loss = compute_loss(predictions, target)
        loss_history.append(loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        # Compute gradients for the output layer.
        dpred = 2 * (predictions - target) / T  # (T, 1)
        dW_out = np.zeros_like(W_out)
        db_out = np.zeros_like(b_out)
        dh_seq_grad = np.zeros_like(h_seq_squeezed)

        for t in range(T):
            dW_out += np.dot(h_seq_squeezed[t].reshape(-1, 1), dpred[t].reshape(1, -1))
            db_out += dpred[t].reshape(b_out.shape)
            dh_seq_grad[t] = np.dot(W_out, dpred[t])

        # Reshape gradient to match LSTM's expected shape: (T, hidden_dim, 1)
        dh_seq_grad = dh_seq_grad.reshape(T, hidden_dim, 1)

        # Backpropagate through the LSTM.
        lstm_grads = model.backward(dh_seq_grad)

        # Update output layer parameters.
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out

        # Update LSTM parameters.
        model.W_f -= learning_rate * lstm_grads['W_f']
        model.b_f -= learning_rate * lstm_grads['b_f']
        model.W_i -= learning_rate * lstm_grads['W_i']
        model.b_i -= learning_rate * lstm_grads['b_i']
        model.W_c -= learning_rate * lstm_grads['W_c']
        model.b_c -= learning_rate * lstm_grads['b_c']
        model.W_o -= learning_rate * lstm_grads['W_o']
        model.b_o -= learning_rate * lstm_grads['b_o']

    # Plot the training loss curve.
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # Final predictions on training data.
    h_seq_final = model.forward(x).squeeze(axis=2)
    predictions_final = np.dot(h_seq_final, W_out) + b_out
    print("\nFinal Predictions on Training Data:")
    print(predictions_final)
    print("\nGround Truth:")
    print(target)

    # ---------------------------
    # Next Month Prediction with User Input (Repeated)
    # ---------------------------
    # Get the final hidden and cell states from the training sequence.
    h_seq_all = model.forward(x)
    h_last = h_seq_all[-1]  # (hidden_dim, 1)
    # Extract final cell state from the last cached tuple.
    _, c_last, _, _, _, _, _ = model.cache[-1]

    # Save the original final states to use for each independent prediction.
    orig_h_last = h_last.copy()
    orig_c_last = c_last.copy()

    # Default weather features (in the same order as your training features).
    default_weather = [10.0, 50.0, 0.0, 0.0, 5.0, 8.0]

    print("\n--- Next Month Prediction ---")
    print("Enter the house coordinates (latitude, longitude) to get the next month's predicted price.")
    print("Type 'q' at any prompt to quit.")

    while True:
        lat_str = input("Enter the latitude of the house: ").strip()
        if lat_str.lower() == 'q':
            break
        lon_str = input("Enter the longitude of the house: ").strip()
        if lon_str.lower() == 'q':
            break
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            print("Invalid input. Please enter numeric values for latitude and longitude.")
            continue

        # Build the new input using default weather features plus the user-provided coordinates.
        new_input = np.array(default_weather + [lat, lon], dtype=np.float32)
        new_input = new_input.reshape(-1, 1)  # shape: (input_dim, 1)

        # Predict using the original final states (to keep each prediction independent).
        prediction_next, _, _ = predict_next_month(model, W_out, b_out, orig_h_last, orig_c_last, new_input)
        print(f"\nPredicted Price for Next Month for the house at (lat, lon) = ({lat}, {lon}):")
        print(prediction_next)
        print()  # Blank line for spacing


if __name__ == "__main__":
    train()
