import numpy as np
from tensorflow.python.ops.gen_experimental_dataset_ops import load_dataset
import pandas as pd
from Model import LSTM

import pandas as pd
import numpy as np


def load_csv_dataset(file_path):
    # Load the CSV file into a pandas DataFrame.
    df = pd.read_csv(file_path)

    # Print the columns available in the CSV for reference.
    print("Columns in dataset:", df.columns.tolist())

    # Drop non-numeric columns (e.g., 'month_year') if present.
    if 'month_year' in df.columns:
        df = df.drop(columns=['month_year'])

    # Separate the target column ('price') and use all other numeric columns as features.
    if 'price' not in df.columns:
        raise ValueError("The dataset must have a 'price' column as the target.")

    input_columns = [col for col in df.columns if col != 'price']
    target_columns = ['price']

    # Convert to numpy arrays and ensure they are of numeric type.
    x = df[input_columns].values.astype(np.float32)
    target = df[target_columns].values.astype(np.float32)

    return x, target


def compute_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def train():
    # Path to your dataset file.
    dataset_path = 'Preprocessing/Output/final/monthly_aggregated_data_test.csv'

    # Load dataset.
    x, target = load_csv_dataset(dataset_path)

    # Determine dimensions from dataset.
    input_dim = x.shape[1]  # Number of input features (should be 8 based on your columns)
    hidden_dim = 5  # Number of hidden neurons in the LSTM

    # Create the LSTM model instance.
    model = LSTM(input_dim, hidden_dim)

    # Initialize an output layer (weights and bias) to map hidden state (of dimension hidden_dim) to a scalar.
    W_out = np.random.randn(hidden_dim, 1) * np.sqrt(1 / hidden_dim)
    b_out = np.zeros((1, 1))

    num_epochs = 10
    learning_rate = 0.01  # Note: Learning rate is not used since weight updates/backprop are not implemented.

    for epoch in range(num_epochs):
        # Forward pass: get hidden state sequence from the LSTM.
        h_seq = model.forward(x)  # h_seq shape: (T, hidden_dim, 1)

        # Remove the last singleton dimension to get shape: (T, hidden_dim)
        h_seq = h_seq.squeeze(axis=2)

        # Apply the output layer: for each time step, map hidden state to scalar prediction.
        predictions = np.dot(h_seq, W_out) + b_out  # predictions shape: (T, 1)

        # Compute and print the loss.
        loss = compute_loss(predictions, target)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # After training, print final predictions and ground truth.
    h_seq_final = model.forward(x).squeeze(axis=2)
    predictions_final = np.dot(h_seq_final, W_out) + b_out

    print("\nFinal Predictions:")
    print(predictions_final)

    print("\nGround Truth:")
    print(target)


if __name__ == "__main__":
    train()