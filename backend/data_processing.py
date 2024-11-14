# data_processing.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize and fit scaler for the expected calories range
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[0], [1000]]))  # Example range for calories

def preprocess_data(calories_input):
    """
    Preprocess the input data (calories) for model inference.

    :param calories_input: Single calories input value.
    :return: Scaled and reshaped input for the model.
    """
    calories_input = np.array([[calories_input]])  # Reshape as 2D array
    scaled_input = scaler.transform(calories_input)  # Scale using pre-fit scaler

    return scaled_input.reshape(-1, 1, 1)  # Reshape for LSTM [samples, time steps, features]
