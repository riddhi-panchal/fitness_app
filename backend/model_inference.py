# model_inference.py
from tensorflow.keras.models import load_model

# Load the trained models
distance_model = load_model("models/distance_recommendation_model.keras")
speed_model = load_model("models/speed_sequence_model.keras")
heart_rate_model = load_model("models/heart_rate_sequence_model.keras")

def generate_recommendations(processed_input):
    """
    Given processed input, generate the corresponding speed sequence,
    recommended distance, and heart rate sequence.

    :param processed_input: Scaled and preprocessed input for the model.
    :return: Dictionary containing speed sequence, recommended distance, and heart rate sequence.
    """
    # Predict the outputs using the respective models
    predicted_distance = distance_model.predict(processed_input)
    predicted_speed = speed_model.predict(processed_input)
    predicted_heart_rate = heart_rate_model.predict(processed_input)

    return {
        "predicted_distance": predicted_distance,
        "predicted_speed": predicted_speed,
        "predicted_heart_rate": predicted_heart_rate
    }
