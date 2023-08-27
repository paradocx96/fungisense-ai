import keras
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load your dataset
df = pd.read_csv('mushrooms.csv')

# Preprocess input data
encoder = OneHotEncoder(drop='first')
X = encoder.fit_transform(df.drop(['class'], axis=1))

# Load the model
loaded_model = keras.models.load_model('model.h5')


# Function to preprocess input data
def preprocess_input(input_data):
    # Transform the input data using the encoder
    encoded_input = encoder.transform(input_data)
    return encoded_input


# Function to predict mushroom poisonous
def predict_mushroom_poisonous(input_data):
    # Preprocess the input data
    preprocessed_input = preprocess_input(input_data)

    # Make predictions
    predictions = loaded_model.predict(preprocessed_input)
    return predictions


# Define a route to predict the mushroom poisonous
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the Request JSON data
        data = request.get_json()

        if data:
            # Create the input data DataFrame
            input_data = pd.DataFrame(data)

            # Make predictions
            predicted_probabilities = predict_mushroom_poisonous(input_data)

            # Get the predicted class
            predicted_classes = (predicted_probabilities > 0.5).astype(int)

            # Check the predicted class
            if predicted_classes[0][0] == 0:
                result = "Edible"
            else:
                result = "Poisonous"

            # Return the prediction result
            return jsonify({"prediction": result})
        else:
            # Return an error message
            return jsonify({"error": "No data provided"})

    # Handle errors
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/')
def home():
    return jsonify({"message": "API is Running!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
