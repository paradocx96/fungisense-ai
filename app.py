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


def input_data_maker(data):
    if data:
        color_mapping = {
            "bell": "b", "conical": "c", "convex": "x", "flat": "f", "knobbed": "k", "sunken": "s", "fibrous": "f",
            "grooves": "g", "scaly": "y", "smooth": "s", "brown": "n", "buff": "b", "cinnamon": "c", "gray": "g",
            "green": "r", "pink": "p", "purple": "u", "red": "e", "white": "w", "yellow": "y", "bruises": "t",
            "no": "f", "almond": "a", "anise": "l", "creosote": "c", "fishy": "y", "foul": "f", "musty": "m",
            "none": "n", "pungent": "p", "spicy": "s", "attached": "a", "descending": "d", "free": "f", "notched": "n",
            "close": "c", "crowded": "w", "distant": "d", "broad": "b", "narrow": "n", "black": "k", "chocolate": "h",
            "orange": "o", "enlarging": "e", "tapering": "t", "bulbous": "b", "club": "c", "cup": "u", "equal": "e",
            "rhizomorphs": "z", "rooted": "r", "missing": "?", "silky": "k", "partial": "p", "universal": "u",
            "one": "o", "two": "t", "cobwebby": "c", "evanescent": "e", "flaring": "f", "large": "l", "pendant": "p",
            "sheathing": "s", "zone": "z", "abundant": "a", "clustered": "c", "numerous": "n", "scattered": "s",
            "several": "v", "solitary": "y", "grasses": "g", "leaves": "l", "meadows": "m", "paths": "p", "urban": "u",
            "waste": "w", "woods": "d"
        }

        cap_shape = color_mapping.get(data.get("cap-shape"), "")
        cap_surface = color_mapping.get(data.get("cap-surface"), "")
        cap_color = color_mapping.get(data.get("cap-color"), "")
        bruises = color_mapping.get(data.get("bruises"), "")
        odor = color_mapping.get(data.get("odor"), "")
        gill_attachment = color_mapping.get(data.get("gill-attachment"), "")
        gill_spacing = color_mapping.get(data.get("gill-spacing"), "")
        gill_size = color_mapping.get(data.get("gill-size"), "")
        gill_color = color_mapping.get(data.get("gill-color"), "")
        stalk_shape = color_mapping.get(data.get("stalk-shape"), "")
        stalk_root = color_mapping.get(data.get("stalk-root"), "")
        stalk_surface_above_ring = color_mapping.get(data.get("stalk-surface-above-ring"), "")
        stalk_surface_below_ring = color_mapping.get(data.get("stalk-surface-below-ring"), "")
        stalk_color_above_ring = color_mapping.get(data.get("stalk-color-above-ring"), "")
        stalk_color_below_ring = color_mapping.get(data.get("stalk-color-below-ring"), "")
        veil_type = color_mapping.get(data.get("veil-type"), "")
        veil_color = color_mapping.get(data.get("veil-color"), "")
        ring_number = color_mapping.get(data.get("ring-number"), "")
        ring_type = color_mapping.get(data.get("ring-type"), "")
        spore_print_color = color_mapping.get(data.get("spore-print-color"), "")
        population = color_mapping.get(data.get("population"), "")
        habitat = color_mapping.get(data.get("habitat"), "")

        json_data = {
            "cap-shape": [cap_shape],
            "cap-surface": [cap_surface],
            "cap-color": [cap_color],
            "bruises": [bruises],
            "odor": [odor],
            "gill-attachment": [gill_attachment],
            "gill-spacing": [gill_spacing],
            "gill-size": [gill_size],
            "gill-color": [gill_color],
            "stalk-shape": [stalk_shape],
            "stalk-root": [stalk_root],
            "stalk-surface-above-ring": [stalk_surface_above_ring],
            "stalk-surface-below-ring": [stalk_surface_below_ring],
            "stalk-color-above-ring": [stalk_color_above_ring],
            "stalk-color-below-ring": [stalk_color_below_ring],
            "veil-type": [veil_type],
            "veil-color": [veil_color],
            "ring-number": [ring_number],
            "ring-type": [ring_type],
            "spore-print-color": [spore_print_color],
            "population": [population],
            "habitat": [habitat]
        }

        return json_data

    return None


# Define a route to predict the mushroom poisonous
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the Request JSON data
        data = request.get_json()

        if data:
            # Modify the input data
            json_data = input_data_maker(data)

            # Create the input data DataFrame
            input_data = pd.DataFrame(json_data)

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
            return jsonify({
                "prediction": result,
                "error": False,
                "message": "Successfully predicted",
                "status": 200
            })
        else:
            # Return an error message
            return jsonify({
                "prediction": None,
                "error": True,
                "message": "Data not sufficient",
                "status": 403
            })

    # Handle errors
    except Exception as e:
        return jsonify({
            "prediction": None,
            "error": True,
            "message": str(e),
            "status": 500
        })


@app.route('/')
def home():
    return jsonify({
        "prediction": None,
        "error": False,
        "message": "API is Running!",
        "status": 200
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
