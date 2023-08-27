import keras
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
df = pd.read_csv('data1.csv')

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


# Example input data for prediction
example_data = {
    'cap-shape': ['x'],
    'cap-surface': ['s'],
    'cap-color': ['n'],
    'bruises': ['t'],
    'odor': ['p'],
    'gill-attachment': ['f'],
    'gill-spacing': ['c'],
    'gill-size': ['n'],
    'gill-color': ['k'],
    'stalk-shape': ['e'],
    'stalk-root': ['e'],
    'stalk-surface-above-ring': ['s'],
    'stalk-surface-below-ring': ['s'],
    'stalk-color-above-ring': ['w'],
    'stalk-color-below-ring': ['w'],
    'veil-type': ['p'],
    'veil-color': ['w'],
    'ring-number': ['o'],
    'ring-type': ['p'],
    'spore-print-color': ['k'],
    'population': ['s'],
    'habitat': ['u']
}

# Create the example input DataFrame
example_input_data = pd.DataFrame(example_data)

# Make predictions
predicted_probabilities = predict_mushroom_poisonous(example_input_data)

# Convert probabilities to class labels (0 or 1)
predicted_classes = (predicted_probabilities > 0.5).astype(int)

print("Probabilities:", predicted_probabilities)
print("Predicted classes:", predicted_classes)

# Check the prediction result
if predicted_classes[0][0] == 0:
    result = "Edible"
else:
    result = "Poisonous"

print("Predicted:", result)
