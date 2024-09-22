from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request body
    data = request.get_json(force=True)
    
    # Extract features from the JSON data
    Sepal_Length = data['Sepal_Length']
    Sepal_Width = data['Sepal_Width']
    Petal_Length = data['Petal_Length']
    Petal_Width = data['Petal_Width']
    
    # Convert the data into the format the model expects
    features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
    
    # Use the model to predict
    prediction = model.predict(features)
    print(prediction)
    # Convert the prediction into a response
    return jsonify({
        'prediction': "The flower species is {}".format(prediction)  # Send the class prediction in JSON format
    })

if __name__ == '__main__':
    app.run(debug=True)
