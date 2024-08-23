from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
app = Flask(__name__)
CORS(app)
model = joblib.load('parkinsons_model.joblib')
@app.route('/')
def index():
    return 'Parkinson\'s Disease Detection API'
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    name = data['name']
    age = data['age']
    symptoms = data['symptoms']

    symptom_values = np.array([[
        int(symptoms['symptom1']),
        int(symptoms['symptom2']),
        int(symptoms['symptom3']),
        int(symptoms['symptom4']),
    ]])

    prediction = model.predict(symptom_values)
    parkinson_detected = prediction[0] == 1

    if parkinson_detected:
        result = f"{name}, aged {age}, is likely to have Parkinson's Disease based on the symptoms provided."
    else:
        result = f"{name}, aged {age}, is not likely to have Parkinson's Disease based on the symptoms provided."
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
