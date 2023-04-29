# create a route to handle the form submission and call the prediction function from model.py
from custom_transformers import DiagnosisEncoder, DataFrameSelector
from flask import Flask, render_template, request, jsonify
from model import load_model, predict_diagnosis


app = Flask(__name__)
model = load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = request.form.getlist('features[]')
        # Convert the features to float
        features = [float(f) for f in features]

        prediction_proba = predict_diagnosis(model, features)

    response = jsonify(prediction_proba.tolist())
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


if __name__ == '__main__':
    app.run(debug=True)
