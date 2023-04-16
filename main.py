from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd

app = Flask(__name__)


# Load the pre-trained machine learning models
def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')


@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')


@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')


@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)

    except:
        message = "Please enter valid Data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)


if __name__ == '__main__':
    app.run(debug=True)
