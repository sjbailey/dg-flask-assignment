#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 18:07:19 2021

@author: Samuel Bailey
As part of my submission for Week 4 Assignment of the Data Glacier Virtual Internship
Adapted from code written by Abhinav Sagar and available on his blog, Towards Data Science (https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4)
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    return render_template("index.html", prediction_text = "Given this waiting time, Old Faithful's next eruption is estimated to last {} seconds.".format(output))

if __name__ == "__main__":
    app.run(port = 5000, debug = True)