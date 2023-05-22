#######################################################
# CISB60 Model Deployment Lab Assignment: App File    #  
# Submission(s): Jovanny Gonzalez and Yvette Vargas   #
# Group 1 Members: Mandy Liu                          # 
# May 23, 2023                                        #
#######################################################

#import flask libraries 
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

#import other libraries 
import pandas as pd
from sklearn.linear_model import LinearRegression 
import pickle

#assign app variable with secret key 
app = Flask(__name__)
app.secret_key = "SECRET_KEY"

#define route to submission html page in templates folder 
@app.route('/')
def index():
    return render_template("submission.html")

#define route to create predictions based on linear regression model 
@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():

    #assign dep_delay variable to DepDelay column in original dataset 
    dep_delay = float(request.form["DepDelay"])

    #assign unpickled_linearModel to open linearmodel.pkl file to read binary 
    #inserted full directory path for file 
    unpickled_linearModel = pickle.load(open('model_deployment/linearModel.pkl', 'rb'))

    #assign prediction variable to assign predictions from dep_delay and compute linear regression predictions 
    prediction = unpickled_linearModel.predict([[dep_delay]])

    #display predictions in prediction.html file from the first column and first entry 
    return render_template("prediction.html", prediction = prediction[0][0])

#debug 
if __name__ == "__main__":
    app.run(debug=True)

##################################################
#                      END                       #                      
##################################################