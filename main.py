from flask import Flask, request
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

#http://localhost:5000/Mobile_Price_Prediction

model_pk = pickle.load(open('Mobile_Price_Pred.pkl' , 'rb'))

@app.route('/Mobile_Price_Prediction' , methods=["GET", "POST"])
def Mobile_Price_Prediction():
    if request.method == "GET":
        return 'Please send post request'
    elif request.method == "POST":
        data = request.get_json()
        
        resoloution=data["resoloution"]
        ppi=data["ppi"]
        cpu_core=data["cpu_core"]
        ram=data["ram"]
        RearCam=data["RearCam"]
        Front_Cam=data["Front_Cam"]
        battery=data["battery"]

        in1 = np.array([[resoloution,ppi,cpu_core,ram,RearCam,Front_Cam,battery]])
        prediction = model_pk.predict(in1)
        return str(prediction)
    
app.run()