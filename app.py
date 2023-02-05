import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

# create flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["post"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    print(float_features)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print(prediction)

    if(prediction == "M"):
        pred_text= "The Breast Cancer is Malignant"
    elif(prediction == "B"):
        pred_text = "The Breast Cancer is Benign"

    return render_template("home.html", prediction_text = pred_text)


if __name__== "__main__":
    app.run(debug=True)

