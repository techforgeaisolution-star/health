from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

diabetes_predict = pickle.load(open("diabetes.pkl", "rb"))
heart_predict = pickle.load(open("heart.pkl", "rb"))
parkinsons_predict = pickle.load(open("parkinsons.pkl", "rb"))


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/parkinsons/")
def parkinsons():
    return render_template("parkinsons.html")


@app.route("/heartdisease/")
def heartdisease():
    return render_template("heartdisease.html")


@app.route("/predictdiabetes/", methods=["POST"])
def predictdiabetes():
    int_features = [x for x in request.form.values()]
    processed_feature_diabetes = [np.array(int_features, dtype=float)]
    prediction = diabetes_predict.predict(processed_feature_diabetes)
    if prediction[0] == 1:
        result = "positive"
        display_text = "This person has Diabetes"
    else:
        result = "negative"
        display_text = "This person does not have Diabetes"
    return render_template(
        "diabetesprecaution.html",
        output_text=display_text,
        result=result
    )


@app.route("/predictparkinson/", methods=["POST"])
def predictparkinsons():
    int_features = [x for x in request.form.values()]
    processed_feature_parkinsons = [np.array(int_features, dtype=float)]
    prediction = parkinsons_predict.predict(processed_feature_parkinsons)
    if prediction[0] == 1:
        result = "positive"
        display_text = "This person has Parkinson's Disease"
    else:
        result = "negative"
        display_text = "This person does not have Parkinson's Disease"
    return render_template(
        "parikson_precaution.html",
        output_text=display_text,
        result=result
    )


@app.route("/predictheartdisease/", methods=["POST"])
def predictheartdisease():
    int_features = [x for x in request.form.values()]
    processed_feature_heart = [np.array(int_features, dtype=float)]
    prediction = heart_predict.predict(processed_feature_heart)
    if prediction[0] == 1:
        result = "positive"
        display_text = "This person has Heart Disease"
    else:
        result = "negative"
        display_text = "This person does not have Heart Disease"
    return render_template(
        "heartPrecautions.html",
        output_text=display_text,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
