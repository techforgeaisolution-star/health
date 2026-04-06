from pathlib import Path

from flask import Flask, render_template, request
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.url_map.strict_slashes = False


def _load_model(name: str):
    path = MODEL_DIR / name
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    # Backward compatibility:
    # - old format: sklearn estimator directly
    # - new format: {"model": estimator, "meta": {...}}
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded
    return {"model": loaded, "meta": {}}


diabetes_predict = _load_model("diabetes.pkl")
heart_predict = _load_model("heart.pkl")
parkinsons_predict = _load_model("parkinsons.pkl")


def get_probability(model_bundle, features):
    """Return risk probability 0–100. Handles proba-capable and SVM models."""
    model = model_bundle["model"]
    try:
        proba = model.predict_proba(features)
        return round(float(proba[0][1]) * 100, 1)
    except AttributeError:
        decision = model.decision_function(features)[0]
        prob = 1 / (1 + np.exp(-decision))
        return round(float(prob) * 100, 1)


def risk_level(prob):
    if prob >= 70:
        return "High"
    if prob >= 40:
        return "Moderate"
    return "Low"


def risk_color(prob):
    if prob >= 70:
        return "high"
    if prob >= 40:
        return "moderate"
    return "low"


def get_probability_graph(model_bundle, prob):
    """
    Backend-driven probability visualization data:
    1) Use sklearn calibration bins (saved during training) when available
    2) Fallback to a smooth model-confidence curve centered around prob
    """
    meta = model_bundle.get("meta", {})
    bins = meta.get("calibration_bins")
    rates = meta.get("calibration_rates")
    if bins and rates and len(bins) == len(rates):
        labels = [int(round(float(x) * 100)) for x in bins]
        values = [round(float(y), 3) for y in rates]
        return {
            "type": "calibration",
            "title": "Model Reliability (Sklearn calibration curve)",
            "x_label": "Predicted probability (%)",
            "y_label": "Observed positive rate",
            "labels": labels,
            "values": values,
            "point": round(prob, 1),
        }

    # fallback if old models are loaded
    labels = list(range(0, 101, 5))
    center = float(prob)
    spread = 16.0
    values = [
        round(float(np.exp(-0.5 * ((x - center) / spread) ** 2)), 4) for x in labels
    ]
    return {
        "type": "confidence",
        "title": "Predicted Risk Profile",
        "x_label": "Risk probability (%)",
        "y_label": "Relative density",
        "labels": labels,
        "values": values,
        "point": round(prob, 1),
    }


def model_name(model_bundle):
    meta = model_bundle.get("meta", {})
    return meta.get("model_name", model_bundle["model"].__class__.__name__)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")


@app.route("/heartdisease")
def heartdisease():
    return render_template("heartdisease.html")


@app.route("/predictdiabetes", methods=["POST"])
def predictdiabetes():
    int_features = list(request.form.values())
    processed = [np.array(int_features, dtype=float)]
    prediction = diabetes_predict["model"].predict(processed)
    prob = get_probability(diabetes_predict, processed)
    prob_graph = get_probability_graph(diabetes_predict, prob)

    result = "positive" if prediction[0] == 1 else "negative"
    display_text = (
        "This person has Diabetes"
        if prediction[0] == 1
        else "This person does not have Diabetes"
    )

    return render_template(
        "diabetes_precaution.html",
        output_text=display_text,
        result=result,
        risk_prob=prob,
        risk_level=risk_level(prob),
        risk_color=risk_color(prob),
        disease="Diabetes",
        model_name=model_name(diabetes_predict),
        prob_positive=prob,
        prob_negative=round(100 - prob, 1),
        prob_graph=prob_graph,
        factors=[
            {
                "label": "Glucose",
                "value": min(100, round(float(int_features[1]) / 200 * 100, 1)),
            },
            {
                "label": "BMI",
                "value": min(100, round(float(int_features[5]) / 67 * 100, 1)),
            },
            {
                "label": "Age",
                "value": min(100, round(float(int_features[7]) / 80 * 100, 1)),
            },
            {
                "label": "Insulin",
                "value": min(100, round(float(int_features[4]) / 846 * 100, 1)),
            },
        ],
    )


@app.route("/predictparkinson", methods=["POST"])
def predictparkinsons():
    int_features = list(request.form.values())
    processed = [np.array(int_features, dtype=float)]
    prediction = parkinsons_predict["model"].predict(processed)
    prob = get_probability(parkinsons_predict, processed)
    prob_graph = get_probability_graph(parkinsons_predict, prob)

    result = "positive" if prediction[0] == 1 else "negative"
    display_text = (
        "This person has Parkinson's Disease"
        if prediction[0] == 1
        else "This person does not have Parkinson's Disease"
    )

    return render_template(
        "parkinsons_precaution.html",
        output_text=display_text,
        result=result,
        risk_prob=prob,
        risk_level=risk_level(prob),
        risk_color=risk_color(prob),
        disease="Parkinson's",
        model_name=model_name(parkinsons_predict),
        prob_positive=prob,
        prob_negative=round(100 - prob, 1),
        prob_graph=prob_graph,
        factors=[
            {
                "label": "Jitter",
                "value": min(100, round(float(int_features[3]) / 0.03316 * 100, 1)),
            },
            {
                "label": "Shimmer",
                "value": min(100, round(float(int_features[8]) / 0.11908 * 100, 1)),
            },
            {
                "label": "HNR (inv)",
                "value": min(
                    100,
                    round((33.047 - float(int_features[15])) / 33.047 * 100, 1),
                ),
            },
            {
                "label": "PPE",
                "value": min(100, round(float(int_features[21]) / 0.527367 * 100, 1)),
            },
        ],
    )


@app.route("/predictheartdisease", methods=["POST"])
def predictheartdisease():
    int_features = list(request.form.values())
    processed = [np.array(int_features, dtype=float)]
    prediction = heart_predict["model"].predict(processed)
    prob = get_probability(heart_predict, processed)
    prob_graph = get_probability_graph(heart_predict, prob)

    result = "positive" if prediction[0] == 1 else "negative"
    display_text = (
        "This person has Heart Disease"
        if prediction[0] == 1
        else "This person does not have Heart Disease"
    )

    return render_template(
        "heart_precaution.html",
        output_text=display_text,
        result=result,
        risk_prob=prob,
        risk_level=risk_level(prob),
        risk_color=risk_color(prob),
        disease="Heart Disease",
        model_name=model_name(heart_predict),
        prob_positive=prob,
        prob_negative=round(100 - prob, 1),
        prob_graph=prob_graph,
        factors=[
            {
                "label": "Cholesterol",
                "value": min(100, round(float(int_features[4]) / 564 * 100, 1)),
            },
            {
                "label": "Age",
                "value": min(100, round(float(int_features[0]) / 80 * 100, 1)),
            },
            {
                "label": "Low Max HR",
                "value": min(
                    100, round((202 - float(int_features[7])) / 202 * 100, 1)
                ),
            },
            {
                "label": "ST Depression",
                "value": min(100, round(float(int_features[9]) / 6.2 * 100, 1)),
            },
        ],
    )


if __name__ == "__main__":
    app.run(debug=True)
