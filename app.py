import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.url_map.strict_slashes = False

SRC_DIR = BASE_DIR / "src"


@app.route("/src/<path:filename>")
def serve_src(filename):
    return send_from_directory(SRC_DIR, filename)


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


@app.route("/about")
def about():
    return render_template("about.html")


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


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    
    if not user_msg:
        return jsonify({"response": "Please ask me a question!"})
    
    response = get_chatbot_response(user_msg)
    return jsonify({"response": response})


def get_chatbot_response(msg):
    msg_lower = msg.lower()
    
    # Greetings
    if any(w in msg_lower for w in ["hi", "hello", "hey"]):
        return "Hello! I'm HealthGuide, your assistant for DiseasePredict. I can help you understand our screening tools, explain input fields, interpret results, and answer privacy questions. What would you like to know?"
    
    # Privacy
    if "privacy" in msg_lower or "store" in msg_lower or "data" in msg_lower:
        return "**Privacy First:** Your health data is never stored. Each prediction is a one-off server request — no accounts, no databases, no tracking. Your information stays private and is only used for that single prediction."
    
    # Accuracy
    if "accura" in msg_lower or "reliable" in msg_lower or "trust" in msg_lower:
        return "Our models are trained on **real clinical datasets** from the UCI ML Repository. However, these are educational tools — not medical diagnostic systems. Results should be used to prepare questions for your doctor, not for self-diagnosis. Always consult a healthcare professional."
    
    # Diabetes
    if "diabetes" in msg_lower:
        if "glucose" in msg_lower:
            return "**Glucose** measures blood sugar levels. Normal fasting glucose is **70–99 mg/dL**. Values of 100–125 mg/dL indicate prediabetes, and 126+ mg/dL suggests diabetes. High glucose is a key risk factor the model evaluates."
        if "bmi" in msg_lower:
            return "**BMI (Body Mass Index)** relates weight to height. Normal range is **18.5–24.9**. Higher BMI (25+) increases diabetes risk because excess body fat affects insulin sensitivity."
        return "**Diabetes Risk Assessment** uses 8 inputs: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. You'll get a risk score, factor breakdown, and lifestyle precautions. Remember — this is educational, not a diagnosis."
    
    # Heart Disease
    if "heart" in msg_lower or "cardiac" in msg_lower:
        if "cholesterol" in msg_lower:
            return "**Cholesterol** measures blood fats. Normal total cholesterol is **below 200 mg/dL**. High cholesterol (240+ mg/dL) increases heart disease risk by contributing to artery blockages."
        if "ecg" in msg_lower or "resting" in msg_lower:
            return "**Resting ECG** shows your heart's electrical activity at rest. Normal results indicate regular rhythm. Abnormalities may suggest heart strain or previous issues, which the model considers."
        return "**Heart Disease Risk Assessment** uses 13 cardiac features including age, chest pain type, blood pressure, cholesterol, ECG results, max heart rate, and more. You'll see risk charts and precautions. Always confirm with a cardiologist."
    
    # Parkinson's
    if "parkinson" in msg_lower:
        if "hnr" in msg_lower:
            return "**HNR (Harmonics-to-Noise Ratio)** measures voice clarity. Normal HNR is typically **above 20 dB**. Lower values indicate breathiness or hoarseness, which can be early Parkinson's signs detected through voice analysis."
        if "jitter" in msg_lower:
            return "**Jitter** measures voice frequency variation. Normal jitter is **below 1%**. Higher jitter indicates vocal instability, a potential biomarker for Parkinson's disease."
        if "shimmer" in msg_lower:
            return "**Shimmer** measures voice amplitude variation. Normal shimmer is **below 3.5%**. Elevated shimmer suggests vocal cord irregularities linked to neurological conditions."
        return "**Parkinson's Speech Screening** analyzes voice biomarkers like jitter, shimmer, HNR, and PPE. These measurements come from specialized voice recordings. Results are research-based but must be confirmed by a neurologist."
    
    # PDF/Download
    if "pdf" in msg_lower or "download" in msg_lower or "report" in msg_lower:
        return "After getting your results, click the **Download PDF Report** button on the results page. This generates a summary you can save or share with your healthcare provider. The PDF includes your risk score, factor breakdown, and precautions."
    
    # Tools comparison
    if "difference" in msg_lower or "which tool" in msg_lower or "choose" in msg_lower:
        return "**Choose based on your data:**\n\n• **Diabetes** — if you have lab values (glucose, BMI, insulin, etc.)\n• **Heart Disease** — if you have cardiac metrics (cholesterol, ECG, blood pressure)\n• **Parkinson's** — if you have voice biomarker measurements\n\nEach tool uses different inputs and models. Pick the one matching your available health data."
    
    # Medical advice
    if "diagnos" in msg_lower or "treatment" in msg_lower or "medication" in msg_lower:
        return "⚠️ **Important:** I cannot provide medical diagnoses or treatment advice. DiseasePredict is for educational purposes only. If you have health concerns, please consult a qualified healthcare professional immediately."
    
    # Urgent symptoms
    if any(w in msg_lower for w in ["chest pain", "can't breathe", "emergency", "urgent"]):
        return "🚨 **URGENT:** If you're experiencing chest pain, difficulty breathing, or other emergency symptoms, please call emergency services (911) or go to the nearest hospital immediately. Do not rely on this tool for urgent medical situations."
    
    # Default
    return "I can help you with:\n\n• **Understanding screening tools** (Diabetes, Heart Disease, Parkinson's)\n• **Explaining input fields** and normal ranges\n• **Interpreting results** in plain language\n• **Privacy questions**\n• **Downloading PDF reports**\n\nWhat specific question do you have?"


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
