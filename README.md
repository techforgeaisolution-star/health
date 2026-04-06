# DiseasePredict

DiseasePredict is a Flask app for educational disease risk screening (Diabetes, Heart Disease, Parkinson's).

## Backend ML improvements included

- Proper sklearn training pipelines (scaling + classifier)
- Calibrated probabilities for SVM-based models using `CalibratedClassifierCV`
- Saved model bundle metadata (`accuracy`, `roc_auc`, calibration bins/rates)
- Backend-driven probability graph data passed to the result UI

## Train models (recommended)

Place datasets here:

- `data/diabetes.csv`
- `data/heart.csv`
- `data/ParkinsonsDisease.csv`

Then run:

```bash
python3 scripts/training/diabetes.py
python3 scripts/training/heart.py
python3 scripts/training/parkinsons.py
```

Model bundles are saved to `models/*.pkl`.

## Run app

```bash
python3 app.py
```

Open `http://127.0.0.1:5000`.

## Note

This project is for educational use only and is not a medical diagnosis system.
