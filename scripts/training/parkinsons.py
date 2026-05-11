"""Train calibrated Parkinson's model and save bundle to models/parkinsons.pkl."""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "ParkinsonsDisease.csv"
MODEL_PATH = ROOT / "models" / "parkinsons.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns="name", axis=1)

    x = df.drop(columns="status", axis=1)
    y = df["status"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
        ]
    )
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    bins, rates = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")

    bundle = {
        "model": model,
        "meta": {
            "model_name": "Calibrated LinearSVC",
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "calibration_bins": [float(v) for v in bins],
            "calibration_rates": [float(v) for v in rates],
            "dataset": "ParkinsonsDisease.csv",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print("Saved:", MODEL_PATH)
    print("Accuracy:", bundle["meta"]["accuracy"])
    print("ROC-AUC:", bundle["meta"]["roc_auc"])


if __name__ == "__main__":
    main()
