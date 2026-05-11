"""Train calibrated heart model and save bundle to models/heart.pkl."""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "models" / "heart.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df.iloc[:, 0:13]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    bins, rates = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")

    bundle = {
        "model": model,
        "meta": {
            "model_name": "Logistic Regression (scaled)",
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "calibration_bins": [float(v) for v in bins],
            "calibration_rates": [float(v) for v in rates],
            "dataset": "heart.csv",
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
