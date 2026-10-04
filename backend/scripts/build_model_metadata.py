"""
Build models/model_metadata.json from the training/testing data.

The backend uses this file so that it does not need the training data at runtime:
  - training range (min / max) of each input feature  -> warnings for unusual inputs
  - training median of each feature                    -> "typical employee" for explanations
  - probability calibration                            -> realistic "% chance of leaving"
  - final test-set metrics                             -> shown in the frontend

Run from the project root after the final model has been saved by 04_ModelOptimization.ipynb:
    python backend/scripts/build_model_metadata.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "processed_data"
RANDOM_STATE = 42
MODEL_THRESHOLD = 0.5   # decision threshold on the model's raw output (Stage 7)

model = joblib.load(MODELS_DIR / "final_random_forest_model.pkl")
selected_features = joblib.load(MODELS_DIR / "selected_features.pkl")

X_train = pd.read_csv(DATA_DIR / "X_train_unscaled.csv")[selected_features]
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
X_test = pd.read_csv(DATA_DIR / "X_test_unscaled.csv")[selected_features]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

# ------------------------------------------------------------------ calibration (training data only)
# The model was trained with class_weight='balanced', which pushes its outputs up for leavers
# (e.g. 0.87 on average for the high-risk group, where 79% actually left). Sigmoid (Platt) calibration
# maps the raw output to a realistic probability. It is fitted on out-of-fold predictions from the same
# 5-fold stratified CV used in Stage 7, so no test data is involved.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
out_of_fold = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
calibrator = LogisticRegression().fit(out_of_fold.reshape(-1, 1), y_train)
slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def calibrate(raw):
    return 1 / (1 + np.exp(-(slope * np.asarray(raw) + intercept)))


test_pred = model.predict(X_test)
test_raw = model.predict_proba(X_test)[:, 1]
test_calibrated = calibrate(test_raw)

# Calibration is monotonic, so "calibrated >= calibrate(0.5)" gives exactly the same decisions as Stage 7.
calibrated_threshold = float(calibrate(MODEL_THRESHOLD))

# Two risk groups split at the decision threshold, with the share of each that actually left in the test set.
# (Raw outputs are bimodal and finer bands below the threshold do not separate leavers.)
high_risk = test_raw >= MODEL_THRESHOLD
risk_groups = {
    "High": {"employees": int(high_risk.sum()),
             "actually_left_rate": round(float(y_test[high_risk].mean()), 4),
             "mean_predicted": round(float(test_calibrated[high_risk].mean()), 4)},
    "Low": {"employees": int((~high_risk).sum()),
            "actually_left_rate": round(float(y_test[~high_risk].mean()), 4),
            "mean_predicted": round(float(test_calibrated[~high_risk].mean()), 4)},
}

metadata = {
    "model_name": "Random Forest (tuned, top 10 features)",
    "model_settings": {k: v for k, v in model.get_params().items()
                       if k in ["n_estimators", "max_depth", "min_samples_split",
                                "min_samples_leaf", "max_features", "class_weight"]},
    "features": selected_features,
    "model_threshold": MODEL_THRESHOLD,
    "calibration": {
        "method": "sigmoid (Platt), fitted on 5-fold out-of-fold training predictions",
        "slope": slope,
        "intercept": intercept,
        "test_brier_before": round(brier_score_loss(y_test, test_raw), 4),
        "test_brier_after": round(brier_score_loss(y_test, test_calibrated), 4),
        "test_log_loss_before": round(log_loss(y_test, test_raw), 4),
        "test_log_loss_after": round(log_loss(y_test, test_calibrated), 4),
    },
    "decision_threshold": calibrated_threshold,   # not rounded, so it stays exactly equivalent to 0.5
    "company_attrition_rate": round(float(y_train.mean()), 4),
    "training_ranges": {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max())}
                        for f in selected_features},
    "training_medians": {f: float(X_train[f].median()) for f in selected_features},
    "test_metrics": {
        "accuracy": round(accuracy_score(y_test, test_pred), 4),
        "precision": round(precision_score(y_test, test_pred), 4),
        "recall": round(recall_score(y_test, test_pred), 4),
        "f1": round(f1_score(y_test, test_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, test_raw), 4),
        "test_size": int(len(y_test)),
    },
    "risk_groups_on_test_set": risk_groups,
}

out_path = MODELS_DIR / "model_metadata.json"
out_path.write_text(json.dumps(metadata, indent=2))
print(f"Saved {out_path}")
print(json.dumps({k: metadata[k] for k in ["calibration", "decision_threshold", "risk_groups_on_test_set"]}, indent=2))
