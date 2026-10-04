"""
Build models/model_metadata.json from the training/testing data.

The backend uses this file so that it does not need the training data at runtime:
  - training range (min / max) of each input feature  -> warnings for unusual inputs
  - training median of each feature                    -> "typical employee" for explanations
  - final test-set metrics                             -> shown in the frontend

Run from the project root after the final model has been saved by 04_ModelOptimization.ipynb:
    python backend/scripts/build_model_metadata.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "processed_data"

model = joblib.load(MODELS_DIR / "final_random_forest_model.pkl")
selected_features = joblib.load(MODELS_DIR / "selected_features.pkl")

X_train = pd.read_csv(DATA_DIR / "X_train_unscaled.csv")[selected_features]
X_test = pd.read_csv(DATA_DIR / "X_test_unscaled.csv")[selected_features]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

# The predicted probabilities are bimodal (most employees score below 0.4 or above 0.8), and finer
# bands below 0.5 do not separate leavers on the test set. Two risk groups split at the decision
# threshold are therefore used, with the share of each group that actually left in the test set.
high_risk = test_prob >= 0.5
risk_groups = {
    "High": {"employees": int(high_risk.sum()),
             "actually_left_rate": round(float(y_test[high_risk].mean()), 4)},
    "Low": {"employees": int((~high_risk).sum()),
            "actually_left_rate": round(float(y_test[~high_risk].mean()), 4)},
}

metadata = {
    "model_name": "Random Forest (tuned, top 10 features)",
    "model_settings": {k: v for k, v in model.get_params().items()
                       if k in ["n_estimators", "max_depth", "min_samples_split",
                                "min_samples_leaf", "max_features", "class_weight"]},
    "features": selected_features,
    "decision_threshold": 0.5,
    "training_ranges": {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max())}
                        for f in selected_features},
    "training_medians": {f: float(X_train[f].median()) for f in selected_features},
    "test_metrics": {
        "accuracy": round(accuracy_score(y_test, test_pred), 4),
        "precision": round(precision_score(y_test, test_pred), 4),
        "recall": round(recall_score(y_test, test_pred), 4),
        "f1": round(f1_score(y_test, test_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, test_prob), 4),
        "test_size": int(len(y_test)),
    },
    "risk_groups_on_test_set": risk_groups,
}

out_path = MODELS_DIR / "model_metadata.json"
out_path.write_text(json.dumps(metadata, indent=2))
print(f"Saved {out_path}")
print(json.dumps(metadata["test_metrics"], indent=2))
