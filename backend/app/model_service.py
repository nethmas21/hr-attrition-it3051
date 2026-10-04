"""Loads the final model once and turns a validated employee into a prediction with explanations."""
import json
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import to_feature_row
from .schemas import EmployeeInput

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Minimum contribution (in probability) for a feature to be reported as a key factor
MIN_IMPACT = 0.02
MAX_FACTORS = 5

# Human-readable label, value format and HR action for each model feature
FEATURE_INFO = {
    "AppraisalRating": ("Appraisal rating", lambda v: f"{v:.0f} out of 5",
                        "Agree a development plan with clear goals, coaching and regular feedback."),
    "JobSatisfaction": ("Job satisfaction", lambda v: f"{v:.0f} out of 5",
                        "Hold a one-to-one conversation to understand what is causing dissatisfaction with the role."),
    "DoesOvertime": ("Works overtime", lambda v: "Yes" if v == 1 else "No",
                     "Review workload and overtime hours; consider redistributing tasks."),
    "WorkLifeBalance": ("Work-life balance", lambda v: f"{v:.0f} out of 4",
                        "Discuss flexible working hours or hybrid / remote-work options."),
    "PreviousCompanies": ("Previous companies", lambda v: f"{v:.0f}",
                          "Strengthen engagement with mentoring and a clear career path."),
    "JoiningYear": ("Year of joining", lambda v: f"{v:.0f}",
                    "Review career progression and recognition for employees at this stage of tenure."),
    "MonthlySalary": ("Monthly salary", lambda v: f"INR {v:,.0f}",
                      "Benchmark salary against the market and internal pay bands."),
    "YearsWithCompany": ("Years with the company", lambda v: f"{v:.0f} years",
                         "Review career progression and recognition for employees at this stage of tenure."),
    "DistanceFromHome": ("Distance from home", lambda v: f"{v:.1f} km",
                         "Consider hybrid working or travel support."),
    "LeavesTaken": ("Leave days taken", lambda v: f"{v:.0f} days",
                    "Check in on the employee's wellbeing and workload."),
}


@dataclass
class ModelService:
    model: object
    features: list[str]
    metadata: dict

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "ModelService":
        model = joblib.load(models_dir / "final_random_forest_model.pkl")
        features = joblib.load(models_dir / "selected_features.pkl")
        metadata = json.loads((models_dir / "model_metadata.json").read_text())

        if list(getattr(model, "feature_names_in_", features)) != list(features):
            raise RuntimeError("selected_features.pkl does not match the features the model was trained on")
        if metadata["features"] != list(features):
            raise RuntimeError("model_metadata.json is out of date - run backend/scripts/build_model_metadata.py")
        unknown = set(features) - set(FEATURE_INFO)
        if unknown:
            raise RuntimeError(f"No display information for model features: {sorted(unknown)}")

        return cls(model=model, features=list(features), metadata=metadata)

    # ------------------------------------------------------------------ prediction
    def predict(self, employee: EmployeeInput) -> dict:
        row = to_feature_row(employee, self.features)
        threshold = self.metadata["decision_threshold"]

        probability, typical_probability, contributions = self._shapley_contributions(row)
        will_leave = probability >= threshold
        risk_level = "High" if will_leave else "Low"

        return {
            "prediction": "Likely to leave" if will_leave else "Likely to stay",
            "will_leave": will_leave,
            "probability_of_leaving": round(probability, 4),
            "typical_employee_probability": round(typical_probability, 4),
            "risk_level": risk_level,
            "risk_explanation": self._risk_explanation(risk_level),
            "decision_threshold": threshold,
            "key_factors": self._key_factors(row, contributions),
            "warnings": self._range_warnings(row),
        }

    def _shapley_contributions(self, row: pd.DataFrame):
        """Exact Shapley values against a 'typical employee' (training medians).

        Every combination of the employee's own values and the typical values is scored
        (2^10 = 1,024 rows, one batch). Each feature's contribution is its average effect on the
        probability of leaving across all those combinations, so the contributions add up exactly to
        (this employee's probability - typical employee's probability).
        """
        n = len(self.features)
        employee = row.to_numpy(dtype=float)[0]
        typical = np.array([self.metadata["training_medians"][f] for f in self.features])

        masks = np.arange(2 ** n)
        uses_employee_value = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
        combinations = pd.DataFrame(np.where(uses_employee_value, employee, typical), columns=self.features)
        probabilities = self.model.predict_proba(combinations)[:, 1]

        subset_size = uses_employee_value.sum(axis=1)
        weights = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])

        contributions = np.zeros(n)
        for i in range(n):
            without_i = ~uses_employee_value[:, i]
            subsets = masks[without_i]
            contributions[i] = np.sum(
                weights[subset_size[without_i]] * (probabilities[subsets | (1 << i)] - probabilities[subsets])
            )
        # masks[-1] = all employee values, masks[0] = all typical values
        return float(probabilities[-1]), float(probabilities[0]), contributions

    def _key_factors(self, row: pd.DataFrame, contributions) -> list[dict]:
        factors = []
        for feature, impact in zip(self.features, contributions):
            impact = float(impact)
            if abs(impact) < MIN_IMPACT:
                continue
            label, fmt, action = FEATURE_INFO[feature]
            increases = impact > 0
            factors.append({
                "feature": feature,
                "label": label,
                "value": fmt(row[feature].iloc[0]),
                "typical_value": fmt(self.metadata["training_medians"][feature]),
                "effect": "increases risk" if increases else "reduces risk",
                "impact": round(impact, 4),
                "suggested_action": action if increases else None,
            })
        factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
        return factors[:MAX_FACTORS]

    def _range_warnings(self, row: pd.DataFrame) -> list[str]:
        warnings = []
        for feature in self.features:
            value = float(row[feature].iloc[0])
            bounds = self.metadata["training_ranges"][feature]
            if not bounds["min"] <= value <= bounds["max"]:
                label, fmt, _ = FEATURE_INFO[feature]
                warnings.append(
                    f"{label} ({fmt(value)}) is outside the range seen in the training data "
                    f"({fmt(bounds['min'])} to {fmt(bounds['max'])}), so this prediction may be less reliable."
                )
        return warnings

    def _risk_explanation(self, risk_level: str) -> str:
        group = self.metadata["risk_groups_on_test_set"][risk_level]
        rate = round(group["actually_left_rate"] * 100)
        tested = f"When tested on {self.metadata['test_metrics']['test_size']:,} past employees"
        if risk_level == "High":
            return f"{tested}, {rate}% of those placed in the high-risk group actually left."
        return f"{tested}, {rate}% of those placed in the low-risk group still left, so low risk does not mean no risk."

    def model_info(self) -> dict:
        return {
            "model_name": self.metadata["model_name"],
            "model_settings": self.metadata["model_settings"],
            "features": [{"feature": f, "label": FEATURE_INFO[f][0],
                          "training_range": self.metadata["training_ranges"][f],
                          "typical_value": self.metadata["training_medians"][f]} for f in self.features],
            "decision_threshold": self.metadata["decision_threshold"],
            "test_metrics": self.metadata["test_metrics"],
            "risk_groups_on_test_set": self.metadata["risk_groups_on_test_set"],
        }
