"""Automated tests for the prediction backend.

Run from the project root:
    python -m pytest backend/tests -v
"""
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.preprocessing import to_feature_row
from backend.app.schemas import EmployeeInput

ROOT = Path(__file__).resolve().parents[2]

HIGH_RISK = {
    "job_satisfaction": 1, "work_life_balance": 1, "appraisal_rating": 1, "does_overtime": "Yes",
    "leaves_taken": 20, "distance_from_home": 35, "date_of_joining": "2021-06-01",
    "years_with_company": 5, "previous_companies": 4, "monthly_salary": 42000,
}
LOW_RISK = {
    "job_satisfaction": 5, "work_life_balance": 4, "appraisal_rating": 5, "does_overtime": "No",
    "leaves_taken": 10, "distance_from_home": 8, "date_of_joining": "2018-02-12",
    "years_with_company": 8, "previous_companies": 1, "monthly_salary": 85000,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the context manager runs start-up, which loads the model
        yield c


@pytest.fixture(scope="module")
def test_set():
    features = joblib.load(ROOT / "models" / "selected_features.pkl")
    X_test = pd.read_csv(ROOT / "processed_data" / "X_test_unscaled.csv")[features]
    return features, X_test


def row_to_input(row: pd.Series) -> dict:
    """Convert a processed test-set row back into the raw form an HR user would enter."""
    return {
        "appraisal_rating": int(row["AppraisalRating"]),
        "job_satisfaction": int(row["JobSatisfaction"]),
        "work_life_balance": int(row["WorkLifeBalance"]),
        "does_overtime": "Yes" if row["DoesOvertime"] == 1 else "No",
        "previous_companies": int(row["PreviousCompanies"]),
        "date_of_joining": f"{int(row['JoiningYear'])}-01-15",
        "years_with_company": int(row["YearsWithCompany"]),
        "monthly_salary": float(row["MonthlySalary"]),
        "distance_from_home": float(row["DistanceFromHome"]),
        "leaves_taken": int(row["LeavesTaken"]),
    }


# ------------------------------------------------------------------ basic endpoints
def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_model_info(client):
    info = client.get("/api/model-info").json()
    assert len(info["features"]) == 10
    assert info["decision_threshold"] == 0.5
    assert info["test_metrics"]["test_size"] == 1000


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Employee Attrition Risk Checker" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


# ------------------------------------------------------------------ same preprocessing as model development
def test_preprocessing_matches_training_data_for_whole_test_set(test_set):
    """Rebuilding every test-set row from raw inputs must give exactly the row the model was evaluated on."""
    features, X_test = test_set
    rebuilt = pd.concat(
        [to_feature_row(EmployeeInput(**row_to_input(row)), features) for _, row in X_test.iterrows()],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(rebuilt.astype(float), X_test.reset_index(drop=True).astype(float))


def test_api_predictions_match_model_directly(client, test_set):
    """The API must return the same probability as calling the saved model on the processed test data."""
    features, X_test = test_set
    model = joblib.load(ROOT / "models" / "final_random_forest_model.pkl")
    sample = X_test.sample(40, random_state=42)
    expected = model.predict_proba(sample)[:, 1]

    for (_, row), expected_probability in zip(sample.iterrows(), expected):
        result = client.post("/api/predict", json=row_to_input(row)).json()
        assert result["probability_of_leaving"] == pytest.approx(expected_probability, abs=1e-4)
        assert result["will_leave"] == (expected_probability >= 0.5)


# ------------------------------------------------------------------ prediction content
def test_high_risk_example(client):
    result = client.post("/api/predict", json=HIGH_RISK).json()
    assert result["risk_level"] == "High"
    assert result["prediction"] == "Likely to leave"
    assert result["probability_of_leaving"] >= 0.5
    drivers = {f["feature"] for f in result["key_factors"] if f["effect"] == "increases risk"}
    assert {"JobSatisfaction", "WorkLifeBalance", "DoesOvertime", "AppraisalRating"} <= drivers
    assert all(f["suggested_action"] for f in result["key_factors"] if f["effect"] == "increases risk")


def test_low_risk_example(client):
    result = client.post("/api/predict", json=LOW_RISK).json()
    assert result["risk_level"] == "Low"
    assert result["prediction"] == "Likely to stay"
    assert result["probability_of_leaving"] < 0.5


def test_contributions_explain_the_difference_from_a_typical_employee(client):
    """Shapley contributions of all 10 features add up to (employee - typical employee) probability."""
    service = client.app.state.model_service
    row = to_feature_row(EmployeeInput(**HIGH_RISK), service.features)
    probability, typical, contributions = service._shapley_contributions(row)
    assert contributions.sum() == pytest.approx(probability - typical, abs=1e-9)


def test_warning_for_value_outside_training_range(client):
    result = client.post("/api/predict", json={**HIGH_RISK, "monthly_salary": 250000})
    assert result.status_code == 200
    assert any("Monthly salary" in w for w in result.json()["warnings"])


def test_no_warnings_for_normal_input(client):
    assert client.post("/api/predict", json=LOW_RISK).json()["warnings"] == []


# ------------------------------------------------------------------ invalid and missing input
@pytest.mark.parametrize("changes, field", [
    ({"job_satisfaction": 7}, "job_satisfaction"),
    ({"work_life_balance": 0}, "work_life_balance"),
    ({"appraisal_rating": 3.5}, "appraisal_rating"),
    ({"does_overtime": "Maybe"}, "does_overtime"),
    ({"monthly_salary": -100}, "monthly_salary"),
    ({"leaves_taken": 400}, "leaves_taken"),
    ({"distance_from_home": "far"}, "distance_from_home"),
    ({"date_of_joining": "not-a-date"}, "date_of_joining"),
    ({"unexpected_field": 1}, "unexpected_field"),
])
def test_invalid_values_are_rejected(client, changes, field):
    response = client.post("/api/predict", json={**HIGH_RISK, **changes})
    assert response.status_code == 422
    assert field in [e["field"] for e in response.json()["errors"]]


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in HIGH_RISK.items() if k != "does_overtime"}
    response = client.post("/api/predict", json=incomplete)
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": "does_overtime", "message": "This field is required"}]


def test_future_joining_date_is_rejected(client):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    response = client.post("/api/predict", json={**HIGH_RISK, "date_of_joining": tomorrow, "years_with_company": 0})
    assert response.status_code == 422
    assert "future" in response.json()["errors"][0]["message"]


def test_years_with_company_longer_than_time_since_joining_is_rejected(client):
    response = client.post("/api/predict", json={**HIGH_RISK, "years_with_company": 20})
    assert response.status_code == 422
    assert "cannot be more than the time since" in response.json()["errors"][0]["message"]


def test_empty_body_is_rejected(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 422
    assert len(response.json()["errors"]) == 10
