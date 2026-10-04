"""Turn validated user input into the exact feature row the model was trained on.

This mirrors the steps in notebooks/02_Preprocessing_FeatureEngineering .ipynb that affect
the 10 selected features:
  - DoesOvertime: "No" -> 0, "Yes" -> 1             (Section 19, binary mapping)
  - JoiningYear: year part of DateOfJoining           (Section 15, date-derived features)
  - all other selected features are used unchanged    (no scaling: Random Forest is scale-invariant)
Columns are returned in the order stored in models/selected_features.pkl.
"""
import pandas as pd

from .schemas import EmployeeInput

OVERTIME_MAP = {"No": 0, "Yes": 1}


def to_feature_row(employee: EmployeeInput, feature_order: list[str]) -> pd.DataFrame:
    features = {
        "AppraisalRating": employee.appraisal_rating,
        "JobSatisfaction": employee.job_satisfaction,
        "DoesOvertime": OVERTIME_MAP[employee.does_overtime],
        "WorkLifeBalance": employee.work_life_balance,
        "PreviousCompanies": employee.previous_companies,
        "JoiningYear": employee.date_of_joining.year,
        "MonthlySalary": employee.monthly_salary,
        "YearsWithCompany": employee.years_with_company,
        "DistanceFromHome": employee.distance_from_home,
        "LeavesTaken": employee.leaves_taken,
    }

    missing = set(feature_order) - set(features)
    if missing:
        # The saved model expects a feature this code does not build - fail loudly rather than guess.
        raise RuntimeError(f"Preprocessing does not produce required model features: {sorted(missing)}")

    return pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
