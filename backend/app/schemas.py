"""Request and response models for the prediction API (input validation lives here)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmployeeInput(BaseModel):
    """The 10 employee details the final model needs.

    Hard limits reject impossible values (e.g. a rating of 7 or a negative salary).
    Values that are possible but outside the range seen in the training data are accepted,
    and the response carries a warning instead (see model_service.range_warnings).
    """

    model_config = ConfigDict(extra="forbid")  # unknown / misspelt fields are rejected

    appraisal_rating: int = Field(..., ge=1, le=5, description="Latest appraisal rating, 1 (lowest) to 5 (highest)")
    job_satisfaction: int = Field(..., ge=1, le=5, description="Job satisfaction survey score, 1 (very low) to 5 (very high)")
    work_life_balance: int = Field(..., ge=1, le=4, description="Work-life balance survey score, 1 (poor) to 4 (excellent)")
    does_overtime: Literal["Yes", "No"] = Field(..., description="Does the employee regularly work overtime?")
    previous_companies: int = Field(..., ge=0, le=30, description="Number of companies worked for before joining")
    date_of_joining: date = Field(..., description="Date the employee joined the company (YYYY-MM-DD)")
    years_with_company: int = Field(..., ge=0, le=50, description="Completed years with the company")
    monthly_salary: float = Field(..., gt=0, le=1_000_000, description="Monthly salary in INR")
    distance_from_home: float = Field(..., ge=0, le=500, description="Distance from home to the office in km")
    leaves_taken: int = Field(..., ge=0, le=365, description="Leave days taken in the last year")

    @model_validator(mode="after")
    def check_dates_are_consistent(self):
        today = date.today()
        if self.date_of_joining > today:
            raise ValueError("date_of_joining cannot be in the future")
        if self.date_of_joining.year < 1970:
            raise ValueError("date_of_joining must be in 1970 or later")
        years_since_joining = (today - self.date_of_joining).days / 365.25
        if self.years_with_company > years_since_joining + 1:
            raise ValueError(
                f"years_with_company ({self.years_with_company}) cannot be more than the time since "
                f"date_of_joining (about {years_since_joining:.1f} years)"
            )
        return self


class KeyFactor(BaseModel):
    feature: str
    label: str
    value: str
    typical_value: str
    effect: Literal["increases risk", "reduces risk"]
    impact: float = Field(..., description="Contribution to the probability of leaving, compared with a typical employee (Shapley value)")
    suggested_action: str | None = None


class PredictionResponse(BaseModel):
    prediction: Literal["Likely to leave", "Likely to stay"]
    will_leave: bool
    probability_of_leaving: float
    typical_employee_probability: float
    risk_level: Literal["High", "Low"]
    risk_explanation: str
    decision_threshold: float
    key_factors: list[KeyFactor]
    warnings: list[str]
