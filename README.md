# Employee Attrition Prediction (IT3051 Mini Project)

Predicts whether an employee is likely to leave the company, explains the main reasons, and suggests HR actions.

- **Dataset:** HR Attrition Indian Dataset (Kaggle, kmldas): 5,000 employee records, target `LeftCompany`.
- **Final model:** tuned Random Forest using 10 features. On the hidden test set (1,000 employees): accuracy 0.787, precision 0.790, recall 0.580, F1-score 0.669, ROC-AUC 0.743.

## Project structure

| Folder | Contents |
|---|---|
| `data/` | Raw dataset |
| `notebooks/` | `01_EDA`, `02_Preprocessing_FeatureEngineering`, `03_ModelDevelopment` (Stage 6), `04_ModelOptimization` (Stage 7) |
| `processed_data/` | Train/test splits created by notebook 02 |
| `models/` | Final model (`final_random_forest_model.pkl`), its feature list and `model_metadata.json` |
| `backend/` | FastAPI prediction service (Stage 9) and its tests |
| `frontend/` | HR web page (Stage 10): `index.html`, `styles.css`, `app.js` |
| `report/figures/` | Charts saved by the notebooks |

## Setup

Requires Python 3.12 or newer. From the project folder:

```bash
pip install -r requirements.txt
```

## Run the prediction system

```bash
python -m uvicorn backend.app.main:app --reload
```

Then open:

- **http://127.0.0.1:8000**: the HR prediction page
- **http://127.0.0.1:8000/docs**: interactive API documentation

The backend serves the frontend, so only one command is needed. Stop the server with `Ctrl + C`.

## How the system works

1. HR enters 10 employee details in the web page. The page checks the values (required fields, allowed ranges, joining date not in the future).
2. The page sends them to `POST /api/predict`. The backend checks them again and rejects invalid or missing values with a clear message for each field.
3. The backend applies the same preprocessing as notebook 02 (`Yes`/`No` → 1/0, joining date → joining year, columns in the model's order) and runs the saved model.
4. The response contains:
   - the probability of leaving and a **High / Low** risk level (decision threshold 0.5)
   - the **main factors**: how many percentage points each detail adds or removes compared with a typical employee (exact Shapley values)
   - **suggested HR actions** for the factors that increase risk
   - **warnings** if a value is outside the range seen in the training data

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check the server is running |
| `GET` | `/api/model-info` | Model name, features, test metrics |
| `POST` | `/api/predict` | Predict for one employee |

Example request body for `/api/predict`:

```json
{
  "job_satisfaction": 1, "work_life_balance": 1, "appraisal_rating": 1,
  "does_overtime": "Yes", "leaves_taken": 20, "distance_from_home": 35,
  "date_of_joining": "2021-06-01", "years_with_company": 5,
  "previous_companies": 4, "monthly_salary": 42000
}
```

## Tests

```bash
python -m pytest
```

The tests check that:

- rebuilding all 1,000 test-set employees from raw inputs reproduces the exact rows the model was evaluated on (same preprocessing)
- API predictions match the saved model
- invalid, missing and unknown inputs are rejected with clear messages
- out-of-range values produce warnings
- the explanations add up correctly

## Rebuilding the model

1. Run the notebooks in order (01 → 04). Notebook 04 saves the final model to `models/`.
2. Run `python backend/scripts/build_model_metadata.py` to refresh `models/model_metadata.json`. The backend refuses to start if this file does not match the model.
