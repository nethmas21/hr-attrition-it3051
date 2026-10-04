"""Employee Attrition Prediction API.

Run from the project root:
    uvicorn backend.app.main:app --reload
Then open http://127.0.0.1:8000 (prediction page) or http://127.0.0.1:8000/docs (API documentation).
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .model_service import ModelService
from .schemas import EmployeeInput, PredictionResponse

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at start-up; the server will not start if the model files are missing or inconsistent.
    app.state.model_service = ModelService.load()
    yield


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts whether an employee is likely to leave, using the final tuned Random Forest model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allows the frontend to call the API even if index.html is opened directly from disk.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return invalid / missing inputs as a simple list the frontend can show next to each field."""
    errors = []
    for error in exc.errors():
        location = [str(part) for part in error["loc"] if part != "body"]
        field = location[0] if location else "general"
        message = error["msg"].removeprefix("Value error, ")
        if error["type"] == "missing":
            message = "This field is required"
        elif error["type"] == "extra_forbidden":
            message = "Unknown field"
        errors.append({"field": field, "message": message})
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info(request: Request):
    return request.app.state.model_service.model_info()


@app.post("/api/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput, request: Request):
    return request.app.state.model_service.predict(employee)


# Serve the frontend (index.html, styles.css, app.js) at the site root. Registered last so /api routes win.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
