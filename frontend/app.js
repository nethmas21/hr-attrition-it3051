// Same origin when served by the backend; fall back to the local server if index.html is opened from disk.
const API_BASE = window.location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";

const form = document.getElementById("employee-form");
const submitBtn = document.getElementById("submit-btn");

// Input rules - kept identical to the backend limits in backend/app/schemas.py
const RULES = {
  job_satisfaction:   { label: "Job satisfaction", type: "int", min: 1, max: 5 },
  work_life_balance:  { label: "Work-life balance", type: "int", min: 1, max: 4 },
  appraisal_rating:   { label: "Appraisal rating", type: "int", min: 1, max: 5 },
  does_overtime:      { label: "Overtime", type: "choice", choices: ["Yes", "No"] },
  leaves_taken:       { label: "Leave days", type: "int", min: 0, max: 365 },
  distance_from_home: { label: "Distance from home", type: "number", min: 0, max: 500 },
  date_of_joining:    { label: "Date of joining", type: "date" },
  years_with_company: { label: "Years with the company", type: "int", min: 0, max: 50 },
  previous_companies: { label: "Previous companies", type: "int", min: 0, max: 30 },
  monthly_salary:     { label: "Monthly salary", type: "number", min: 1, max: 1000000 },
};

const EXAMPLES = {
  high: {
    job_satisfaction: 1, work_life_balance: 1, appraisal_rating: 1, does_overtime: "Yes",
    leaves_taken: 20, distance_from_home: 35, date_of_joining: "2021-06-01",
    years_with_company: 5, previous_companies: 4, monthly_salary: 42000,
  },
  low: {
    job_satisfaction: 5, work_life_balance: 4, appraisal_rating: 5, does_overtime: "No",
    leaves_taken: 10, distance_from_home: 8, date_of_joining: "2018-02-12",
    years_with_company: 8, previous_companies: 1, monthly_salary: 85000,
  },
};

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const percent = (p) => `${Math.round(p * 100)}%`;
const todayISO = () => new Date().toISOString().slice(0, 10);

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setFieldError(name, message) {
  const el = form.querySelector(`.error[data-for="${name}"]`);
  if (el) el.textContent = message || "";
  const field = el ? el.closest(".field") : null;
  if (field) field.classList.toggle("invalid", Boolean(message));
}

function clearErrors() {
  form.querySelectorAll(".error").forEach((el) => (el.textContent = ""));
  form.querySelectorAll(".field.invalid").forEach((el) => el.classList.remove("invalid"));
  $("result-error").hidden = true;
}

// ------------------------------------------------------------------ read + validate the form
function readForm() {
  const data = new FormData(form);
  const values = {};
  const errors = {};

  for (const [name, rule] of Object.entries(RULES)) {
    const raw = (data.get(name) ?? "").toString().trim();
    if (raw === "") { errors[name] = "This field is required"; continue; }

    if (rule.type === "choice") {
      if (!rule.choices.includes(raw)) errors[name] = `Choose ${rule.choices.join(" or ")}`;
      else values[name] = raw;
    } else if (rule.type === "date") {
      if (raw > todayISO()) errors[name] = "Date of joining cannot be in the future";
      else if (raw < "1970-01-01") errors[name] = "Date of joining must be in 1970 or later";
      else values[name] = raw;
    } else {
      const num = Number(raw);
      if (!Number.isFinite(num)) errors[name] = "Enter a number";
      else if (rule.type === "int" && !Number.isInteger(num)) errors[name] = "Enter a whole number";
      else if (num < rule.min || num > rule.max) errors[name] = `Must be between ${rule.min.toLocaleString()} and ${rule.max.toLocaleString()}`;
      else values[name] = num;
    }
  }

  // Years with the company cannot be more than the time since joining (+1 year tolerance)
  if (values.date_of_joining && values.years_with_company !== undefined) {
    const yearsSince = (Date.now() - new Date(values.date_of_joining).getTime()) / (365.25 * 24 * 3600 * 1000);
    if (values.years_with_company > yearsSince + 1) {
      errors.years_with_company = `Cannot be more than the time since joining (about ${yearsSince.toFixed(1)} years)`;
    }
  }
  return { values, errors };
}

// ------------------------------------------------------------------ render the result
function renderResult(r) {
  $("result-empty").hidden = true;
  $("result").hidden = false;

  const level = r.risk_level.toLowerCase();
  $("risk-banner").className = `risk-banner ${level}`;
  $("risk-badge").textContent = `${r.risk_level} risk`;
  $("prediction-text").textContent = r.prediction;
  $("probability-text").textContent = percent(r.probability_of_leaving);
  $("risk-explanation").textContent = r.risk_explanation;
  $("context-note").textContent =
    `For comparison, ${percent(r.company_attrition_rate)} of all employees in the data left the company. ` +
    `Employees are flagged high risk from ${percent(r.decision_threshold)}.`;

  const fill = $("meter-fill");
  fill.className = `meter-fill ${level}`;
  fill.style.width = percent(r.probability_of_leaving);
  $("meter-threshold").style.left = `${r.decision_threshold * 100}%`;
  const thresholdLabel = $("meter-threshold-label");
  thresholdLabel.style.left = `${r.decision_threshold * 100}%`;
  thresholdLabel.textContent = `${percent(r.decision_threshold)} high-risk threshold`;

  const warnings = $("warnings");
  warnings.hidden = r.warnings.length === 0;
  warnings.innerHTML = r.warnings.length
    ? `<strong>Please note:</strong><ul>${r.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
    : "";

  $("factors-intro").textContent =
    `A typical employee has a ${percent(r.typical_employee_probability)} chance of leaving. ` +
    `Each detail below shows how many percentage points (pts) it adds or removes for this employee ` +
    `(only details worth 2 pts or more are listed).`;

  const factors = $("factors");
  factors.innerHTML = r.key_factors.length
    ? r.key_factors.map((f) => {
        const up = f.effect === "increases risk";
        const points = Math.round(Math.abs(f.impact) * 100);
        return `<li>
          <span class="factor-icon ${up ? "up" : "down"}" aria-hidden="true">${up ? "&uarr;" : "&darr;"}</span>
          <div>
            <div class="factor-name">${escapeHtml(f.label)}: ${escapeHtml(f.value)}</div>
            <div class="factor-detail">${up ? "Increases" : "Reduces"} risk (typical employee: ${escapeHtml(f.typical_value)})</div>
          </div>
          <span class="factor-impact ${up ? "up" : "down"}">${up ? "+" : "&minus;"}${points} pts</span>
        </li>`;
      }).join("")
    : `<li class="no-factors">No single detail stands out: this employee's chance of leaving is close to that of a typical employee.</li>`;

  const actions = [...new Set(r.key_factors.map((f) => f.suggested_action).filter(Boolean))];
  $("actions-block").hidden = actions.length === 0;
  $("actions-list").innerHTML = actions.map((a) => `<li>${escapeHtml(a)}</li>`).join("");
}

function showError(message) {
  const box = $("result-error");
  box.textContent = message;
  box.hidden = false;
}

// ------------------------------------------------------------------ events
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();

  const { values, errors } = readForm();
  if (Object.keys(errors).length) {
    Object.entries(errors).forEach(([name, msg]) => setFieldError(name, msg));
    form.querySelector(".field.invalid input, .field.invalid select")?.focus();
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Checking...";
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const body = await response.json();

    if (response.status === 422) {
      // Server-side validation: show each message next to its field
      body.errors.forEach((e) => setFieldError(RULES[e.field] ? e.field : "general", e.message));
      return;
    }
    if (!response.ok) throw new Error(body.detail || `Server error (${response.status})`);
    renderResult(body);
  } catch (err) {
    showError(`Could not get a prediction. Make sure the backend server is running. (${err.message})`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check attrition risk";
  }
});

form.addEventListener("reset", () => {
  clearErrors();
  $("result").hidden = true;
  $("result-empty").hidden = false;
});

document.querySelectorAll("[data-example]").forEach((btn) =>
  btn.addEventListener("click", () => {
    clearErrors();
    const example = EXAMPLES[btn.dataset.example];
    for (const [name, value] of Object.entries(example)) {
      const inputs = form.elements[name];
      if (inputs instanceof RadioNodeList) {
        inputs.forEach((radio) => (radio.checked = radio.value === value));
      } else {
        inputs.value = value;
      }
    }
  })
);

$("date_of_joining").max = todayISO();

// ------------------------------------------------------------------ model information in the footer
fetch(`${API_BASE}/api/model-info`)
  .then((r) => r.json())
  .then((info) => {
    const m = info.test_metrics;
    $("model-info").textContent =
      `Model: ${info.model_name}. Tested on ${m.test_size.toLocaleString()} past employees: ` +
      `accuracy ${percent(m.accuracy)}, precision ${percent(m.precision)}, recall ${percent(m.recall)}, ` +
      `F1-score ${m.f1.toFixed(2)}, ROC-AUC ${m.roc_auc.toFixed(2)}.`;
  })
  .catch(() => {
    $("model-info").textContent = "Model information unavailable - is the backend server running?";
  });
