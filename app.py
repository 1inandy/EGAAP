from __future__ import annotations

import csv
import io
import math
import os
import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "geomagnetic_model.pkl"

# Feature order must match how the model was trained (see train_model.py).
FEATURES = ["speed", "bt", "temperature", "bz_gsm", "density"]
FEATURE_CONFIG = {
    "speed": {
        "label": "Solar wind speed",
        "unit": "km/s",
        "default": 420,
        "min": 250,
        "max": 900,
        "step": 1,
        "hint": "Typical calm wind sits near 300-450 km/s.",
    },
    "bt": {
        "label": "Magnetic field Bt",
        "unit": "nT",
        "default": 8,
        "min": 0,
        "max": 45,
        "step": 0.1,
        "hint": "Higher total field gives storms more room to develop.",
    },
    "temperature": {
        "label": "Temperature",
        "unit": "K",
        "default": 90000,
        "min": 10000,
        "max": 1000000,
        "step": 1000,
        "hint": "Use the proton temperature reported with the solar wind.",
    },
    "bz_gsm": {
        "label": "Bz GSM",
        "unit": "nT",
        "default": -4,
        "min": -40,
        "max": 30,
        "step": 0.1,
        "hint": "Negative Bz usually matters most for storm growth.",
    },
    "density": {
        "label": "Proton density",
        "unit": "p/cm^3",
        "default": 6,
        "min": 0,
        "max": 80,
        "step": 0.1,
        "hint": "Higher density can sharpen the magnetosphere response.",
    },
}

SEVERITY_BANDS = [
    {
        "classification": "Quiet",
        "min": 0,
        "max": None,
        "range": "Dst >= 0 nT",
        "effects": "No meaningful geomagnetic disturbance expected.",
        "tone": "quiet",
    },
    {
        "classification": "Weak",
        "min": -20,
        "max": 0,
        "range": "-20 to 0 nT",
        "effects": "Minor fluctuations; routine monitoring is enough.",
        "tone": "weak",
    },
    {
        "classification": "Moderate",
        "min": -50,
        "max": -20,
        "range": "-50 to -20 nT",
        "effects": "Small radio, navigation, and satellite drag disturbances are possible.",
        "tone": "moderate",
    },
    {
        "classification": "Strong",
        "min": -100,
        "max": -50,
        "range": "-100 to -50 nT",
        "effects": "Possible satellite, GPS, radio, and power-grid effects.",
        "tone": "strong",
    },
    {
        "classification": "Severe",
        "min": -200,
        "max": -100,
        "range": "-200 to -100 nT",
        "effects": "Widespread disruptions possible; auroras may reach lower latitudes.",
        "tone": "severe",
    },
    {
        "classification": "Extreme",
        "min": None,
        "max": -200,
        "range": "Dst < -200 nT",
        "effects": "Major satellite, radio, and power-grid impacts are possible.",
        "tone": "extreme",
    },
]

SCENARIOS = [
    {
        "name": "Quiet background wind",
        "values": {"speed": 360, "bt": 5.5, "temperature": 65000, "bz_gsm": 1.5, "density": 3.5},
    },
    {
        "name": "Southward IMF watch",
        "values": {"speed": 470, "bt": 13, "temperature": 125000, "bz_gsm": -8, "density": 8},
    },
    {
        "name": "Fast stream",
        "values": {"speed": 650, "bt": 10, "temperature": 260000, "bz_gsm": -5, "density": 5},
    },
    {
        "name": "CME-like compression",
        "values": {"speed": 780, "bt": 24, "temperature": 420000, "bz_gsm": -18, "density": 22},
    },
]

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def feature_definitions() -> list[dict[str, Any]]:
    return [{"key": key, **FEATURE_CONFIG[key]} for key in FEATURES]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def generate_random_values() -> dict[str, Any]:
    scenario = random.choice(SCENARIOS)
    values: dict[str, float] = {}
    for key, baseline in scenario["values"].items():
        config = FEATURE_CONFIG[key]
        span = config["max"] - config["min"]
        jitter = random.uniform(-0.045 * span, 0.045 * span)
        value = clamp(baseline + jitter, config["min"], config["max"])
        values[key] = round(value, 1 if config["step"] < 1 else 0)
    return {**values, "scenario": scenario["name"]}


def normalize_header(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def canonicalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {normalize_header(key): value for key, value in row.items()}
    return {key: normalized.get(key) for key in FEATURES}


def validate_feature_payload(data: dict[str, Any]) -> dict[str, float]:
    missing = [key for key in FEATURES if data.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    values: dict[str, float] = {}
    invalid: list[str] = []
    for key in FEATURES:
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            invalid.append(key)
            continue
        if not math.isfinite(value):
            invalid.append(key)
            continue
        values[key] = value

    if invalid:
        raise ValueError(f"Numeric values required for: {', '.join(invalid)}")
    return values


def predict_storm(values: dict[str, float]) -> float:
    if model is None:
        raise RuntimeError("Model not loaded.")
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(frame)
    return float(prediction[0])


def interpret_prediction(prediction: float) -> dict[str, Any]:
    for band in SEVERITY_BANDS:
        lower = band["min"]
        upper = band["max"]
        above_lower = lower is None or prediction >= lower
        below_upper = upper is None or prediction < upper
        if above_lower and below_upper:
            return dict(band)
    return dict(SEVERITY_BANDS[-1])


def prediction_payload(values: dict[str, float]) -> dict[str, Any]:
    prediction = predict_storm(values)
    severity = interpret_prediction(prediction)
    return {
        "input": values,
        "prediction": round(prediction, 2),
        "classification": severity["classification"],
        "effects": severity["effects"],
        "severity": severity,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        app_config={
            "features": feature_definitions(),
            "severity_bands": SEVERITY_BANDS,
            "model_ready": model is not None,
        },
    )


@app.route("/metadata")
def metadata():
    return jsonify(
        {
            "features": feature_definitions(),
            "severity_bands": SEVERITY_BANDS,
            "model_ready": model is not None,
        }
    )


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py to create geomagnetic_model.pkl."}), 503

    if "file" not in request.files:
        data = request.get_json(silent=True) or {}
        try:
            values = validate_feature_payload(data)
            return jsonify(prediction_payload(values))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No CSV file selected."}), 400

    try:
        stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
    except UnicodeDecodeError:
        return jsonify({"error": "CSV must be UTF-8 encoded."}), 400

    csv_reader = csv.DictReader(stream)
    if not csv_reader.fieldnames:
        return jsonify({"error": "CSV is empty or missing a header row."}), 400

    normalized_headers = {normalize_header(header) for header in csv_reader.fieldnames}
    missing_columns = [key for key in FEATURES if key not in normalized_headers]
    if missing_columns:
        return jsonify({"error": "CSV is missing columns: " + ", ".join(missing_columns)}), 400

    results: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    total_rows = 0

    for line_number, row in enumerate(csv_reader, start=2):
        total_rows += 1
        try:
            values = validate_feature_payload(canonicalize_row(row))
            results.append(prediction_payload(values))
        except ValueError as exc:
            invalid_rows.append({"line": line_number, "error": str(exc)})

    if not results:
        return jsonify({"error": "No valid rows found.", "invalid_rows": invalid_rows[:25]}), 400

    return jsonify(
        {
            "results": results,
            "summary": {
                "total_rows": total_rows,
                "valid_rows": len(results),
                "invalid_rows": len(invalid_rows),
            },
            "invalid_rows": invalid_rows[:25],
        }
    )


@app.route("/randomize", methods=["GET"])
def randomize():
    return jsonify(generate_random_values())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
