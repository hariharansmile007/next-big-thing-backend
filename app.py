"""
IPL Player Selection Prediction API
Flask backend that loads a trained ML model and serves predictions.
"""

import os
import traceback
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded successfully from {MODEL_PATH}")
    print(f"     Model type: {type(model).__name__}")
except Exception as e:
    model = None
    print(f"[FAIL] Failed to load model: {e}")

# ---------------------------------------------------------------------------
# Feature order (must match training data)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Age", "Matches", "Innings", "Runs", "Batting_Average",
    "Strike_Rate", "Fifties", "Hundreds", "Wickets", "Bowling_Average",
    "Economy", "Catches", "Runouts", "Recent_Form", "Consistency_Score",
    "Injury_Status", "Role_Batsman", "Role_Bowler", "Role_Allrounder",
    "Role_Wicketkeeper",
]


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict IPL selection probability.

    Expects JSON body with all 20 features in the correct order.
    Returns: { prediction: 0|1, probability: float, confidence_label: str, warning?: str }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # Build feature vector in the correct order
        features = []
        missing = []
        for feat in FEATURE_NAMES:
            if feat not in data:
                missing.append(feat)
            else:
                features.append(float(data[feat]))

        if missing:
            return jsonify({
                "error": f"Missing features: {', '.join(missing)}"
            }), 400

        input_array = np.array(features).reshape(1, -1)

        # ---- Collect warnings ----
        warnings = []

        # ---- Input normalization (match training scale) ----
        # Recent_Form (index 13): UI sends 0-100, model expects 0-1
        if input_array[0][13] > 1:
            input_array[0][13] = input_array[0][13] / 100.0

        # Consistency_Score (index 14): UI sends 0-100, model expects 0-1
        if input_array[0][14] > 1:
            input_array[0][14] = input_array[0][14] / 100.0

        # ---- Sanity checks for extreme values ----
        age = input_array[0][0]
        if age > 45:
            warnings.append("Age outside realistic IPL range")
        if age < 15:
            warnings.append("Age below realistic professional range")

        strike_rate = input_array[0][5]
        if strike_rate > 250:
            warnings.append("Strike rate unusually high — verify input")

        batting_avg = input_array[0][4]
        if batting_avg > 100:
            warnings.append("Batting average unusually high — verify input")

        economy = input_array[0][10]
        if economy > 15:
            warnings.append("Economy rate unusually high — verify input")

        # ---- Probability-first approach ----
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_array)[0]
            probability = round(float(proba[1]) * 100, 2)
        else:
            raw_pred = int(model.predict(input_array)[0])
            probability = 100.0 if raw_pred == 1 else 0.0

        # ---- Stricter classification threshold (85%) ----
        prediction = 1 if probability >= 85 else 0

        # ---- Interpretation labels ----
        if probability >= 85:
            confidence_label = "High Chance of IPL Selection"
        elif probability >= 65:
            confidence_label = "Moderate Chance"
        else:
            confidence_label = "Low Chance"

        response = {
            "prediction": prediction,
            "probability": probability,
            "confidence_label": confidence_label,
        }

        # ---- Attach warnings if any ----
        if warnings:
            response["warning"] = " | ".join(warnings)

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/feature-names", methods=["GET"])
def feature_names():
    """Return the ordered list of feature names."""
    return jsonify({"features": FEATURE_NAMES})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
