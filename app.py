"""
IPL Player Selection Prediction API
Flask backend that loads a trained ML model and serves predictions.
"""

import os
import traceback
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)

# Set to False (or via env var) to silence debug prints in production
DEBUG_MODE = os.environ.get("DEBUG_MODE", "1").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Load model and scaler
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ipl_model_fs.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler_fs.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Model loaded successfully from {MODEL_PATH}")
    print(f"[OK] Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    model = None
    scaler = None
    print(f"[FAIL] Failed to load model or scaler: {e}")

# ---------------------------------------------------------------------------
# Feature order (must match training data)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'Consistency_Score',
    'Recent_Form',
    'Strike_Rate',
    'Batting_Average',
    'Runs',
    'Impact_Score',
    'Experience'
]

# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------
def get_top_reasons(features_dict):
    """
    Identify the top 3 strongest contributing features based on value thresholds.
    Uses domain-aware heuristic rules to generate human-readable explanations.
    """
    reasons = []

    # Strike Rate analysis
    sr = features_dict.get('strike_rate', 0)
    if sr >= 145:
        reasons.append(("High Strike Rate — boosts T20 scoring impact", 0.95))
    elif sr >= 130:
        reasons.append(("Good Strike Rate — supports aggressive batting", 0.70))
    elif sr < 100:
        reasons.append(("Low Strike Rate — limits run-scoring ability", -0.60))

    # Consistency Score analysis
    cs = features_dict.get('consistency', 0)
    if cs >= 0.75:
        reasons.append(("Strong Consistency — reliable match-to-match output", 0.90))
    elif cs >= 0.5:
        reasons.append(("Moderate Consistency — some variation in performance", 0.50))
    elif cs < 0.3:
        reasons.append(("Poor Consistency — unpredictable performances", -0.50))

    # Recent Form analysis
    rf = features_dict.get('recent_form', 0)
    if rf >= 0.6:
        reasons.append(("Excellent Recent Form — strong momentum heading in", 0.90))
    elif rf >= 0.35:
        reasons.append(("Good Recent Form — showing promising match returns", 0.60))
    elif rf < 0.15:
        reasons.append(("Poor Recent Form — declining recent match output", -0.55))

    # Batting Average analysis
    ba = features_dict.get('batting_avg', 0)
    if ba >= 40:
        reasons.append(("High Batting Average — elite run accumulation", 0.85))
    elif ba >= 28:
        reasons.append(("Solid Batting Average — dependable contributor", 0.55))
    elif ba < 18:
        reasons.append(("Low Batting Average — struggles to anchor innings", -0.50))

    # Runs analysis
    runs = features_dict.get('runs', 0)
    if runs >= 4000:
        reasons.append(("High Run Scorer — proven IPL-level volume", 0.85))
    elif runs >= 1500:
        reasons.append(("Decent Run Scorer — solid career accumulation", 0.50))
    elif runs < 300:
        reasons.append(("Low Career Runs — limited run output", -0.45))

    # Experience analysis
    matches = features_dict.get('matches', 0)
    if matches >= 100:
        reasons.append(("Highly Experienced — pressure-tested at top level", 0.75))
    elif matches >= 50:
        reasons.append(("Moderate Experience — growing match exposure", 0.40))
    elif matches < 15:
        reasons.append(("Limited Experience — unproven at IPL scale", -0.40))

    # Impact Score analysis
    impact = features_dict.get('impact_score', 0)
    if impact >= 5000:
        reasons.append(("Very High Impact — game-changing ability", 0.80))
    elif impact >= 2000:
        reasons.append(("Good Impact Score — meaningful match contributions", 0.50))

    # Sort by absolute contribution strength and pick top 3
    reasons.sort(key=lambda x: abs(x[1]), reverse=True)
    return [r[0] for r in reasons[:3]]



def generate_summary(tier, confidence, reasons):
    """
    Build a concise, human-readable prediction summary sentence.
    """
    # Pick only the short label (before the em-dash) for readability
    short_reasons = [r.split(' — ')[0].lower() for r in reasons[:2]]

    if tier == "Elite":
        base = "This player shows elite potential"
    elif tier == "Core":
        base = "This player shows competitive potential"
    elif tier == "Fringe":
        base = "This player is on the fringe of selection"
    else:
        base = "This player currently falls below selection thresholds"

    if short_reasons:
        base += f" driven by {' and '.join(short_reasons)}"

    base += f" (model confidence: {confidence.lower()})."
    return base


def check_edge_case_warnings(features_dict):
    """
    Detect unusual / contradictory stat combinations and return warning messages.
    """
    warnings = []

    ba = features_dict.get('batting_avg', 0)
    sr = features_dict.get('strike_rate', 0)
    runs = features_dict.get('runs', 0)
    matches = features_dict.get('matches', 0)
    consistency = features_dict.get('consistency', 0)
    recent_form = features_dict.get('recent_form', 0)

    # High average but very low strike rate
    if ba > 70 and sr < 100:
        warnings.append("Unusual stat combination: very high batting average with low strike rate")

    # Very high runs but very few matches
    if runs > 3000 and matches < 20:
        warnings.append("Unusual stat combination: very high runs with very few matches played")

    # Perfect consistency with poor recent form
    if consistency > 0.9 and recent_form < 0.1:
        warnings.append("Unusual stat combination: high consistency but very poor recent form")

    # Zero runs but high strike rate
    if runs == 0 and sr > 100:
        warnings.append("Unusual stat combination: zero runs but non-trivial strike rate")

    # Very high strike rate with very low batting average
    if sr > 200 and ba < 15:
        warnings.append("Unusual stat combination: extremely high strike rate with very low average")

    return "; ".join(warnings) if warnings else None


# ---------------------------------------------------------------------------
# Role Suitability helpers
# ---------------------------------------------------------------------------
def compute_role_scores(data):
    """
    Compute role-specific suitability scores based on player stats.
    Returns dict with batsman_score, bowler_score, allrounder_score, best_role.
    """
    runs = float(data.get('runs', 0))
    batting_avg = float(data.get('batting_avg', 0))
    strike_rate = float(data.get('strike_rate', 0))
    wickets = float(data.get('wickets', 0))
    bowling_avg = float(data.get('bowling_avg', 50))
    economy = float(data.get('economy', 8))

    # --- Batsman Score ---
    batsman_score = (
        (runs / 7000) * 0.4 +
        (batting_avg / 50) * 0.3 +
        (strike_rate / 150) * 0.3
    )

    # --- Bowler Score (safe against negative components) ---
    bowling_avg_component = max(0, (50 - bowling_avg) / 50)
    economy_component = max(0, (10 - economy) / 10)
    bowler_score = (
        (wickets / 100) * 0.5 +
        bowling_avg_component * 0.3 +
        economy_component * 0.2
    )

    # --- Allrounder Score (strict: limited by weakest dimension) ---
    allrounder_score = min(batsman_score, bowler_score)

    # Clamp all to [0, 1]
    batsman_score = max(0, min(batsman_score, 1))
    bowler_score = max(0, min(bowler_score, 1))
    allrounder_score = max(0, min(allrounder_score, 1))

    # Determine best role
    scores = {
        'Batsman': batsman_score,
        'Bowler': bowler_score,
        'Allrounder': allrounder_score,
    }
    best_role = max(scores, key=scores.get)

    return {
        'batsman_score': round(batsman_score, 3),
        'bowler_score': round(bowler_score, 3),
        'allrounder_score': round(allrounder_score, 3),
        'best_role': best_role,
    }


def adjust_probability_for_role(prob, selected_role, scores):
    """
    Blend ML prediction (75 %) with role-suitability score (25 %) so
    that role mismatch moderates the final probability without
    dominating the ML model's output.

    Formula:  adjusted = base * 0.75 + role_score_pct * 0.25

    Returns (adjusted_prob, penalty_applied).
    """
    # Pick the role score that corresponds to the selected role
    if selected_role == 'Bowler':
        role_score = scores['bowler_score']
    elif selected_role == 'Allrounder':
        role_score = scores['allrounder_score']
    elif selected_role in ('Batsman', 'Wicketkeeper'):
        role_score = scores['batsman_score']
    else:
        role_score = 1.0  # unknown role — no adjustment

    role_score_pct = role_score * 100  # convert 0-1 → 0-100

    adjusted = prob * 0.75 + role_score_pct * 0.25
    adjusted = max(0, min(adjusted, 100))

    # Flag when the adjustment shifts the probability by more than 1 %
    penalty_applied = bool(abs(prob - adjusted) > 1.0)

    return round(adjusted, 2), penalty_applied


INJURY_PENALTIES = {
    'Minor':    {'factor': 0.85, 'note': 'Minor injury slightly reduces selection probability due to fitness concerns.'},
    'Moderate': {'factor': 0.70, 'note': 'Moderate injury significantly reduces selection probability — recovery timeline uncertain.'},
    'Severe':   {'factor': 0.50, 'note': 'Severe injury drastically reduces selection probability — player fitness is a major risk.'},
}

def apply_injury_reduction(prob, injury_status):
    """
    Reduce probability based on injury severity.
    Applied AFTER role moderation in the pipeline:
        Base ML → Role Moderation → Injury Reduction → Final Output

    injury_status: 'Healthy', 'Minor', 'Moderate', or 'Severe'
    Returns (adjusted_prob, is_injured, injury_note, injury_severity).
    """
    if injury_status in INJURY_PENALTIES:
        penalty = INJURY_PENALTIES[injury_status]
        adjusted = prob * penalty['factor']
        adjusted = max(0, min(adjusted, 100))
        return round(adjusted, 2), True, penalty['note'], injury_status

    return prob, False, None, None


def generate_role_explanation(selected_role, best_role, scores):
    """
    Generate a human-readable explanation about role suitability.
    """
    bat = scores['batsman_score']
    bowl = scores['bowler_score']

    if best_role == 'Batsman':
        strength = "strong batting metrics"
        weakness = "limited bowling contribution" if bowl < 0.4 else "moderate bowling ability"
    elif best_role == 'Bowler':
        strength = "strong bowling figures"
        weakness = "limited batting output" if bat < 0.4 else "moderate batting ability"
    else:
        strength = "balanced batting and bowling contributions"
        weakness = None

    explanation = f"This player is best suited as a {best_role} due to {strength}"
    if weakness:
        explanation += f" and {weakness}"
    explanation += "."

    if selected_role != best_role and selected_role != 'Wicketkeeper':
        explanation += f" The selected role ({selected_role}) differs from the player's primary strength, though strong overall stats remain the dominant factor."
    elif selected_role == 'Wicketkeeper' and best_role != 'Batsman':
        explanation += f" Wicketkeeper role relies on batting strength which is not this player's primary skill."

    return explanation


def compute_dynamic_features(last_5_runs):
    """
    Compute recent_form and consistency_score from last 5 match runs.
    This is computed server-side for architectural integrity.

    Returns (recent_form, consistency_score).
    """
    runs_arr = np.array(last_5_runs, dtype=float)
    mean = np.mean(runs_arr)
    std = np.std(runs_arr)

    # Recent Form: average normalized to 0-1 scale
    recent_form = float(np.clip(mean / 100.0, 0, 1))

    # Consistency Score: 1 - (std / mean), clamped to [0, 1]
    if mean > 0:
        consistency = float(np.clip(1.0 - (std / mean), 0, 1))
    else:
        consistency = 0.0

    return round(recent_form, 4), round(consistency, 4)


def validate_inputs(data):
    """
    Validate input fields and return a list of error messages.
    Returns empty list if all inputs are valid.
    """
    errors = []

    # Required field presence
    REQUIRED = ['runs', 'strike_rate', 'batting_avg', 'matches', 'last_5_match_runs']
    missing = [f for f in REQUIRED if f not in data]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
        return errors  # Can't validate values if fields are missing

    try:
        runs = float(data['runs'])
        strike_rate = float(data['strike_rate'])
        batting_avg = float(data['batting_avg'])
        matches = float(data['matches'])
    except (ValueError, TypeError) as e:
        errors.append(f"All inputs must be valid numbers: {e}")
        return errors

    # Validate last_5_match_runs
    last5 = data.get('last_5_match_runs', [])
    if not isinstance(last5, list) or len(last5) != 5:
        errors.append("last_5_match_runs must be a list of exactly 5 numbers")
    else:
        try:
            for v in last5:
                fv = float(v)
                if fv < 0:
                    errors.append("last_5_match_runs values must be non-negative")
                    break
        except (ValueError, TypeError):
            errors.append("last_5_match_runs must contain valid numbers")

    # Runs, matches, strike_rate must be non-negative
    if runs < 0:
        errors.append("Runs must be a positive number")
    if matches < 0:
        errors.append("Matches must be a positive number")
    if strike_rate < 0:
        errors.append("Strike rate must be a positive number")
    if batting_avg < 0:
        errors.append("Batting average must be a positive number")

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict IPL selection probability.

    Expects JSON body with features: runs, strike_rate, batting_avg, matches,
    last_5_match_runs.  Optionally: role, wickets, bowling_avg, economy,
    injury_status for post-processing layers.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # ── Input Validation Guardrails ─────────────────────────────
        validation_errors = validate_inputs(data)
        if validation_errors:
            return jsonify({"error": "Invalid input values", "details": validation_errors}), 400

        # ── Extract core model inputs ───────────────────────────────
        runs = float(data['runs'])
        strike_rate = float(data['strike_rate'])
        batting_avg = float(data['batting_avg'])
        matches = float(data['matches'])

        # ── Compute dynamic features from last 5 match runs ────────
        last_5_runs = [float(v) for v in data['last_5_match_runs']]
        recent_form, consistency = compute_dynamic_features(last_5_runs)

        # ── Extract role-suitability inputs (optional) ──────────────
        selected_role = data.get('role', 'Allrounder')
        wickets = float(data.get('wickets', 0))
        bowling_avg = float(data.get('bowling_avg', 50))
        economy = float(data.get('economy', 8))

        # ── Extract injury status ──────────────────────────────────
        injury_status = data.get('injury_status', 'Healthy')

        # ── Compute derived features for model ─────────────────────
        impact_score = (runs * strike_rate) / 100.0
        experience = matches / 100.0

        # ── Create feature array in EXACT order matching training ───
        features = [
            consistency,
            recent_form,
            strike_rate,
            batting_avg,
            runs,
            impact_score,
            experience
        ]

        # ── Scale and predict ──────────────────────────────────────
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prob = model.predict_proba(features_scaled)[0][1] * 100
        prob = max(0, min(prob, 100))

        # ── Save base probability BEFORE role adjustment ───────────
        base_probability = round(prob, 2)

        # ── Role Suitability Analysis ──────────────────────────────
        role_data = {
            'runs': runs,
            'batting_avg': batting_avg,
            'strike_rate': strike_rate,
            'wickets': wickets,
            'bowling_avg': bowling_avg,
            'economy': economy,
        }
        role_scores = compute_role_scores(role_data)
        prob, penalty_applied = adjust_probability_for_role(
            prob, selected_role, role_scores
        )
        role_explanation = generate_role_explanation(
            selected_role, role_scores['best_role'], role_scores
        )

        # ── Injury Reduction (applied AFTER role moderation) ───────
        prob, is_injured, injury_note, injury_severity = apply_injury_reduction(
            prob, injury_status
        )

        # ── Tier Classification (from ADJUSTED probability) ────────
        if prob >= 90:
            prediction = 1
            tier = "Elite"
            confidence = "High"
            confidence_label = "High Chance of IPL Selection"
        elif prob >= 75:
            prediction = 1
            tier = "Core"
            confidence = "Medium"
            confidence_label = "Moderate Chance"
        elif prob >= 60:
            prediction = 0
            tier = "Fringe"
            confidence = "Medium"
            confidence_label = "Fringe Candidate"
        else:
            prediction = 0
            tier = "Backup"
            confidence = "Low"
            confidence_label = "Low Chance"

        # ── Explainability: Top Reasons ─────────────────────────────
        features_dict = {
            'runs': runs,
            'strike_rate': strike_rate,
            'batting_avg': batting_avg,
            'matches': matches,
            'consistency': consistency,
            'recent_form': recent_form,
            'impact_score': impact_score,
            'experience': experience
        }
        reasons = get_top_reasons(features_dict)

        # ── Edge Case Warnings ──────────────────────────────────────
        warning = check_edge_case_warnings(features_dict)

        # ── Prediction Summary ──────────────────────────────────────
        summary = generate_summary(tier, confidence, reasons)

        # ── Debug Logging ──────────────────────────────────────────
        if DEBUG_MODE:
            print(f"[DEBUG] Input Features: {features}")
            print(f"[DEBUG] Recent Form: {recent_form:.4f} | Consistency: {consistency:.4f}")
            print(f"[DEBUG] Base Probability: {base_probability:.2f}%")
            print(f"[DEBUG] Adjusted Probability: {prob:.2f}%")
            print(f"[DEBUG] Role: {selected_role} | Best: {role_scores['best_role']}")
            print(f"[DEBUG] Role Scores: bat={role_scores['batsman_score']}, bowl={role_scores['bowler_score']}, all={role_scores['allrounder_score']}")
            print(f"[DEBUG] Penalty Applied: {penalty_applied}")
            print(f"[DEBUG] Injured: {is_injured} (Severity: {injury_severity})")
            print(f"[DEBUG] Tier: {tier} | Confidence: {confidence}")
            print(f"[DEBUG] Reasons: {reasons}")
            if warning:
                print(f"[DEBUG] Warning: {warning}")

        # ── Response ───────────────────────────────────────────────
        response = {
            "prediction": prediction,
            "probability": round(prob, 2),
            "base_probability": base_probability,
            "confidence_label": confidence_label,
            "confidence": confidence,
            "tier": tier,
            "reasons": reasons,
            "summary": summary,
            "recent_form": round(recent_form, 4),
            "consistency_score": round(consistency, 4),
            "role_analysis": {
                "batsman_score": role_scores['batsman_score'],
                "bowler_score": role_scores['bowler_score'],
                "allrounder_score": role_scores['allrounder_score'],
                "best_role": role_scores['best_role'],
                "selected_role": selected_role,
                "role_match": role_scores['best_role'] == selected_role or
                              (selected_role == 'Wicketkeeper' and role_scores['best_role'] == 'Batsman'),
                "penalty_applied": penalty_applied,
                "explanation": role_explanation,
            },
        }

        # Only include warning if present
        if warning:
            response["warning"] = warning

        # Include injury info if injured
        if injury_note:
            response["injury_note"] = injury_note
            response["injury_severity"] = injury_severity

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/feature-names", methods=["GET"])
def feature_names():
    """Return the ordered list of feature names."""
    return jsonify({"features": FEATURE_NAMES})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
