"""
Final Validation Test Suite for IPL Player Selection Prediction System.

Tests 5 scenarios:
  1. Elite player (high stats)
  2. Weak player (low stats)
  3. Medium player (average stats)
  4. Edge-case input (unusual combinations)
  5. Invalid input (missing / out-of-range values)

Run with:  python test_validation.py
Requires the Flask server to be running on localhost:5000.
"""

import requests
import json
import sys
import io

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API = "http://localhost:5000"

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label, condition):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  {status}  {label}")
    return condition


# ── 0. Health Check ─────────────────────────────────────────
separator("0. Health Check")
try:
    r = requests.get(f"{API}/health", timeout=5)
    health = r.json()
    check("Server is reachable", r.status_code == 200)
    check("Model loaded", health.get("model_loaded") is True)
    check("Scaler loaded", health.get("scaler_loaded") is True)
except Exception as e:
    print(f"  {FAIL}  Cannot reach server: {e}")
    print(f"\n  !!  Start the Flask backend first:  python app.py\n")
    sys.exit(1)


# ── 1. Elite Player ────────────────────────────────────────
separator("1. Elite Player (Virat Kohli profile)")
payload = {
    "runs": 7263,
    "strike_rate": 131.97,
    "batting_avg": 37.25,
    "matches": 237,
    "consistency": 0.85,
    "recent_form": 0.72
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: prob={d.get('probability')}, tier={d.get('tier')}, confidence={d.get('confidence')}")
print(f"  Summary: {d.get('summary')}")
print(f"  Reasons: {d.get('reasons')}")
check("Status 200", r.status_code == 200)
check("Probability >= 85", d.get("probability", 0) >= 85)
check("Tier is Elite", d.get("tier") == "Elite")
check("Confidence is High", d.get("confidence") == "High")
check("Has reasons list", len(d.get("reasons", [])) > 0)
check("Has summary string", isinstance(d.get("summary"), str) and len(d["summary"]) > 10)
check("No warning", d.get("warning") is None)


# ── 2. Weak Player ─────────────────────────────────────────
separator("2. Weak Player (low stats)")
payload = {
    "runs": 150,
    "strike_rate": 95,
    "batting_avg": 12.5,
    "matches": 10,
    "consistency": 0.25,
    "recent_form": 0.08
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: prob={d.get('probability')}, tier={d.get('tier')}, confidence={d.get('confidence')}")
print(f"  Summary: {d.get('summary')}")
print(f"  Reasons: {d.get('reasons')}")
check("Status 200", r.status_code == 200)
check("Probability < 65", d.get("probability", 100) < 65)
check("Tier is Backup", d.get("tier") == "Backup")
check("Confidence is Low", d.get("confidence") == "Low")
check("Has reasons", len(d.get("reasons", [])) > 0)
check("Has summary", isinstance(d.get("summary"), str))


# ── 3. Medium Player ───────────────────────────────────────
separator("3. Medium Player (borderline stats)")
payload = {
    "runs": 800,
    "strike_rate": 118,
    "batting_avg": 24,
    "matches": 40,
    "consistency": 0.5,
    "recent_form": 0.35
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: prob={d.get('probability')}, tier={d.get('tier')}, confidence={d.get('confidence')}")
print(f"  Summary: {d.get('summary')}")
print(f"  Reasons: {d.get('reasons')}")
check("Status 200", r.status_code == 200)
prob_val = d.get("probability", 0)
check(f"Probability between 10-95 (got {prob_val})", 10 <= prob_val <= 95)
check("Tier is Core or Backup", d.get("tier") in ("Core", "Backup"))
check("Has reasons", len(d.get("reasons", [])) > 0)
check("Has summary", isinstance(d.get("summary"), str))


# ── 4. Edge Case Input ─────────────────────────────────────
separator("4. Edge Case (batting_avg=80, strike_rate=90)")
payload = {
    "runs": 5000,
    "strike_rate": 90,
    "batting_avg": 80,
    "matches": 15,
    "consistency": 0.95,
    "recent_form": 0.05
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: prob={d.get('probability')}, tier={d.get('tier')}")
print(f"  Warning: {d.get('warning')}")
check("Status 200", r.status_code == 200)
check("Warning is present", d.get("warning") is not None)


# ── 5. Invalid Input ───────────────────────────────────────
separator("5a. Invalid Input — consistency out of range")
payload = {
    "runs": 1000,
    "strike_rate": 120,
    "batting_avg": 25,
    "matches": 50,
    "consistency": 1.5,   # OUT OF RANGE
    "recent_form": 0.4
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: {d}")
check("Status 400", r.status_code == 400)
check("Has error message", "error" in d)

separator("5b. Invalid Input — missing fields")
payload = {
    "runs": 1000,
    "batting_avg": 25
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
print(f"  Response: {d}")
check("Status 400", r.status_code == 400)
check("Has error + details", "error" in d)


# ── 6. Data Flow Check ─────────────────────────────────────
separator("6. Data Flow — feature-names endpoint")
r = requests.get(f"{API}/feature-names")
d = r.json()
expected = ['Consistency_Score', 'Recent_Form', 'Strike_Rate', 'Batting_Average', 'Runs', 'Impact_Score', 'Experience']
check("Feature names match model training order", d.get("features") == expected)


# ── 7. Response Schema Check ───────────────────────────────
separator("7. Response Schema Completeness")
payload = {
    "runs": 4000, "strike_rate": 135, "batting_avg": 38,
    "matches": 120, "consistency": 0.8, "recent_form": 0.65
}
r = requests.post(f"{API}/predict", json=payload)
d = r.json()
REQUIRED_KEYS = ["prediction", "probability", "confidence_label", "confidence", "tier", "reasons", "summary", "recent_form", "consistency_score"]
missing = [k for k in REQUIRED_KEYS if k not in d]
print(f"  Keys present: {list(d.keys())}")
if missing:
    print(f"  Missing keys: {missing}")
check("All required keys present", len(missing) == 0)
check("Probability is clamped 0-100", 0 <= d.get("probability", -1) <= 100)
check("Prediction is 0 or 1", d.get("prediction") in (0, 1))


# ── Summary ────────────────────────────────────────────────
separator("FINAL RESULTS")
passed = sum(results)
total = len(results)
print(f"\n  {passed}/{total} checks passed\n")
if passed == total:
    print("  >>> ALL TESTS PASSED -- System is demo-ready!\n")
else:
    print(f"  WARNING: {total - passed} check(s) failed -- review above.\n")
