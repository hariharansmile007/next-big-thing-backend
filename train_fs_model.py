"""
Train the feature-selected IPL model using only 7 key features.
Outputs: ipl_model_fs.joblib, scaler_fs.joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = r"C:\Users\Hariharan\OneDrive\IPL Player Stats - 2016 till 2019.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# 2. Clean & engineer features
# ---------------------------------------------------------------------------
# Convert columns to numeric where needed
df['Runds Scored'] = pd.to_numeric(df['Runds Scored'], errors='coerce').fillna(0)
df['Batting Strike Rate'] = pd.to_numeric(df['Batting Strike Rate'], errors='coerce').fillna(0)
df['Batting Average'] = pd.to_numeric(df['Batting Average'], errors='coerce').fillna(0)
df['Matches'] = pd.to_numeric(df['Matches'], errors='coerce').fillna(0)

# Map raw columns to model features
df['Runs'] = df['Runds Scored'].astype(float)
df['Strike_Rate'] = df['Batting Strike Rate'].astype(float)
df['Batting_Average'] = df['Batting Average'].astype(float)

# Derived features
df['Impact_Score'] = (df['Runs'] * df['Strike_Rate']) / 100.0
df['Experience'] = df['Matches'].astype(float) / 100.0

# Simulate Recent_Form and Consistency_Score from available data
# Recent_Form: normalized run contribution per match
df['Recent_Form'] = np.where(
    df['Matches'] > 0,
    (df['Runs'] / df['Matches']) / 100.0,
    0
)
# Consistency_Score: based on batting average relative to strike rate variance
np.random.seed(42)
noise = np.random.normal(0, 0.05, len(df))
df['Consistency_Score'] = np.clip(
    df['Batting_Average'] / (df['Batting_Average'].max() + 1) + noise,
    0, 1
)

# ---------------------------------------------------------------------------
# 3. Create target variable (Selected = 1 if the player is IPL-caliber)
# ---------------------------------------------------------------------------
# Heuristic: players with good batting avg, decent runs, and high strike rate
df['Selected'] = (
    (df['Batting_Average'] >= 25) &
    (df['Runs'] >= 100) &
    (df['Strike_Rate'] >= 110)
).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Selected distribution:\n{df['Selected'].value_counts()}")

# ---------------------------------------------------------------------------
# 4. Prepare feature matrix (EXACT order matching model spec)
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

X = df[FEATURE_NAMES].values
y = df['Selected'].values

# ---------------------------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 6. Train Random Forest
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Feature importances
for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
    print(f"  {name:25s} {imp:.4f}")

# ---------------------------------------------------------------------------
# 8. Save model and scaler
# ---------------------------------------------------------------------------
joblib.dump(model, "ipl_model_fs.joblib")
joblib.dump(scaler, "scaler_fs.joblib")

print("\n[OK] Saved ipl_model_fs.joblib")
print("[OK] Saved scaler_fs.joblib")
print(f"Model expects {model.n_features_in_} features in order: {FEATURE_NAMES}")
