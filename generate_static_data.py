"""
Generate static prediction data for GitHub Pages deployment.
Pre-computes predictions for all LSAT/GPA combinations.
"""

import pickle
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "/Users/tygribble/Desktop/Law_Data/law_school_predictor/models"
OUTPUT_DIR = "/Users/tygribble/Desktop/Law_Data/law_school_predictor/docs"

print("Loading models...")

# Load models
with open(os.path.join(MODEL_DIR, 'non_urm_baseline.pkl'), 'rb') as f:
    non_urm_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'urm_baseline.pkl'), 'rb') as f:
    urm_model = pickle.load(f)

# Load selectivity rankings
selectivity_rankings = pd.read_csv(os.path.join(MODEL_DIR, 'school_selectivity.csv'), index_col=0)
schools = selectivity_rankings.index.tolist()

print(f"Loaded {len(schools)} schools")

# Feature columns (must match training order)
FEATURE_COLS = ['lsat', 'gpa', 'app_timing_days', 'has_work_experience', 'softs_numeric', 'school_selectivity', 'cycle_year']

# Default values
DEFAULT_TIMING = 90
DEFAULT_WORK_EXP = 0
DEFAULT_SOFTS = 2
DEFAULT_CYCLE = 2025

# Generate predictions using vectorized operations for speed
lsat_range = list(range(140, 181))
gpa_range = [round(x * 0.1, 1) for x in range(25, 41)]

print(f"Generating predictions for {len(lsat_range)} LSAT x {len(gpa_range)} GPA x {len(schools)} schools...")

# Create all combinations at once
rows = []
for lsat in lsat_range:
    for gpa in gpa_range:
        for school in schools:
            rows.append({
                'lsat': lsat,
                'gpa': gpa,
                'app_timing_days': DEFAULT_TIMING,
                'has_work_experience': DEFAULT_WORK_EXP,
                'softs_numeric': DEFAULT_SOFTS,
                'school_selectivity': selectivity_rankings.loc[school, 'selectivity'],
                'cycle_year': DEFAULT_CYCLE,
                'school': school
            })

print(f"Created {len(rows):,} prediction combinations")

df = pd.DataFrame(rows)
X = df[FEATURE_COLS]

# Predict all at once
print("Running Non-URM predictions...")
probs_non_urm = non_urm_model.predict_proba(X)[:, 1]

print("Running URM predictions...")
probs_urm = urm_model.predict_proba(X)[:, 1]

df['prob_non_urm'] = (probs_non_urm * 100).round(1)
df['prob_urm'] = (probs_urm * 100).round(1)

print("Organizing results...")

# Organize into nested structure
predictions_non_urm = {}
predictions_urm = {}

for lsat in lsat_range:
    predictions_non_urm[lsat] = {}
    predictions_urm[lsat] = {}

    for gpa in gpa_range:
        gpa_key = str(gpa)
        predictions_non_urm[lsat][gpa_key] = {}
        predictions_urm[lsat][gpa_key] = {}

# Fill in predictions
for _, row in df.iterrows():
    lsat = int(row['lsat'])
    gpa_key = str(row['gpa'])
    school = row['school']

    predictions_non_urm[lsat][gpa_key][school] = row['prob_non_urm']
    predictions_urm[lsat][gpa_key][school] = row['prob_urm']

print("Saving files...")

# Save as JSON files
with open(os.path.join(OUTPUT_DIR, 'predictions_non_urm.json'), 'w') as f:
    json.dump(predictions_non_urm, f)

with open(os.path.join(OUTPUT_DIR, 'predictions_urm.json'), 'w') as f:
    json.dump(predictions_urm, f)

# Save school list with selectivity
schools_data = []
for school in schools:
    schools_data.append({
        'name': school,
        'selectivity': round(float(selectivity_rankings.loc[school, 'selectivity']), 1)
    })

schools_data.sort(key=lambda x: x['selectivity'], reverse=True)

with open(os.path.join(OUTPUT_DIR, 'schools.json'), 'w') as f:
    json.dump(schools_data, f)

print(f"\nFiles saved to {OUTPUT_DIR}:")
for fname in ['predictions_non_urm.json', 'predictions_urm.json', 'schools.json']:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname}: {size/1024/1024:.2f} MB")

print("\nDone!")
