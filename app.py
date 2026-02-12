"""
Law School Admissions Prediction - Flask Web Application (v2)
==============================================================
Simple web interface for predicting law school admissions chances.
Uses v2 models with temporal weighting, interaction features, and years_out.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load models
print("Loading models...")
with open(os.path.join(MODEL_DIR, 'non_urm_baseline.pkl'), 'rb') as f:
    non_urm_baseline_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'non_urm_enhanced.pkl'), 'rb') as f:
    non_urm_enhanced_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'urm_baseline.pkl'), 'rb') as f:
    urm_baseline_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'urm_enhanced.pkl'), 'rb') as f:
    urm_enhanced_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'feature_info.pkl'), 'rb') as f:
    feature_info = pickle.load(f)

# Load selectivity rankings
selectivity_rankings = pd.read_csv(os.path.join(MODEL_DIR, 'school_selectivity.csv'), index_col=0)
school_list = selectivity_rankings.index.tolist()

print(f"Loaded {len(school_list)} schools")
print("Models loaded successfully!")


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def home():
    """Main page with prediction form"""
    return render_template('index.html', schools=school_list)


@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions for all schools"""
    try:
        data = request.json

        # Extract inputs
        lsat = float(data['lsat'])
        gpa = float(data['gpa'])
        timing_days = float(data.get('timing_days', 90))  # Default 90 days (early October)
        is_urm = data.get('is_urm', False)
        has_we = int(data.get('has_we', False))
        softs = int(data.get('softs', 0))  # 0-4 scale
        cycle_year = int(data.get('cycle_year', 2025))
        years_out = float(data.get('years_out', 0))  # v2: continuous years out

        # Decisions from other schools
        decisions = data.get('decisions', [])  # List of {school, result}

        # Calculate decision features
        accepted_schools = [d['school'] for d in decisions if d['result'] == 'Accepted']
        rejected_schools = [d['school'] for d in decisions if d['result'] == 'Rejected']
        waitlisted_schools = [d['school'] for d in decisions if d['result'] == 'Waitlisted']

        def get_selectivity(schools):
            valid = [s for s in schools if s in selectivity_rankings.index]
            return [selectivity_rankings.loc[s, 'selectivity'] for s in valid] if valid else []

        accept_sel = get_selectivity(accepted_schools)
        reject_sel = get_selectivity(rejected_schools)
        wl_sel = get_selectivity(waitlisted_schools)

        # Select appropriate model
        has_decisions = len(decisions) > 0

        if is_urm:
            model = urm_enhanced_model if has_decisions else urm_baseline_model
        else:
            model = non_urm_enhanced_model if has_decisions else non_urm_baseline_model

        # Generate predictions for all schools
        predictions = []

        for school in school_list:
            school_selectivity = selectivity_rankings.loc[school, 'selectivity']

            # v2: compute interaction features
            lsat_x_sel = lsat * school_selectivity
            gpa_x_sel = gpa * school_selectivity
            timing_x_sel = timing_days * school_selectivity

            if has_decisions:
                # Enhanced features (must match v2 training order)
                features = np.array([[
                    lsat,                   # lsat
                    gpa,                    # gpa
                    timing_days,            # app_timing_days
                    has_we,                 # has_work_experience
                    softs,                  # softs_numeric
                    school_selectivity,     # school_selectivity
                    cycle_year,             # cycle_year
                    years_out,              # years_out
                    lsat_x_sel,             # lsat_x_selectivity
                    gpa_x_sel,              # gpa_x_selectivity
                    timing_x_sel,           # timing_x_selectivity
                    len(accepted_schools),  # num_acceptances
                    len(rejected_schools),  # num_rejections
                    len(waitlisted_schools),# num_waitlists
                    len(decisions),         # num_other_apps
                    np.mean(accept_sel) if accept_sel else 0,   # accept_selectivity_avg
                    np.mean(reject_sel) if reject_sel else 0,   # reject_selectivity_avg
                    np.mean(wl_sel) if wl_sel else 0,           # wl_selectivity_avg
                    max(accept_sel) if accept_sel else 0,       # max_acceptance_selectivity
                    min(reject_sel) if reject_sel else 0        # min_rejection_selectivity
                ]])
            else:
                # Baseline features (must match v2 training order)
                features = np.array([[
                    lsat,               # lsat
                    gpa,                # gpa
                    timing_days,        # app_timing_days
                    has_we,             # has_work_experience
                    softs,              # softs_numeric
                    school_selectivity, # school_selectivity
                    cycle_year,         # cycle_year
                    years_out,          # years_out
                    lsat_x_sel,         # lsat_x_selectivity
                    gpa_x_sel,          # gpa_x_selectivity
                    timing_x_sel        # timing_x_selectivity
                ]])

            prob = model.predict_proba(features)[0][1]

            predictions.append({
                'school': school,
                'probability': round(float(prob) * 100, 1),
                'selectivity': round(float(school_selectivity), 1)
            })

        # Sort by probability descending
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_type': 'enhanced' if has_decisions else 'baseline',
            'is_urm': is_urm
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/schools')
def get_schools():
    """Return list of schools with selectivity"""
    schools_data = []
    for school in school_list:
        schools_data.append({
            'name': school,
            'selectivity': round(float(selectivity_rankings.loc[school, 'selectivity']), 1)
        })
    return jsonify(schools_data)


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
