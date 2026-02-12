"""
Law School Admissions Prediction - Flask Web Application (v2)
==============================================================
Uses v2 models with temporal weighting, interaction features, and years_out.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

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

selectivity_rankings = pd.read_csv(os.path.join(MODEL_DIR, 'school_selectivity.csv'), index_col=0)
school_list = selectivity_rankings.index.tolist()

# Load per-school accepted student distributions
with open(os.path.join(MODEL_DIR, 'school_stats.json'), 'r') as f:
    school_stats = json.load(f)

# T14 schools (by selectivity ranking)
T14 = school_list[:14]

# School domain mapping for logos (favicon API)
SCHOOL_DOMAINS = {
    'Yale University': 'yale.edu',
    'Stanford University': 'stanford.edu',
    'Harvard University': 'harvard.edu',
    'University of Chicago': 'uchicago.edu',
    'Columbia University': 'columbia.edu',
    'New York University': 'nyu.edu',
    'University of California\u2014Berkeley': 'berkeley.edu',
    'University of Pennsylvania': 'upenn.edu',
    'Duke University': 'duke.edu',
    'University of Virginia': 'virginia.edu',
    'Cornell University': 'cornell.edu',
    'Northwestern University': 'northwestern.edu',
    'Georgetown University': 'georgetown.edu',
    'University of Michigan': 'umich.edu',
    'University of California\u2014Los Angeles': 'ucla.edu',
    'University of Southern California': 'usc.edu',
    'Vanderbilt University': 'vanderbilt.edu',
    'University of Texas at Austin': 'utexas.edu',
    'Boston University': 'bu.edu',
    'Washington University in St. Louis': 'wustl.edu',
    'Emory University': 'emory.edu',
    'University of Notre Dame': 'nd.edu',
    'George Washington University': 'gwu.edu',
    'University of Minnesota': 'umn.edu',
    'University of Iowa': 'uiowa.edu',
    'Boston College': 'bc.edu',
    'University of Wisconsin': 'wisc.edu',
    'Fordham University': 'fordham.edu',
    'University of North Carolina': 'unc.edu',
    'Ohio State University': 'osu.edu',
    'Wake Forest University': 'wfu.edu',
    'University of Florida': 'ufl.edu',
    'Indiana University\u2014Bloomington': 'indiana.edu',
    'University of Georgia': 'uga.edu',
    'George Mason University': 'gmu.edu',
    'University of Illinois': 'illinois.edu',
    'University of Alabama': 'ua.edu',
    'Arizona State University': 'asu.edu',
    'Tulane University': 'tulane.edu',
    'University of Colorado': 'colorado.edu',
    'University of Washington': 'uw.edu',
    'Brigham Young University': 'byu.edu',
    'Temple University': 'temple.edu',
    'University of Maryland': 'umd.edu',
    'Penn State University': 'psu.edu',
    'Cardozo School of Law': 'yu.edu',
    'Pepperdine University': 'pepperdine.edu',
    'Southern Methodist University': 'smu.edu',
    'University of Arizona': 'arizona.edu',
    'William & Mary': 'wm.edu',
}

# Build schools data with domains
schools_data = []
for school in school_list:
    row = selectivity_rankings.loc[school]
    domain = SCHOOL_DOMAINS.get(school, '')
    schools_data.append({
        'name': school,
        'selectivity': round(float(row['selectivity']), 1),
        'lsat_median': round(float(row['lsat']), 0),
        'gpa_median': round(float(row['gpa']), 2),
        'domain': domain,
        'is_t14': school in T14
    })

print(f"Loaded {len(school_list)} schools, T14: {len(T14)}")
print("Models loaded successfully!")

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def home():
    return render_template('index.html',
                           schools_json=json.dumps(schools_data),
                           t14_json=json.dumps(T14))

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions for selected schools."""
    try:
        data = request.json

        lsat = float(data['lsat'])
        gpa = float(data['gpa'])
        timing_days = float(data.get('timing_days', 90))
        is_urm = data.get('is_urm', False)
        has_we = int(data.get('has_we', False))
        softs = int(data.get('softs', 0))
        cycle_year = int(data.get('cycle_year', 2025))
        years_out = float(data.get('years_out', 0))

        # Target schools (predict only for these)
        target_schools = data.get('schools', school_list)

        # Decisions from other schools
        decisions = data.get('decisions', [])
        accepted_schools = [d['school'] for d in decisions if d['result'] == 'Accepted']
        rejected_schools = [d['school'] for d in decisions if d['result'] == 'Rejected']
        waitlisted_schools = [d['school'] for d in decisions if d['result'] == 'Waitlisted']

        def get_selectivity_vals(schools):
            valid = [s for s in schools if s in selectivity_rankings.index]
            return [selectivity_rankings.loc[s, 'selectivity'] for s in valid] if valid else []

        accept_sel = get_selectivity_vals(accepted_schools)
        reject_sel = get_selectivity_vals(rejected_schools)
        wl_sel = get_selectivity_vals(waitlisted_schools)

        has_decisions = len(decisions) > 0
        if is_urm:
            model = urm_enhanced_model if has_decisions else urm_baseline_model
        else:
            model = non_urm_enhanced_model if has_decisions else non_urm_baseline_model

        baseline_cols = feature_info['baseline_features']
        enhanced_cols = feature_info['enhanced_features']

        predictions = []
        for school in target_schools:
            if school not in selectivity_rankings.index:
                continue
            sel = selectivity_rankings.loc[school, 'selectivity']
            lsat_x_sel = lsat * sel
            gpa_x_sel = gpa * sel
            timing_x_sel = timing_days * sel

            if has_decisions:
                features = pd.DataFrame([[
                    lsat, gpa, timing_days, has_we, softs, sel, cycle_year,
                    years_out, lsat_x_sel, gpa_x_sel, timing_x_sel,
                    len(accepted_schools), len(rejected_schools),
                    len(waitlisted_schools), len(decisions),
                    np.mean(accept_sel) if accept_sel else 0,
                    np.mean(reject_sel) if reject_sel else 0,
                    np.mean(wl_sel) if wl_sel else 0,
                    max(accept_sel) if accept_sel else 0,
                    min(reject_sel) if reject_sel else 0
                ]], columns=enhanced_cols)
            else:
                features = pd.DataFrame([[
                    lsat, gpa, timing_days, has_we, softs, sel, cycle_year,
                    years_out, lsat_x_sel, gpa_x_sel, timing_x_sel
                ]], columns=baseline_cols)

            prob = model.predict_proba(features)[0][1]
            domain = SCHOOL_DOMAINS.get(school, '')
            stats = school_stats.get(school, {})

            predictions.append({
                'school': school,
                'probability': round(float(prob) * 100, 1),
                'selectivity': round(float(sel), 1),
                'domain': domain,
                'is_t14': school in T14,
                'stats': stats
            })

        # Sort by selectivity descending (most competitive first)
        predictions.sort(key=lambda x: x['selectivity'], reverse=True)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_type': 'enhanced' if has_decisions else 'baseline',
            'is_urm': is_urm,
            'applicant': {'lsat': lsat, 'gpa': gpa}
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/schools')
def get_schools():
    return jsonify(schools_data)


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
