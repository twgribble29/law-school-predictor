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

# Load pre-computed school profiles (scatter data + metrics)
with open(os.path.join(MODEL_DIR, 'school_profiles.json'), 'r') as f:
    school_profiles = json.load(f)

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

# Compute selectivity percentiles (lower acceptance = higher percentile)
# Using rank-based percentile on selectivity scores
all_selectivity_values = selectivity_rankings['selectivity'].values
selectivity_percentiles = {}
for school in school_list:
    sel = selectivity_rankings.loc[school, 'selectivity']
    # Percentile = fraction of schools with LOWER selectivity
    pct = float(np.sum(all_selectivity_values < sel)) / len(all_selectivity_values) * 100
    selectivity_percentiles[school] = round(pct, 1)

# Estimate historical waitlist rates per school from training data
# Default to 15% if no data available
school_waitlist_rates = {}
try:
    wl_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'non_urm_train.csv'))
    for school in school_list:
        school_rows = wl_data[wl_data['school_name'] == school]
        if len(school_rows) > 20:
            total = len(school_rows)
            wl_count = len(school_rows[school_rows['result_clean'] == 'Waitlisted'])
            school_waitlist_rates[school] = round(wl_count / total, 3) if total > 0 else 0.15
        else:
            school_waitlist_rates[school] = 0.15
except Exception:
    for school in school_list:
        school_waitlist_rates[school] = 0.15

# Build schools data with domains
schools_data = []
for school in school_list:
    row = selectivity_rankings.loc[school]
    domain = SCHOOL_DOMAINS.get(school, '')
    schools_data.append({
        'name': school,
        'selectivity': round(float(row['selectivity']), 1),
        'selectivity_percentile': selectivity_percentiles.get(school),
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
                'selectivity_percentile': selectivity_percentiles.get(school),
                'waitlist_rate': school_waitlist_rates.get(school, 0.15),
                'domain': domain,
                'is_t14': school in T14,
                'stats': stats,
                '_sel': sel,
                '_prob_raw': float(prob)
            })

        # ── Marginal Impact Analysis ────────────────────────────────
        # Build counterfactual scenarios for each school
        scenarios = []
        for s_name in ['lsat+1', 'lsat+2', 'lsat+3', 'lsat+5',
                        'gpa+0.1', 'gpa+0.2', 'gpa+0.3',
                        'softs+1', 'timing-30']:
            for pred in predictions:
                sel = pred['_sel']
                cf_lsat = lsat
                cf_gpa = gpa
                cf_softs = softs
                cf_timing = timing_days
                label = ''

                if s_name == 'lsat+1':
                    cf_lsat = min(lsat + 1, 180); label = 'LSAT +1'
                elif s_name == 'lsat+2':
                    cf_lsat = min(lsat + 2, 180); label = 'LSAT +2'
                elif s_name == 'lsat+3':
                    cf_lsat = min(lsat + 3, 180); label = 'LSAT +3'
                elif s_name == 'lsat+5':
                    cf_lsat = min(lsat + 5, 180); label = 'LSAT +5'
                elif s_name == 'gpa+0.1':
                    cf_gpa = min(gpa + 0.1, 4.33); label = 'GPA +0.1'
                elif s_name == 'gpa+0.2':
                    cf_gpa = min(gpa + 0.2, 4.33); label = 'GPA +0.2'
                elif s_name == 'gpa+0.3':
                    cf_gpa = min(gpa + 0.3, 4.33); label = 'GPA +0.3'
                elif s_name == 'softs+1':
                    if softs < 4:
                        cf_softs = softs + 1; label = 'Softs tier up'
                    else:
                        continue
                elif s_name == 'timing-30':
                    if timing_days > 30:
                        cf_timing = timing_days - 30; label = 'Apply 1 month earlier'
                    else:
                        continue

                # Skip if at max already
                if cf_lsat == lsat and 'lsat' in s_name:
                    continue
                if cf_gpa == gpa and 'gpa' in s_name:
                    continue

                row = [cf_lsat, cf_gpa, cf_timing, has_we, cf_softs, sel,
                       cycle_year, years_out,
                       cf_lsat * sel, cf_gpa * sel, cf_timing * sel]
                if has_decisions:
                    row.extend([
                        len(accepted_schools), len(rejected_schools),
                        len(waitlisted_schools), len(decisions),
                        np.mean(accept_sel) if accept_sel else 0,
                        np.mean(reject_sel) if reject_sel else 0,
                        np.mean(wl_sel) if wl_sel else 0,
                        max(accept_sel) if accept_sel else 0,
                        min(reject_sel) if reject_sel else 0
                    ])
                scenarios.append({
                    'row': row,
                    'school': pred['school'],
                    'label': label,
                    'base_prob': pred['_prob_raw']
                })

        # Batch predict all counterfactuals
        best_marginals = {}
        if scenarios:
            cols = enhanced_cols if has_decisions else baseline_cols
            cf_features = pd.DataFrame([s['row'] for s in scenarios], columns=cols)
            cf_probs = model.predict_proba(cf_features)[:, 1]

            for i, sc in enumerate(scenarios):
                delta = float(cf_probs[i]) - sc['base_prob']
                delta_pct = round(delta * 100, 1)
                if delta_pct > 0:
                    key = sc['school']
                    if key not in best_marginals or delta_pct > best_marginals[key]['delta']:
                        best_marginals[key] = {
                            'label': sc['label'],
                            'delta': delta_pct
                        }

        # Attach best marginal to each prediction and clean up
        for pred in predictions:
            marginal = best_marginals.get(pred['school'])
            if marginal and marginal['delta'] >= 0.5:
                pred['marginal'] = marginal
            del pred['_sel']
            del pred['_prob_raw']

        # Sort by selectivity descending (most competitive first)
        predictions.sort(key=lambda x: x['selectivity'], reverse=True)

        # ── Prediction Confidence ──────────────────────────────────
        # Based on how much profile data was provided
        confidence_factors = []
        if softs > 0:
            confidence_factors.append('softs')
        if years_out > 0:
            confidence_factors.append('years_out')
        if has_we:
            confidence_factors.append('work_exp')
        if has_decisions:
            confidence_factors.append('decisions')

        if len(confidence_factors) >= 3:
            confidence = 'high'
        elif len(confidence_factors) >= 1:
            confidence = 'standard'
        else:
            confidence = 'limited'

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_type': 'enhanced' if has_decisions else 'baseline',
            'is_urm': is_urm,
            'applicant': {'lsat': lsat, 'gpa': gpa},
            'confidence': confidence,
            'confidence_factors': confidence_factors
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/schools')
def schools_page():
    """Render the School Profiles page."""
    return render_template('schools.html',
                           schools_json=json.dumps(schools_data),
                           t14_json=json.dumps(T14))


@app.route('/api/school-profile/<school_name>')
def get_school_profile(school_name):
    """Return scatter data, metrics, and probability contours for one school."""
    if school_name not in school_profiles:
        return jsonify({'error': 'School not found'}), 404

    profile = school_profiles[school_name]
    stats = school_stats.get(school_name, {})
    domain = SCHOOL_DOMAINS.get(school_name, '')
    contours = generate_contours(school_name)

    return jsonify({
        'school': school_name,
        'scatter': profile['scatter'],
        'metrics': profile['metrics'],
        'n_total': profile.get('n_total', 0),
        'stats': stats,
        'domain': domain,
        'is_t14': school_name in T14,
        'contours': contours
    })


def generate_contours(school_name):
    """Generate probability grid for contour overlay using the baseline model."""
    sel = selectivity_rankings.loc[school_name, 'selectivity']
    baseline_cols = feature_info['baseline_features']

    lsat_range = list(range(140, 181))  # 41 points
    gpa_range = [round(2.5 + i * 0.05, 2) for i in range(31)]  # 31 points

    # Build all feature rows at once for batch prediction
    rows = []
    for lsat in lsat_range:
        for gpa in gpa_range:
            rows.append([
                lsat, gpa, 90, 0, 0, sel, 2025, 0,
                lsat * sel, gpa * sel, 90 * sel
            ])

    features = pd.DataFrame(rows, columns=baseline_cols)
    probs = non_urm_baseline_model.predict_proba(features)[:, 1]

    # Reshape into 2D grid (lsat rows x gpa cols)
    prob_grid = probs.reshape(len(lsat_range), len(gpa_range))

    return {
        'lsat': lsat_range,
        'gpa': gpa_range,
        'z': [[round(float(p), 3) for p in row] for row in prob_grid]
    }


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
