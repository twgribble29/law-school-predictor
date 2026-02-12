"""
Law School Admissions Prediction Model - Model Training
=========================================================
This script trains baseline and enhanced Gradient Boosting models for prediction.
Uses scikit-learn's HistGradientBoostingClassifier (fast, no OpenMP dependency).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import warnings
import os
import json
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "/Users/tygribble/Desktop/Law_Data/law_school_predictor/"
MODEL_DIR = f"{DATA_DIR}models/"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 70)
print("LAW SCHOOL ADMISSIONS - MODEL TRAINING")
print("=" * 70)

# Load Non-URM data
df_train = pd.read_csv(f"{DATA_DIR}non_urm_train.csv")
df_val = pd.read_csv(f"{DATA_DIR}non_urm_val.csv")
df_test = pd.read_csv(f"{DATA_DIR}non_urm_test_2025.csv")

# Load URM data
df_urm_train = pd.read_csv(f"{DATA_DIR}urm_train.csv")
df_urm_val = pd.read_csv(f"{DATA_DIR}urm_val.csv")
df_urm_test = pd.read_csv(f"{DATA_DIR}urm_test_2025.csv")

# Load selectivity rankings
selectivity_rankings = pd.read_csv(f"{MODEL_DIR}school_selectivity.csv", index_col=0)

print(f"Non-URM: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")
print(f"URM: Train={len(df_urm_train):,}, Val={len(df_urm_val):,}, Test={len(df_urm_test):,}")
print(f"Schools with selectivity rankings: {len(selectivity_rankings)}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_baseline_features(df, selectivity_rankings):
    """Prepare baseline features (stats only, no decision information)"""
    features = df[['lsat', 'gpa', 'app_timing_days', 'has_work_experience']].copy()

    # Add softs as numeric
    features['softs_numeric'] = df['softs_tier'].map({'T1': 4, 'T2': 3, 'T3': 2, 'T4': 1}).fillna(0)

    # Add school selectivity
    features['school_selectivity'] = df['school_name'].map(selectivity_rankings['selectivity'])

    # Add cycle year (to capture temporal drift)
    features['cycle_year'] = df['cycle_year']

    # Fill missing values with medians
    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    return features


def prepare_enhanced_features_loo(df, selectivity_rankings, verbose=False):
    """
    Prepare enhanced features with leave-one-out to avoid leakage.
    For each application, use decisions from OTHER schools only.
    """
    baseline_features = prepare_baseline_features(df, selectivity_rankings)
    decision_features_list = []

    if verbose:
        print("Computing decision features (leave-one-out)...")

    total_users = df['user_uuid'].nunique()

    for i, uuid in enumerate(df['user_uuid'].unique()):
        if verbose and i % 5000 == 0:
            print(f"  Processing user {i+1}/{total_users}...")

        user_apps = df[df['user_uuid'] == uuid]

        for idx in user_apps.index:
            target_school = user_apps.loc[idx, 'school_name']

            # Get decisions from OTHER schools only
            user_other_apps = user_apps[user_apps['school_name'] != target_school]

            if len(user_other_apps) == 0:
                # No other decisions
                decision_features_list.append({
                    'index': idx,
                    'num_acceptances': 0,
                    'num_rejections': 0,
                    'num_waitlists': 0,
                    'num_other_apps': 0,
                    'accept_selectivity_avg': 0,
                    'reject_selectivity_avg': 0,
                    'wl_selectivity_avg': 0,
                    'max_acceptance_selectivity': 0,
                    'min_rejection_selectivity': 0
                })
                continue

            accepted_schools = user_other_apps[user_other_apps['result_clean'] == 'Accepted']['school_name'].tolist()
            rejected_schools = user_other_apps[user_other_apps['result_clean'] == 'Rejected']['school_name'].tolist()
            waitlisted_schools = user_other_apps[user_other_apps['result_clean'] == 'Waitlisted']['school_name'].tolist()

            # Get selectivity values
            def get_selectivity(schools):
                valid = [s for s in schools if s in selectivity_rankings.index]
                return [selectivity_rankings.loc[s, 'selectivity'] for s in valid] if valid else []

            accept_sel = get_selectivity(accepted_schools)
            reject_sel = get_selectivity(rejected_schools)
            wl_sel = get_selectivity(waitlisted_schools)

            decision_features_list.append({
                'index': idx,
                'num_acceptances': len(accepted_schools),
                'num_rejections': len(rejected_schools),
                'num_waitlists': len(waitlisted_schools),
                'num_other_apps': len(user_other_apps),
                'accept_selectivity_avg': np.mean(accept_sel) if accept_sel else 0,
                'reject_selectivity_avg': np.mean(reject_sel) if reject_sel else 0,
                'wl_selectivity_avg': np.mean(wl_sel) if wl_sel else 0,
                'max_acceptance_selectivity': max(accept_sel) if accept_sel else 0,
                'min_rejection_selectivity': min(reject_sel) if reject_sel else 0
            })

    decision_df = pd.DataFrame(decision_features_list).set_index('index')
    enhanced_features = baseline_features.join(decision_df)

    # Fill any remaining NaN
    enhanced_features = enhanced_features.fillna(0)

    return enhanced_features


# =============================================================================
# TRAIN BASELINE MODELS
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING BASELINE MODELS (Stats Only)")
print("=" * 70)

# --- Non-URM Baseline ---
print("\n--- Non-URM Baseline Model ---")
X_train_baseline = prepare_baseline_features(df_train, selectivity_rankings)
y_train = (df_train['result_clean'] == 'Accepted').astype(int)

X_val_baseline = prepare_baseline_features(df_val, selectivity_rankings)
y_val = (df_val['result_clean'] == 'Accepted').astype(int)

X_test_baseline = prepare_baseline_features(df_test, selectivity_rankings)
y_test = (df_test['result_clean'] == 'Accepted').astype(int)

print(f"Feature columns: {list(X_train_baseline.columns)}")

baseline_model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

baseline_model.fit(X_train_baseline, y_train)

# Evaluate
y_val_pred_baseline = baseline_model.predict_proba(X_val_baseline)[:, 1]
y_test_pred_baseline = baseline_model.predict_proba(X_test_baseline)[:, 1]

auc_val_baseline = roc_auc_score(y_val, y_val_pred_baseline)
auc_test_baseline = roc_auc_score(y_test, y_test_pred_baseline)

print(f"Validation AUC: {auc_val_baseline:.4f}")
print(f"2025 Test AUC:  {auc_test_baseline:.4f}")

# --- URM Baseline ---
print("\n--- URM Baseline Model ---")
X_urm_train_baseline = prepare_baseline_features(df_urm_train, selectivity_rankings)
y_urm_train = (df_urm_train['result_clean'] == 'Accepted').astype(int)

X_urm_val_baseline = prepare_baseline_features(df_urm_val, selectivity_rankings)
y_urm_val = (df_urm_val['result_clean'] == 'Accepted').astype(int)

X_urm_test_baseline = prepare_baseline_features(df_urm_test, selectivity_rankings)
y_urm_test = (df_urm_test['result_clean'] == 'Accepted').astype(int)

urm_baseline_model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

urm_baseline_model.fit(X_urm_train_baseline, y_urm_train)

y_urm_val_pred_baseline = urm_baseline_model.predict_proba(X_urm_val_baseline)[:, 1]
y_urm_test_pred_baseline = urm_baseline_model.predict_proba(X_urm_test_baseline)[:, 1]

auc_urm_val_baseline = roc_auc_score(y_urm_val, y_urm_val_pred_baseline)
auc_urm_test_baseline = roc_auc_score(y_urm_test, y_urm_test_pred_baseline)

print(f"Validation AUC: {auc_urm_val_baseline:.4f}")
print(f"2025 Test AUC:  {auc_urm_test_baseline:.4f}")

# =============================================================================
# TRAIN ENHANCED MODELS (With Decision Features)
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING ENHANCED MODELS (With Decision Features)")
print("=" * 70)

# --- Non-URM Enhanced ---
print("\n--- Non-URM Enhanced Model ---")
X_train_enhanced = prepare_enhanced_features_loo(df_train, selectivity_rankings, verbose=True)
X_val_enhanced = prepare_enhanced_features_loo(df_val, selectivity_rankings, verbose=False)
X_test_enhanced = prepare_enhanced_features_loo(df_test, selectivity_rankings, verbose=False)

print(f"Enhanced feature columns: {list(X_train_enhanced.columns)}")

enhanced_model = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

enhanced_model.fit(X_train_enhanced, y_train)

y_val_pred_enhanced = enhanced_model.predict_proba(X_val_enhanced)[:, 1]
y_test_pred_enhanced = enhanced_model.predict_proba(X_test_enhanced)[:, 1]

auc_val_enhanced = roc_auc_score(y_val, y_val_pred_enhanced)
auc_test_enhanced = roc_auc_score(y_test, y_test_pred_enhanced)

print(f"Validation AUC: {auc_val_enhanced:.4f} (baseline: {auc_val_baseline:.4f}, improvement: +{auc_val_enhanced-auc_val_baseline:.4f})")
print(f"2025 Test AUC:  {auc_test_enhanced:.4f} (baseline: {auc_test_baseline:.4f}, improvement: +{auc_test_enhanced-auc_test_baseline:.4f})")

# --- URM Enhanced ---
print("\n--- URM Enhanced Model ---")
X_urm_train_enhanced = prepare_enhanced_features_loo(df_urm_train, selectivity_rankings, verbose=True)
X_urm_val_enhanced = prepare_enhanced_features_loo(df_urm_val, selectivity_rankings, verbose=False)
X_urm_test_enhanced = prepare_enhanced_features_loo(df_urm_test, selectivity_rankings, verbose=False)

urm_enhanced_model = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

urm_enhanced_model.fit(X_urm_train_enhanced, y_urm_train)

y_urm_val_pred_enhanced = urm_enhanced_model.predict_proba(X_urm_val_enhanced)[:, 1]
y_urm_test_pred_enhanced = urm_enhanced_model.predict_proba(X_urm_test_enhanced)[:, 1]

auc_urm_val_enhanced = roc_auc_score(y_urm_val, y_urm_val_pred_enhanced)
auc_urm_test_enhanced = roc_auc_score(y_urm_test, y_urm_test_pred_enhanced)

print(f"Validation AUC: {auc_urm_val_enhanced:.4f} (baseline: {auc_urm_val_baseline:.4f}, improvement: +{auc_urm_val_enhanced-auc_urm_val_baseline:.4f})")
print(f"2025 Test AUC:  {auc_urm_test_enhanced:.4f} (baseline: {auc_urm_test_baseline:.4f}, improvement: +{auc_urm_test_enhanced-auc_urm_test_baseline:.4f})")

# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS (Non-URM Enhanced on 2025 Test)")
print("=" * 70)

prob_true, prob_pred = calibration_curve(y_test, y_test_pred_enhanced, n_bins=10)

print("\nCalibration (Predicted vs Actual by Decile):")
print("-" * 45)
print(f"{'Decile':<10} {'Predicted':>12} {'Actual':>12} {'Diff':>10}")
print("-" * 45)
for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
    diff = true - pred
    print(f"{i+1:<10} {pred*100:>11.1f}% {true*100:>11.1f}% {diff*100:>+9.1f}%")

# =============================================================================
# SAVE MODELS
# =============================================================================

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Save models
with open(f"{MODEL_DIR}non_urm_baseline.pkl", 'wb') as f:
    pickle.dump(baseline_model, f)
print(f"Saved: {MODEL_DIR}non_urm_baseline.pkl")

with open(f"{MODEL_DIR}non_urm_enhanced.pkl", 'wb') as f:
    pickle.dump(enhanced_model, f)
print(f"Saved: {MODEL_DIR}non_urm_enhanced.pkl")

with open(f"{MODEL_DIR}urm_baseline.pkl", 'wb') as f:
    pickle.dump(urm_baseline_model, f)
print(f"Saved: {MODEL_DIR}urm_baseline.pkl")

with open(f"{MODEL_DIR}urm_enhanced.pkl", 'wb') as f:
    pickle.dump(urm_enhanced_model, f)
print(f"Saved: {MODEL_DIR}urm_enhanced.pkl")

# Save feature names for loading
feature_info = {
    'baseline_features': list(X_train_baseline.columns),
    'enhanced_features': list(X_train_enhanced.columns)
}
with open(f"{MODEL_DIR}feature_info.pkl", 'wb') as f:
    pickle.dump(feature_info, f)
print(f"Saved: {MODEL_DIR}feature_info.pkl")

# =============================================================================
# VALIDATION REPORT
# =============================================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION REPORT")
print("=" * 70)

validation_report = {
    'non_urm': {
        'baseline': {
            'val_auc': float(auc_val_baseline),
            'test_auc': float(auc_test_baseline)
        },
        'enhanced': {
            'val_auc': float(auc_val_enhanced),
            'test_auc': float(auc_test_enhanced)
        },
        'improvement': {
            'val_auc': float(auc_val_enhanced - auc_val_baseline),
            'test_auc': float(auc_test_enhanced - auc_test_baseline)
        }
    },
    'urm': {
        'baseline': {
            'val_auc': float(auc_urm_val_baseline),
            'test_auc': float(auc_urm_test_baseline)
        },
        'enhanced': {
            'val_auc': float(auc_urm_val_enhanced),
            'test_auc': float(auc_urm_test_enhanced)
        },
        'improvement': {
            'val_auc': float(auc_urm_val_enhanced - auc_urm_val_baseline),
            'test_auc': float(auc_urm_test_enhanced - auc_urm_test_baseline)
        }
    }
}

with open(f"{MODEL_DIR}validation_report.json", 'w') as f:
    json.dump(validation_report, f, indent=2)

print("\n+------------------+------------+------------+")
print("| Model            | Val AUC    | Test AUC   |")
print("+------------------+------------+------------+")
print(f"| Non-URM Baseline | {auc_val_baseline:.4f}     | {auc_test_baseline:.4f}     |")
print(f"| Non-URM Enhanced | {auc_val_enhanced:.4f}     | {auc_test_enhanced:.4f}     |")
print(f"| URM Baseline     | {auc_urm_val_baseline:.4f}     | {auc_urm_test_baseline:.4f}     |")
print(f"| URM Enhanced     | {auc_urm_val_enhanced:.4f}     | {auc_urm_test_enhanced:.4f}     |")
print("+------------------+------------+------------+")

print(f"\nValidation report saved to: {MODEL_DIR}validation_report.json")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)
