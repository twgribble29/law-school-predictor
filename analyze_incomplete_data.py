"""
Batch 2C: Leveraging Incomplete Data - Analysis & Flexible Models
=================================================================
Analyzes missingness patterns in the training data and tests strategies
for using records with missing softs_tier and years_out to improve
prediction quality.

Strategies tested:
1. Baseline: Complete-case only (current approach, fill missing with 0)
2. Missingness indicators: Add binary flags for missing features
3. Multiple imputation: Fill missing values using IterativeImputer
4. Flexible models: Train separate models for different feature availability
5. Combined: Best-of strategies with confidence levels

Run: python3 analyze_incomplete_data.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DATA_DIR, 'models')

print("=" * 70)
print("INCOMPLETE DATA ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA & ANALYZE MISSINGNESS
# =============================================================================

print("\n1. LOADING DATA AND ANALYZING MISSINGNESS PATTERNS")
print("-" * 50)

df_train = pd.read_csv(os.path.join(DATA_DIR, 'non_urm_train.csv'))
df_val = pd.read_csv(os.path.join(DATA_DIR, 'non_urm_val.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'non_urm_test_2025.csv'))

selectivity_rankings = pd.read_csv(
    os.path.join(MODEL_DIR, 'school_selectivity.csv'), index_col=0
)

print(f"Training: {len(df_train):,} rows")
print(f"Validation: {len(df_val):,} rows")
print(f"Test: {len(df_test):,} rows")

# Map softs_tier to numeric
softs_map = {'T1': 4, 'T2': 3, 'T3': 2, 'T4': 1}

for df in [df_train, df_val, df_test]:
    df['softs'] = df['softs_tier'].map(softs_map)  # NaN if missing
    df['selectivity'] = df['school_name'].map(
        selectivity_rankings['selectivity']
    )

# Missingness analysis
print("\nMissingness in training data:")
for col in ['softs', 'years_out']:
    missing = df_train[col].isna().sum()
    pct = missing / len(df_train) * 100
    print(f"  {col}: {missing:,} missing ({pct:.1f}%)")

# =============================================================================
# 2. MISSINGNESS PATTERN ANALYSIS (MCAR/MAR/MNAR)
# =============================================================================

print("\n2. MISSINGNESS PATTERN ANALYSIS")
print("-" * 50)

# Test: Does missingness correlate with outcome?
y_train = (df_train['result_clean'] == 'Accepted').astype(int)

for col in ['softs', 'years_out']:
    missing_mask = df_train[col].isna()
    if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
        accept_when_missing = y_train[missing_mask].mean()
        accept_when_present = y_train[~missing_mask].mean()
        print(f"\n  {col}:")
        print(f"    Accept rate when {col} is MISSING: {accept_when_missing*100:.1f}%")
        print(f"    Accept rate when {col} is PRESENT: {accept_when_present*100:.1f}%")
        diff = accept_when_present - accept_when_missing
        if abs(diff) > 0.02:
            print(f"    -> MNAR likely: {abs(diff)*100:.1f}pp difference")
            print(f"       Missingness itself is informative!")
        else:
            print(f"    -> MCAR/MAR likely: only {abs(diff)*100:.1f}pp difference")

# Cross-tabulation: are certain LSAT ranges more likely to have missing softs?
df_train['lsat_bucket'] = pd.cut(df_train['lsat'], bins=[119, 150, 160, 170, 180])
crosstab = df_train.groupby('lsat_bucket')['softs'].apply(
    lambda x: x.isna().mean() * 100
)
print("\n  Softs missingness by LSAT bucket:")
for bucket, pct in crosstab.items():
    print(f"    {bucket}: {pct:.1f}% missing")

# =============================================================================
# 3. PREPARE FEATURES
# =============================================================================

print("\n3. PREPARING FEATURES FOR MODEL COMPARISON")
print("-" * 50)


def prepare_features(df, strategy='baseline'):
    """Prepare feature matrix using specified strategy."""
    df = df.copy()

    # Core features always available
    base_feats = ['lsat', 'gpa', 'app_timing_days', 'has_work_experience',
                  'selectivity', 'cycle_year']

    # Handle softs & years_out depending on strategy
    if strategy == 'baseline':
        # Current approach: fill NaN with 0
        df['softs_clean'] = df['softs'].fillna(0)
        df['years_out_clean'] = df['years_out'].fillna(0)
        feats = base_feats + ['softs_clean', 'years_out_clean']

    elif strategy == 'missing_indicators':
        # Add binary flags + fill with 0
        df['softs_clean'] = df['softs'].fillna(0)
        df['years_out_clean'] = df['years_out'].fillna(0)
        df['softs_missing'] = df['softs'].isna().astype(int)
        df['years_out_missing'] = df['years_out'].isna().astype(int)
        feats = base_feats + ['softs_clean', 'years_out_clean',
                              'softs_missing', 'years_out_missing']

    elif strategy == 'imputed':
        # Will be handled separately with IterativeImputer
        df['softs_clean'] = df['softs']  # keep NaN for imputer
        df['years_out_clean'] = df['years_out']
        feats = base_feats + ['softs_clean', 'years_out_clean']

    elif strategy == 'native_nan':
        # HistGradientBoosting handles NaN natively
        df['softs_clean'] = df['softs']
        df['years_out_clean'] = df['years_out']
        feats = base_feats + ['softs_clean', 'years_out_clean']

    elif strategy == 'native_nan_with_indicators':
        # Native NaN handling + missingness indicators
        df['softs_clean'] = df['softs']
        df['years_out_clean'] = df['years_out']
        df['softs_missing'] = df['softs'].isna().astype(int)
        df['years_out_missing'] = df['years_out'].isna().astype(int)
        feats = base_feats + ['softs_clean', 'years_out_clean',
                              'softs_missing', 'years_out_missing']

    # Add interaction features
    df['lsat_x_sel'] = df['lsat'] * df['selectivity']
    df['gpa_x_sel'] = df['gpa'] * df['selectivity']
    df['timing_x_sel'] = df['app_timing_days'] * df['selectivity']
    feats += ['lsat_x_sel', 'gpa_x_sel', 'timing_x_sel']

    X = df[feats].copy()
    y = (df['result_clean'] == 'Accepted').astype(int)

    return X, y, feats


# =============================================================================
# 4. TRAIN AND COMPARE MODELS
# =============================================================================

print("\n4. TRAINING AND COMPARING STRATEGIES")
print("-" * 50)

model_params = dict(
    max_iter=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=50,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
)

results = {}

# Strategy 1: Baseline (fill with 0)
print("\n  Training: Baseline (fill missing with 0)...")
X_train_b, y_train_b, feats_b = prepare_features(df_train, 'baseline')
X_val_b, y_val_b, _ = prepare_features(df_val, 'baseline')
X_test_b, y_test_b, _ = prepare_features(df_test, 'baseline')

model_baseline = HistGradientBoostingClassifier(**model_params)
model_baseline.fit(X_train_b, y_train_b)

results['baseline'] = {
    'val_auc': roc_auc_score(y_val_b, model_baseline.predict_proba(X_val_b)[:, 1]),
    'test_auc': roc_auc_score(y_test_b, model_baseline.predict_proba(X_test_b)[:, 1]),
    'val_brier': brier_score_loss(y_val_b, model_baseline.predict_proba(X_val_b)[:, 1]),
    'n_features': len(feats_b),
}

# Strategy 2: Missingness indicators
print("  Training: Missingness indicators...")
X_train_mi, y_train_mi, feats_mi = prepare_features(df_train, 'missing_indicators')
X_val_mi, y_val_mi, _ = prepare_features(df_val, 'missing_indicators')
X_test_mi, y_test_mi, _ = prepare_features(df_test, 'missing_indicators')

model_mi = HistGradientBoostingClassifier(**model_params)
model_mi.fit(X_train_mi, y_train_mi)

results['missing_indicators'] = {
    'val_auc': roc_auc_score(y_val_mi, model_mi.predict_proba(X_val_mi)[:, 1]),
    'test_auc': roc_auc_score(y_test_mi, model_mi.predict_proba(X_test_mi)[:, 1]),
    'val_brier': brier_score_loss(y_val_mi, model_mi.predict_proba(X_val_mi)[:, 1]),
    'n_features': len(feats_mi),
}

# Strategy 3: Multiple imputation
print("  Training: Multiple imputation (IterativeImputer)...")
X_train_imp, y_train_imp, feats_imp = prepare_features(df_train, 'imputed')
X_val_imp, y_val_imp, _ = prepare_features(df_val, 'imputed')
X_test_imp, y_test_imp, _ = prepare_features(df_test, 'imputed')

imputer = IterativeImputer(random_state=42, max_iter=10)
X_train_imp_filled = pd.DataFrame(
    imputer.fit_transform(X_train_imp), columns=feats_imp
)
X_val_imp_filled = pd.DataFrame(
    imputer.transform(X_val_imp), columns=feats_imp
)
X_test_imp_filled = pd.DataFrame(
    imputer.transform(X_test_imp), columns=feats_imp
)

model_imp = HistGradientBoostingClassifier(**model_params)
model_imp.fit(X_train_imp_filled, y_train_imp)

results['imputed'] = {
    'val_auc': roc_auc_score(y_val_imp, model_imp.predict_proba(X_val_imp_filled)[:, 1]),
    'test_auc': roc_auc_score(y_test_imp, model_imp.predict_proba(X_test_imp_filled)[:, 1]),
    'val_brier': brier_score_loss(y_val_imp, model_imp.predict_proba(X_val_imp_filled)[:, 1]),
    'n_features': len(feats_imp),
}

# Strategy 4: Native NaN handling (HistGradientBoosting supports NaN)
print("  Training: Native NaN handling...")
X_train_nn, y_train_nn, feats_nn = prepare_features(df_train, 'native_nan')
X_val_nn, y_val_nn, _ = prepare_features(df_val, 'native_nan')
X_test_nn, y_test_nn, _ = prepare_features(df_test, 'native_nan')

model_nn = HistGradientBoostingClassifier(**model_params)
model_nn.fit(X_train_nn, y_train_nn)

results['native_nan'] = {
    'val_auc': roc_auc_score(y_val_nn, model_nn.predict_proba(X_val_nn)[:, 1]),
    'test_auc': roc_auc_score(y_test_nn, model_nn.predict_proba(X_test_nn)[:, 1]),
    'val_brier': brier_score_loss(y_val_nn, model_nn.predict_proba(X_val_nn)[:, 1]),
    'n_features': len(feats_nn),
}

# Strategy 5: Native NaN + missingness indicators (RECOMMENDED)
print("  Training: Native NaN + missingness indicators...")
X_train_nni, y_train_nni, feats_nni = prepare_features(df_train, 'native_nan_with_indicators')
X_val_nni, y_val_nni, _ = prepare_features(df_val, 'native_nan_with_indicators')
X_test_nni, y_test_nni, _ = prepare_features(df_test, 'native_nan_with_indicators')

model_nni = HistGradientBoostingClassifier(**model_params)
model_nni.fit(X_train_nni, y_train_nni)

results['native_nan_indicators'] = {
    'val_auc': roc_auc_score(y_val_nni, model_nni.predict_proba(X_val_nni)[:, 1]),
    'test_auc': roc_auc_score(y_test_nni, model_nni.predict_proba(X_test_nni)[:, 1]),
    'val_brier': brier_score_loss(y_val_nni, model_nni.predict_proba(X_val_nni)[:, 1]),
    'n_features': len(feats_nni),
}

# =============================================================================
# 5. FLEXIBLE MODELS (different feature sets)
# =============================================================================

print("\n5. TRAINING FLEXIBLE MODELS (feature-set specific)")
print("-" * 50)

# Model A: Full features (softs + years_out available)
complete_mask_train = df_train['softs'].notna() & df_train['years_out'].notna()
complete_mask_val = df_val['softs'].notna() & df_val['years_out'].notna()
complete_mask_test = df_test['softs'].notna() & df_test['years_out'].notna()

print(f"\n  Complete-case records: {complete_mask_train.sum():,} train, "
      f"{complete_mask_val.sum():,} val, {complete_mask_test.sum():,} test")

if complete_mask_train.sum() > 500:
    X_comp_train, y_comp_train, _ = prepare_features(
        df_train[complete_mask_train], 'baseline'
    )
    X_comp_val, y_comp_val, _ = prepare_features(
        df_val[complete_mask_val], 'baseline'
    )

    model_complete = HistGradientBoostingClassifier(**model_params)
    model_complete.fit(X_comp_train, y_comp_train)

    if complete_mask_val.sum() > 50:
        auc_comp = roc_auc_score(
            y_comp_val, model_complete.predict_proba(X_comp_val)[:, 1]
        )
        print(f"  Complete-case model AUC: {auc_comp:.4f} "
              f"(trained on {complete_mask_train.sum():,} rows)")
    else:
        auc_comp = None
        print("  Not enough complete validation data to evaluate")

# Model B: Without softs (LSAT + GPA + timing + years_out + selectivity)
print("\n  Training: No-softs model...")
no_softs_feats = ['lsat', 'gpa', 'app_timing_days', 'has_work_experience',
                  'selectivity', 'cycle_year', 'lsat_x_sel', 'gpa_x_sel',
                  'timing_x_sel']

for df in [df_train, df_val, df_test]:
    df['lsat_x_sel'] = df['lsat'] * df['selectivity']
    df['gpa_x_sel'] = df['gpa'] * df['selectivity']
    df['timing_x_sel'] = df['app_timing_days'] * df['selectivity']

X_nosofts_train = df_train[no_softs_feats].fillna(0)
y_nosofts_train = (df_train['result_clean'] == 'Accepted').astype(int)
X_nosofts_val = df_val[no_softs_feats].fillna(0)
y_nosofts_val = (df_val['result_clean'] == 'Accepted').astype(int)

model_nosofts = HistGradientBoostingClassifier(**model_params)
model_nosofts.fit(X_nosofts_train, y_nosofts_train)

auc_nosofts = roc_auc_score(
    y_nosofts_val, model_nosofts.predict_proba(X_nosofts_val)[:, 1]
)
print(f"  No-softs model AUC: {auc_nosofts:.4f}")

# =============================================================================
# 6. RESULTS COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Strategy':<35} {'Val AUC':>10} {'Test AUC':>10} {'Brier':>10} {'Features':>10}")
print("-" * 75)

# Sort by val AUC
sorted_results = sorted(results.items(), key=lambda x: x[1]['val_auc'], reverse=True)
best_strategy = sorted_results[0][0]
baseline_auc = results['baseline']['val_auc']

for name, r in sorted_results:
    delta = r['val_auc'] - baseline_auc
    marker = " <-- BEST" if name == best_strategy else ""
    print(f"  {name:<33} {r['val_auc']:>10.4f} {r['test_auc']:>10.4f} "
          f"{r['val_brier']:>10.4f} {r['n_features']:>10}{marker}")

print(f"\n  Baseline AUC: {baseline_auc:.4f}")
print(f"  Best strategy: {best_strategy}")
print(f"  Improvement: {sorted_results[0][1]['val_auc'] - baseline_auc:+.4f} AUC")

# =============================================================================
# 7. ANALYZE MISSINGNESS AS SIGNAL BY SCHOOL TIER
# =============================================================================

print("\n" + "=" * 70)
print("MISSINGNESS AS SIGNAL - BY SCHOOL TIER")
print("=" * 70)

# Add selectivity tiers
df_train['sel_tier'] = pd.qcut(
    df_train['selectivity'].fillna(df_train['selectivity'].median()),
    q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High']
)

for tier in ['High', 'Mid-High', 'Mid-Low', 'Low']:
    tier_data = df_train[df_train['sel_tier'] == tier]
    softs_missing_accept = (
        tier_data[tier_data['softs'].isna()]['result_clean'] == 'Accepted'
    ).mean()
    softs_present_accept = (
        tier_data[tier_data['softs'].notna()]['result_clean'] == 'Accepted'
    ).mean()
    diff = softs_present_accept - softs_missing_accept
    print(f"\n  {tier} selectivity schools:")
    print(f"    Softs present: {softs_present_accept*100:.1f}% accepted "
          f"(n={tier_data['softs'].notna().sum():,})")
    print(f"    Softs missing: {softs_missing_accept*100:.1f}% accepted "
          f"(n={tier_data['softs'].isna().sum():,})")
    print(f"    Difference: {diff*100:+.1f}pp")

# =============================================================================
# 8. SAVE BEST MODEL AND REPORT
# =============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save the analysis report
report = {
    'missingness': {
        'softs_missing_pct': round(df_train['softs'].isna().mean() * 100, 1),
        'years_out_missing_pct': round(df_train['years_out'].isna().mean() * 100, 1),
    },
    'strategy_comparison': {
        name: {
            'val_auc': round(r['val_auc'], 4),
            'test_auc': round(r['test_auc'], 4),
            'val_brier': round(r['val_brier'], 4),
            'n_features': r['n_features'],
        }
        for name, r in results.items()
    },
    'best_strategy': best_strategy,
    'improvement_over_baseline': round(
        sorted_results[0][1]['val_auc'] - baseline_auc, 4
    ),
    'recommendation': (
        f"Use '{best_strategy}' strategy. "
        f"It improves AUC by {sorted_results[0][1]['val_auc'] - baseline_auc:+.4f} "
        f"over the baseline approach of filling missing values with 0."
    ),
}

report_path = os.path.join(MODEL_DIR, 'incomplete_data_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n  Report saved to: {report_path}")

# If the best strategy meaningfully improves on baseline, save that model
if sorted_results[0][1]['val_auc'] - baseline_auc >= 0.001:
    best_name = best_strategy
    # Map strategy name to its trained model
    model_map = {
        'baseline': model_baseline,
        'missing_indicators': model_mi,
        'imputed': model_imp,
        'native_nan': model_nn,
        'native_nan_indicators': model_nni,
    }
    best_model = model_map[best_name]
    model_path = os.path.join(MODEL_DIR, 'non_urm_baseline_improved.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Best model saved to: {model_path}")

    # Save feature info for the improved model
    feat_map = {
        'baseline': feats_b,
        'missing_indicators': feats_mi,
        'imputed': feats_imp,
        'native_nan': feats_nn,
        'native_nan_indicators': feats_nni,
    }
    improved_feats = feat_map[best_name]
    feat_path = os.path.join(MODEL_DIR, 'improved_feature_info.json')
    with open(feat_path, 'w') as f:
        json.dump({
            'strategy': best_name,
            'features': improved_feats,
        }, f, indent=2)
    print(f"  Feature info saved to: {feat_path}")
else:
    print("\n  No meaningful improvement found. Keeping baseline model.")

print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print(f"""
Current State:
  - Training data: {len(df_train):,} rows
  - softs_tier missing in {df_train['softs'].isna().mean()*100:.1f}% of rows
  - years_out missing in {df_train['years_out'].isna().mean()*100:.1f}% of rows

Key Findings:
  - Best strategy: {best_strategy}
  - AUC improvement over baseline: {sorted_results[0][1]['val_auc'] - baseline_auc:+.4f}

Recommendations:
  1. {"Use missingness indicators - they capture the MNAR signal"
     if 'indicator' in best_strategy else
     "Current 0-fill approach is already near-optimal"}
  2. HistGradientBoosting handles NaN natively - no need for imputation
  3. Show confidence levels to users:
     - "High confidence" when softs + years_out are provided
     - "Standard confidence" when using defaults
  4. For school-specific analysis, missing softs correlates with outcomes
     differently across selectivity tiers - this is exploitable signal

Next Steps:
  - If improvement > 0.005 AUC, integrate the improved model into app.py
  - Add confidence badge to prediction results
  - Consider training URM version with same strategy
""")

print("ANALYSIS COMPLETE")
print("=" * 70)
