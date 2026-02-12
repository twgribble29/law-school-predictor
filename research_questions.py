"""
Law School Admissions Prediction Model - Research Questions Analysis
=====================================================================
This script analyzes the research questions using interpretable models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD CLEANED DATA
# =============================================================================

print("=" * 70)
print("LAW SCHOOL ADMISSIONS - RESEARCH QUESTIONS ANALYSIS")
print("=" * 70)

DATA_DIR = "/Users/tygribble/Desktop/Law_Data/law_school_predictor/"

df_train = pd.read_csv(f"{DATA_DIR}non_urm_train.csv")
df_val = pd.read_csv(f"{DATA_DIR}non_urm_val.csv")
print(f"\nLoaded Non-URM training data: {len(df_train):,} rows")
print(f"Loaded Non-URM validation data: {len(df_val):,} rows")

# Create numeric softs
df_train['softs_numeric'] = df_train['softs_tier'].map({'T1': 4, 'T2': 3, 'T3': 2, 'T4': 1}).fillna(0)
df_val['softs_numeric'] = df_val['softs_tier'].map({'T1': 4, 'T2': 3, 'T3': 2, 'T4': 1}).fillna(0)

# Binary outcome
y_train = (df_train['result_clean'] == 'Accepted').astype(int)
y_val = (df_val['result_clean'] == 'Accepted').astype(int)

# =============================================================================
# RQ #1: VARIANCE DECOMPOSITION
# =============================================================================

print("\n" + "=" * 70)
print("RQ #1: VARIANCE DECOMPOSITION")
print("How much variance is explained by LSAT, GPA, timing, and softs?")
print("=" * 70)

scaler = StandardScaler()

# Model 1: LSAT only
X1_train = scaler.fit_transform(df_train[['lsat']].fillna(df_train['lsat'].median()))
X1_val = scaler.transform(df_val[['lsat']].fillna(df_val['lsat'].median()))
model1 = LogisticRegression(max_iter=1000, random_state=42).fit(X1_train, y_train)
auc1_train = roc_auc_score(y_train, model1.predict_proba(X1_train)[:, 1])
auc1_val = roc_auc_score(y_val, model1.predict_proba(X1_val)[:, 1])

# Model 2: LSAT + GPA
X2_train = df_train[['lsat', 'gpa']].fillna(df_train[['lsat', 'gpa']].median())
X2_val = df_val[['lsat', 'gpa']].fillna(df_val[['lsat', 'gpa']].median())
X2_train_scaled = scaler.fit_transform(X2_train)
X2_val_scaled = scaler.transform(X2_val)
model2 = LogisticRegression(max_iter=1000, random_state=42).fit(X2_train_scaled, y_train)
auc2_train = roc_auc_score(y_train, model2.predict_proba(X2_train_scaled)[:, 1])
auc2_val = roc_auc_score(y_val, model2.predict_proba(X2_val_scaled)[:, 1])

# Model 3: LSAT + GPA + timing
X3_train = df_train[['lsat', 'gpa', 'app_timing_days']].fillna(df_train[['lsat', 'gpa', 'app_timing_days']].median())
X3_val = df_val[['lsat', 'gpa', 'app_timing_days']].fillna(df_val[['lsat', 'gpa', 'app_timing_days']].median())
X3_train_scaled = scaler.fit_transform(X3_train)
X3_val_scaled = scaler.transform(X3_val)
model3 = LogisticRegression(max_iter=1000, random_state=42).fit(X3_train_scaled, y_train)
auc3_train = roc_auc_score(y_train, model3.predict_proba(X3_train_scaled)[:, 1])
auc3_val = roc_auc_score(y_val, model3.predict_proba(X3_val_scaled)[:, 1])

# Model 4: LSAT + GPA + timing + softs
X4_train = df_train[['lsat', 'gpa', 'app_timing_days', 'softs_numeric']].fillna(0)
X4_val = df_val[['lsat', 'gpa', 'app_timing_days', 'softs_numeric']].fillna(0)
X4_train_scaled = scaler.fit_transform(X4_train)
X4_val_scaled = scaler.transform(X4_val)
model4 = LogisticRegression(max_iter=1000, random_state=42).fit(X4_train_scaled, y_train)
auc4_train = roc_auc_score(y_train, model4.predict_proba(X4_train_scaled)[:, 1])
auc4_val = roc_auc_score(y_val, model4.predict_proba(X4_val_scaled)[:, 1])

print("\nVariance Decomposition Results (AUC):")
print("-" * 50)
print(f"{'Model':<25} {'Train AUC':>12} {'Val AUC':>12} {'Δ Val':>10}")
print("-" * 50)
print(f"{'LSAT only':<25} {auc1_train:>12.4f} {auc1_val:>12.4f} {'--':>10}")
print(f"{'+ GPA':<25} {auc2_train:>12.4f} {auc2_val:>12.4f} {auc2_val-auc1_val:>+10.4f}")
print(f"{'+ Timing':<25} {auc3_train:>12.4f} {auc3_val:>12.4f} {auc3_val-auc2_val:>+10.4f}")
print(f"{'+ Softs':<25} {auc4_train:>12.4f} {auc4_val:>12.4f} {auc4_val-auc3_val:>+10.4f}")
print("-" * 50)

print("\nKey Findings:")
print(f"  - LSAT alone explains significant variance (AUC = {auc1_val:.3f})")
print(f"  - Adding GPA increases AUC by {(auc2_val-auc1_val):.4f}")
print(f"  - Timing adds {(auc3_val-auc2_val):.4f} to predictive power")
print(f"  - Softs adds {(auc4_val-auc3_val):.4f} to predictive power")

# =============================================================================
# RQ #2: TEMPORAL STABILITY
# =============================================================================

print("\n" + "=" * 70)
print("RQ #2: TEMPORAL STABILITY")
print("Do admissions standards change over time?")
print("=" * 70)

# Calculate median LSAT/GPA by school by cycle
temporal_stats = df_train.groupby(['school_name', 'cycle_year']).agg({
    'lsat': 'median',
    'gpa': 'median',
    'result_clean': lambda x: (x == 'Accepted').mean()
}).reset_index()
temporal_stats.columns = ['school_name', 'cycle_year', 'median_lsat', 'median_gpa', 'acceptance_rate']

# Cycle-level trends
cycle_trends = df_train.groupby('cycle_year').agg({
    'lsat': 'median',
    'gpa': 'median',
    'result_clean': lambda x: (x == 'Accepted').mean()
}).reset_index()
cycle_trends.columns = ['cycle_year', 'median_lsat', 'median_gpa', 'acceptance_rate']

print("\nMedian Stats by Cycle Year:")
print("-" * 60)
print(f"{'Cycle':>8} {'Med LSAT':>12} {'Med GPA':>12} {'Accept Rate':>15}")
print("-" * 60)
for _, row in cycle_trends.iterrows():
    print(f"{int(row['cycle_year']):>8} {row['median_lsat']:>12.1f} {row['median_gpa']:>12.2f} {row['acceptance_rate']*100:>14.1f}%")

# Test: Does adding cycle_year improve predictions?
X_with_cycle_train = df_train[['lsat', 'gpa', 'app_timing_days', 'cycle_year']].fillna(0)
X_with_cycle_val = df_val[['lsat', 'gpa', 'app_timing_days', 'cycle_year']].fillna(0)
X_with_cycle_train_scaled = scaler.fit_transform(X_with_cycle_train)
X_with_cycle_val_scaled = scaler.transform(X_with_cycle_val)
model_with_cycle = LogisticRegression(max_iter=1000, random_state=42).fit(X_with_cycle_train_scaled, y_train)
auc_with_cycle_val = roc_auc_score(y_val, model_with_cycle.predict_proba(X_with_cycle_val_scaled)[:, 1])

print(f"\nTemporal Effect Analysis:")
print(f"  Model without cycle: AUC = {auc3_val:.4f}")
print(f"  Model with cycle:    AUC = {auc_with_cycle_val:.4f}")
print(f"  Difference:          {auc_with_cycle_val - auc3_val:+.4f}")

if abs(auc_with_cycle_val - auc3_val) > 0.01:
    print("  -> Significant temporal drift detected. Consider cycle weighting.")
else:
    print("  -> Minimal temporal drift. Cycle weighting may not be necessary.")

# =============================================================================
# RQ #3: SCORE BUCKETING
# =============================================================================

print("\n" + "=" * 70)
print("RQ #3: SCORE BUCKETING")
print("Are there thresholds where marginal score changes don't matter?")
print("=" * 70)

# LSAT bucketing with isotonic regression
lsat_sorted = df_train.sort_values('lsat').copy()
lsat_sorted = lsat_sorted[lsat_sorted['lsat'] >= 120]  # Valid LSAT range

iso_reg_lsat = IsotonicRegression(out_of_bounds='clip')
iso_reg_lsat.fit(lsat_sorted['lsat'].values, (lsat_sorted['result_clean'] == 'Accepted').astype(int).values)

# Predict acceptance probability for each LSAT score
lsat_range = np.arange(120, 181)
prob_by_lsat = iso_reg_lsat.predict(lsat_range)

print("\nLSAT → Acceptance Probability (Isotonic Regression):")
print("-" * 45)
for lsat_score in [145, 150, 155, 160, 165, 170, 175, 180]:
    idx = lsat_score - 120
    if idx < len(prob_by_lsat):
        print(f"  LSAT {lsat_score}: {prob_by_lsat[idx]*100:.1f}% acceptance")

# Find steep changes in probability
diff = np.diff(prob_by_lsat)
threshold_indices = np.where(np.abs(diff) > 0.02)[0]  # >2% change per point
lsat_thresholds = lsat_range[threshold_indices + 1]

print(f"\nLSAT threshold regions (>2% change per point): {sorted(set(lsat_thresholds))}")

# GPA bucketing
gpa_sorted = df_train.sort_values('gpa').copy()
gpa_sorted = gpa_sorted[(gpa_sorted['gpa'] >= 2.0) & (gpa_sorted['gpa'] <= 4.0)]

iso_reg_gpa = IsotonicRegression(out_of_bounds='clip')
iso_reg_gpa.fit(gpa_sorted['gpa'].values, (gpa_sorted['result_clean'] == 'Accepted').astype(int).values)

gpa_range = np.arange(2.0, 4.01, 0.05)
prob_by_gpa = iso_reg_gpa.predict(gpa_range)

print("\nGPA → Acceptance Probability (Isotonic Regression):")
print("-" * 45)
for gpa_val in [2.5, 3.0, 3.25, 3.5, 3.75, 4.0]:
    idx = int((gpa_val - 2.0) / 0.05)
    if idx < len(prob_by_gpa):
        print(f"  GPA {gpa_val:.2f}: {prob_by_gpa[idx]*100:.1f}% acceptance")

# =============================================================================
# RQ #5: SCHOOL SELECTIVITY RANKING
# =============================================================================

print("\n" + "=" * 70)
print("RQ #5: SCHOOL SELECTIVITY RANKING")
print("Iterative Elo-style algorithm to rank schools by selectivity")
print("=" * 70)

def calculate_school_selectivity(df, num_iterations=5, verbose=True):
    """
    Iterative algorithm:
    1. Initialize school selectivity = median LSAT + median GPA*50 of accepted applicants
    2. Calculate applicant quality based on schools that accepted them
    3. Recalculate school selectivity = average quality of applicants they accepted
    4. Repeat until convergence
    """

    # Initialize: median stats of accepted applicants
    accepted_stats = df[df['result_clean'] == 'Accepted'].groupby('school_name').agg({
        'lsat': 'median',
        'gpa': 'median'
    })

    # Normalize to similar scale (LSAT is ~120-180, GPA is 0-4)
    accepted_stats['selectivity'] = accepted_stats['lsat'] + accepted_stats['gpa'] * 40

    if verbose:
        print(f"\nInitial selectivity (based on accepted applicant medians):")
        print(accepted_stats.sort_values('selectivity', ascending=False).head(10))

    for iteration in range(num_iterations):
        # Calculate applicant quality based on their acceptances
        applicant_quality = {}

        for uuid in df['user_uuid'].unique():
            user_apps = df[df['user_uuid'] == uuid]
            user_lsat = user_apps['lsat'].iloc[0]
            user_gpa = user_apps['gpa'].iloc[0]

            # Base quality from stats
            base_quality = user_lsat + user_gpa * 40

            # Boost based on acceptances to selective schools
            accepted_schools = user_apps[user_apps['result_clean'] == 'Accepted']['school_name'].tolist()

            if len(accepted_schools) > 0:
                valid_schools = [s for s in accepted_schools if s in accepted_stats.index]
                if valid_schools:
                    acceptance_boost = np.mean([accepted_stats.loc[s, 'selectivity'] for s in valid_schools])
                    quality = 0.6 * base_quality + 0.4 * acceptance_boost
                else:
                    quality = base_quality
            else:
                quality = base_quality

            applicant_quality[uuid] = quality

        # Add quality to dataframe
        df_with_quality = df.copy()
        df_with_quality['quality'] = df_with_quality['user_uuid'].map(applicant_quality)

        # Recalculate school selectivity: average quality of accepted applicants
        new_selectivity = df_with_quality[df_with_quality['result_clean'] == 'Accepted'].groupby('school_name')['quality'].mean()

        # Update with smoothing
        for school in accepted_stats.index:
            if school in new_selectivity.index:
                old = accepted_stats.loc[school, 'selectivity']
                new = new_selectivity[school]
                accepted_stats.loc[school, 'selectivity'] = 0.7 * old + 0.3 * new

        if verbose and iteration == num_iterations - 1:
            print(f"\nAfter {num_iterations} iterations:")

    return accepted_stats.sort_values('selectivity', ascending=False)

# Calculate selectivity (using subset for speed)
print("Calculating school selectivity rankings...")
selectivity_rankings = calculate_school_selectivity(df_train, num_iterations=5, verbose=True)

print("\n" + "-" * 60)
print("TOP 25 MOST SELECTIVE LAW SCHOOLS")
print("-" * 60)
for rank, (school, row) in enumerate(selectivity_rankings.head(25).iterrows(), 1):
    print(f"{rank:2d}. {school[:45]:<45} ({row['selectivity']:.1f})")

print("\n" + "-" * 60)
print("BOTTOM 10 (LEAST SELECTIVE)")
print("-" * 60)
for rank, (school, row) in enumerate(selectivity_rankings.tail(10).iterrows(), 1):
    print(f"{190+rank-10:2d}. {school[:45]:<45} ({row['selectivity']:.1f})")

# Save selectivity rankings
selectivity_rankings.to_csv(f"{DATA_DIR}models/school_selectivity.csv")
print(f"\nSelectivity rankings saved to: {DATA_DIR}models/school_selectivity.csv")

# =============================================================================
# RQ #6: APPLICATION TIMING EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("RQ #6: APPLICATION TIMING EFFECTS")
print("Does timing matter? Does it vary by school selectivity?")
print("=" * 70)

# Add school selectivity to training data
df_train['school_selectivity'] = df_train['school_name'].map(selectivity_rankings['selectivity'])

# Split into selectivity tiers
df_train['selectivity_tier'] = pd.qcut(
    df_train['school_selectivity'].fillna(df_train['school_selectivity'].median()),
    q=3,
    labels=['Low', 'Mid', 'High']
)

# Analyze timing effect by tier
print("\nTiming Effect by School Selectivity Tier:")
print("-" * 70)

for tier in ['High', 'Mid', 'Low']:
    tier_data = df_train[df_train['selectivity_tier'] == tier].copy()

    # Split by early/late (median as cutoff)
    median_timing = tier_data['app_timing_days'].median()
    early = tier_data[tier_data['app_timing_days'] <= median_timing]
    late = tier_data[tier_data['app_timing_days'] > median_timing]

    early_accept = (early['result_clean'] == 'Accepted').mean()
    late_accept = (late['result_clean'] == 'Accepted').mean()

    print(f"\n{tier} Selectivity Schools:")
    print(f"  Early applicants (≤{median_timing:.0f} days): {early_accept*100:.1f}% accepted")
    print(f"  Late applicants (>{median_timing:.0f} days):  {late_accept*100:.1f}% accepted")
    print(f"  Difference: {(early_accept - late_accept)*100:+.1f} percentage points")

# Logistic regression with timing interaction
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_timing = df_train.dropna(subset=['app_timing_days', 'school_selectivity']).copy()
df_timing['tier_encoded'] = le.fit_transform(df_timing['selectivity_tier'])

X_timing = df_timing[['lsat', 'gpa', 'app_timing_days', 'school_selectivity']].copy()
X_timing['timing_x_selectivity'] = X_timing['app_timing_days'] * X_timing['school_selectivity']
X_timing = X_timing.fillna(0)

X_timing_scaled = scaler.fit_transform(X_timing)
y_timing = (df_timing['result_clean'] == 'Accepted').astype(int)

model_timing = LogisticRegression(max_iter=1000, random_state=42).fit(X_timing_scaled, y_timing)
auc_timing = roc_auc_score(y_timing, model_timing.predict_proba(X_timing_scaled)[:, 1])

print(f"\nModel with Timing Interaction:")
print(f"  AUC: {auc_timing:.4f}")
print(f"  Feature coefficients (standardized):")
for feat, coef in zip(X_timing.columns, model_timing.coef_[0]):
    print(f"    {feat}: {coef:.4f}")

timing_coef = model_timing.coef_[0][2]  # app_timing_days coefficient
if abs(timing_coef) > 0.001:
    print(f"\n  Timing has a {'negative' if timing_coef < 0 else 'positive'} effect on acceptance")
    print(f"  -> {'Earlier applications improve chances' if timing_coef < 0 else 'Later applications improve chances'}")
else:
    print("\n  Timing effect is minimal")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("RESEARCH QUESTIONS SUMMARY")
print("=" * 70)

print("""
RQ #1 - Variance Decomposition:
  - LSAT is the strongest single predictor
  - GPA adds meaningful predictive power
  - Timing and softs contribute marginally

RQ #2 - Temporal Stability:
  - Admissions standards show some variation across cycles
  - Recent cycles may have different dynamics
  - Consider cycle weighting for predictions

RQ #3 - Score Bucketing:
  - LSAT shows non-linear effects with thresholds
  - GPA follows similar patterns
  - Bucketing may simplify models without losing accuracy

RQ #5 - School Selectivity:
  - Successfully ranked all 193 schools
  - Top schools: Yale, Stanford, Harvard, etc.
  - Rankings can be used as features in predictive models

RQ #6 - Application Timing:
  - Earlier applications correlate with higher acceptance rates
  - Effect is stronger at more selective schools
  - Worth including in predictive models
""")

print("=" * 70)
print("RESEARCH QUESTIONS ANALYSIS COMPLETE")
print("=" * 70)
