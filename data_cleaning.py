"""
Law School Admissions Prediction Model - Data Cleaning Pipeline
================================================================
This script loads and cleans the LSD.Law data according to specifications.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 60)
print("LAW SCHOOL ADMISSIONS DATA CLEANING PIPELINE")
print("=" * 60)

# Load data, skip attribution row
DATA_PATH = "/Users/tygribble/Desktop/Law_Data/lsdata.csv"
print(f"\nLoading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, skiprows=1)
print(f"Initial rows: {len(df):,}")
print(f"Initial columns: {list(df.columns)}")

# =============================================================================
# STEP 2: EXCLUSIONS (Apply in Order)
# =============================================================================

print("\n" + "=" * 60)
print("APPLYING EXCLUSIONS")
print("=" * 60)

# 2.1 Missing LSAT
initial_count = len(df)
df = df[df['lsat'].notna()]
print(f"After removing missing LSAT: {len(df):,} (removed {initial_count - len(df):,})")

# 2.2 Missing GPA
initial_count = len(df)
df = df[df['gpa'].notna()]
print(f"After removing missing GPA: {len(df):,} (removed {initial_count - len(df):,})")

# 2.3 Missing complete_at
initial_count = len(df)
df = df[df['complete_at'].notna()]
print(f"After removing missing complete_at: {len(df):,} (removed {initial_count - len(df):,})")

# 2.4 Standardize Results
def clean_result(result):
    """Standardize result to Accepted/Rejected/Waitlisted or None for exclusion"""
    if pd.isna(result):
        return None
    result_lower = str(result).lower().strip()

    # Withdrawals - exclude
    if 'withdrawn' in result_lower:
        return None

    # Accepted (direct accepts only - WL then accepted treated as WL)
    if result_lower == 'accepted':
        return 'Accepted'

    # WL then accepted - treat initial decision as WL
    if 'wl, accepted' in result_lower or 'wl_accepted' in result_lower:
        return 'Waitlisted'

    # Rejected (direct rejects only)
    if result_lower == 'rejected':
        return 'Rejected'

    # WL then rejected - treat initial decision as WL
    if 'wl, rejected' in result_lower or 'wl_rejected' in result_lower:
        return 'Waitlisted'

    # Waitlisted
    if result_lower in ['waitlisted', 'wl']:
        return 'Waitlisted'

    # Hold treated as Waitlisted
    if 'hold' in result_lower and 'withdrawn' not in result_lower:
        return 'Waitlisted'

    return None

initial_count = len(df)
df['result_clean'] = df['result'].apply(clean_result)
df = df[df['result_clean'].notna()]
print(f"After standardizing results: {len(df):,} (removed {initial_count - len(df):,})")

# Show result distribution
print("\nResult distribution:")
print(df['result_clean'].value_counts())

# 2.5 School Size Filter (≥100 applications)
school_counts = df['school_name'].value_counts()
valid_schools = school_counts[school_counts >= 100].index
initial_count = len(df)
initial_schools = df['school_name'].nunique()
df = df[df['school_name'].isin(valid_schools)]
print(f"\nAfter school filter (≥100 apps): {len(df):,} (removed {initial_count - len(df):,})")
print(f"Schools retained: {df['school_name'].nunique()} (removed {initial_schools - df['school_name'].nunique()})")

# =============================================================================
# STEP 3: CREATE DERIVED VARIABLES
# =============================================================================

print("\n" + "=" * 60)
print("CREATING DERIVED VARIABLES")
print("=" * 60)

# 3.1 Cycle year
df['cycle_year'] = df['matriculating_year'] - 1
print(f"Cycle years present: {sorted(df['cycle_year'].dropna().unique())}")

# 3.2 Work experience dummy
df['has_work_experience'] = (df['years_out'].fillna(0) != 0).astype(int)
print(f"Has work experience: {df['has_work_experience'].sum():,} ({df['has_work_experience'].mean()*100:.1f}%)")

# 3.3 Parse dates and calculate timing
df['complete_date'] = pd.to_datetime(df['complete_at'], format='%Y-%m-%d', errors='coerce')
# Try alternative format if first doesn't work
mask = df['complete_date'].isna()
df.loc[mask, 'complete_date'] = pd.to_datetime(df.loc[mask, 'complete_at'], format='%m/%d/%Y', errors='coerce')

def days_since_cycle_start(row):
    """Calculate days since Sept 1 of cycle year"""
    if pd.isna(row['complete_date']) or pd.isna(row['cycle_year']):
        return None
    try:
        cycle_start = datetime(int(row['cycle_year']), 9, 1)
        return (row['complete_date'] - cycle_start).days
    except:
        return None

df['app_timing_days'] = df.apply(days_since_cycle_start, axis=1)
print(f"Application timing calculated: {df['app_timing_days'].notna().sum():,} valid values")
print(f"Timing stats: min={df['app_timing_days'].min():.0f}, median={df['app_timing_days'].median():.0f}, max={df['app_timing_days'].max():.0f}")

# 3.4 Standardize softs
def parse_softs(soft_val):
    """Parse softs tier (T1-T4)"""
    if pd.isna(soft_val):
        return None
    soft_str = str(soft_val).lower().strip()
    if 't1' in soft_str:
        return 'T1'
    elif 't2' in soft_str:
        return 'T2'
    elif 't3' in soft_str:
        return 'T3'
    elif 't4' in soft_str:
        return 'T4'
    return None

df['softs_tier'] = df['softs'].apply(parse_softs)
print(f"\nSofts tier distribution:")
print(df['softs_tier'].value_counts(dropna=False))

# =============================================================================
# STEP 4: STRATIFY BY URM STATUS
# =============================================================================

print("\n" + "=" * 60)
print("STRATIFYING BY URM STATUS")
print("=" * 60)

df['is_urm'] = df['urm'].fillna(False).astype(bool)

df_non_urm = df[~df['is_urm']].copy()
df_urm = df[df['is_urm']].copy()

print(f"Non-URM applications: {len(df_non_urm):,} ({len(df_non_urm)/len(df)*100:.1f}%)")
print(f"  - Unique applicants: {df_non_urm['user_uuid'].nunique():,}")
print(f"  - Acceptance rate: {(df_non_urm['result_clean'] == 'Accepted').mean()*100:.1f}%")

print(f"\nURM applications: {len(df_urm):,} ({len(df_urm)/len(df)*100:.1f}%)")
print(f"  - Unique applicants: {df_urm['user_uuid'].nunique():,}")
print(f"  - Acceptance rate: {(df_urm['result_clean'] == 'Accepted').mean()*100:.1f}%")

# =============================================================================
# STEP 5: TRAIN/VALIDATION/TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("CREATING TRAIN/VALIDATION/TEST SPLITS")
print("=" * 60)

from sklearn.model_selection import train_test_split

# --- Non-URM Split ---
# Hold out entire 2025 cycle for temporal validation
df_non_urm_train_temp = df_non_urm[df_non_urm['cycle_year'] < 2025].copy()
df_non_urm_test_2025 = df_non_urm[df_non_urm['cycle_year'] == 2025].copy()

# Further split training data: 80% train, 20% validation (applicant-level split)
train_uuids, val_uuids = train_test_split(
    df_non_urm_train_temp['user_uuid'].unique(),
    test_size=0.2,
    random_state=42
)

df_non_urm_train = df_non_urm_train_temp[df_non_urm_train_temp['user_uuid'].isin(train_uuids)].copy()
df_non_urm_val = df_non_urm_train_temp[df_non_urm_train_temp['user_uuid'].isin(val_uuids)].copy()

print("Non-URM Splits:")
print(f"  Training: {len(df_non_urm_train):,} applications from {len(train_uuids):,} applicants")
print(f"  Validation: {len(df_non_urm_val):,} applications from {len(val_uuids):,} applicants")
print(f"  2025 Test: {len(df_non_urm_test_2025):,} applications")

# --- URM Split ---
df_urm_train_temp = df_urm[df_urm['cycle_year'] < 2025].copy()
df_urm_test_2025 = df_urm[df_urm['cycle_year'] == 2025].copy()

urm_train_uuids, urm_val_uuids = train_test_split(
    df_urm_train_temp['user_uuid'].unique(),
    test_size=0.2,
    random_state=42
)

df_urm_train = df_urm_train_temp[df_urm_train_temp['user_uuid'].isin(urm_train_uuids)].copy()
df_urm_val = df_urm_train_temp[df_urm_train_temp['user_uuid'].isin(urm_val_uuids)].copy()

print("\nURM Splits:")
print(f"  Training: {len(df_urm_train):,} applications from {len(urm_train_uuids):,} applicants")
print(f"  Validation: {len(df_urm_val):,} applications from {len(urm_val_uuids):,} applicants")
print(f"  2025 Test: {len(df_urm_test_2025):,} applications")

# =============================================================================
# SAVE CLEANED DATA
# =============================================================================

print("\n" + "=" * 60)
print("SAVING CLEANED DATA")
print("=" * 60)

OUTPUT_DIR = "/Users/tygribble/Desktop/Law_Data/law_school_predictor/"

# Save all splits
df_non_urm_train.to_csv(f"{OUTPUT_DIR}non_urm_train.csv", index=False)
df_non_urm_val.to_csv(f"{OUTPUT_DIR}non_urm_val.csv", index=False)
df_non_urm_test_2025.to_csv(f"{OUTPUT_DIR}non_urm_test_2025.csv", index=False)

df_urm_train.to_csv(f"{OUTPUT_DIR}urm_train.csv", index=False)
df_urm_val.to_csv(f"{OUTPUT_DIR}urm_val.csv", index=False)
df_urm_test_2025.to_csv(f"{OUTPUT_DIR}urm_test_2025.csv", index=False)

# Save full cleaned dataset
df.to_csv(f"{OUTPUT_DIR}cleaned_data_full.csv", index=False)

print("Saved files:")
print(f"  - non_urm_train.csv ({len(df_non_urm_train):,} rows)")
print(f"  - non_urm_val.csv ({len(df_non_urm_val):,} rows)")
print(f"  - non_urm_test_2025.csv ({len(df_non_urm_test_2025):,} rows)")
print(f"  - urm_train.csv ({len(df_urm_train):,} rows)")
print(f"  - urm_val.csv ({len(df_urm_val):,} rows)")
print(f"  - urm_test_2025.csv ({len(df_urm_test_2025):,} rows)")
print(f"  - cleaned_data_full.csv ({len(df):,} rows)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL CLEAN DATASET SUMMARY")
print("=" * 60)
print(f"Total applications: {len(df):,}")
print(f"Unique applicants: {df['user_uuid'].nunique():,}")
print(f"Unique schools: {df['school_name'].nunique()}")
print(f"Cycles: {int(df['cycle_year'].min())} - {int(df['cycle_year'].max())}")

print("\nBy Result:")
for result in ['Accepted', 'Waitlisted', 'Rejected']:
    count = (df['result_clean'] == result).sum()
    pct = count / len(df) * 100
    print(f"  {result}: {count:,} ({pct:.1f}%)")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)
