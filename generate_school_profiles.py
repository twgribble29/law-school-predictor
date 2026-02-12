"""
Generate School Profiles Data
==============================
Pre-computes per-school scatter data (downsampled) and admissions metrics
(Holistic Review, Splitter Friendliness, Reverse Splitter Friendliness)
from cleaned_data_full.csv.

Output: models/school_profiles.json
"""

import pandas as pd
import numpy as np
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'cleaned_data_full.csv')
STATS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'school_stats.json')
SELECTIVITY_PATH = os.path.join(os.path.dirname(__file__), 'models', 'school_selectivity.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'school_profiles.json')

MAX_POINTS = 500
MIN_BIN_SIZE = 5
MIN_SPLITTER_COUNT = 10


def load_data():
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    # Keep only rows with valid LSAT, GPA, school, and result
    df = df.dropna(subset=['lsat', 'gpa', 'school_name', 'result_clean'])
    df = df[df['result_clean'].isin(['Accepted', 'Rejected', 'Waitlisted'])]
    print(f"  {len(df):,} valid records across {df['school_name'].nunique()} schools")
    return df


def load_school_stats():
    with open(STATS_PATH, 'r') as f:
        return json.load(f)


def downsample_scatter(school_df, max_points=MAX_POINTS):
    """Stratified downsample to ~max_points, balanced across outcomes."""
    accepted = school_df[school_df['result_clean'] == 'Accepted']
    rejected = school_df[school_df['result_clean'] == 'Rejected']
    waitlisted = school_df[school_df['result_clean'] == 'Waitlisted']

    total = len(school_df)
    if total <= max_points:
        sampled = school_df
    else:
        # Allocate proportionally but ensure at least some of each
        n_acc = min(len(accepted), max(10, int(max_points * len(accepted) / total)))
        n_rej = min(len(rejected), max(10, int(max_points * len(rejected) / total)))
        n_wl = min(len(waitlisted), max(5, int(max_points * len(waitlisted) / total)))

        # Cap total
        while n_acc + n_rej + n_wl > max_points:
            if n_acc > n_rej and n_acc > n_wl:
                n_acc -= 1
            elif n_rej > n_wl:
                n_rej -= 1
            else:
                n_wl -= 1

        parts = []
        if n_acc > 0 and len(accepted) > 0:
            parts.append(accepted.sample(n=min(n_acc, len(accepted)), random_state=42))
        if n_rej > 0 and len(rejected) > 0:
            parts.append(rejected.sample(n=min(n_rej, len(rejected)), random_state=42))
        if n_wl > 0 and len(waitlisted) > 0:
            parts.append(waitlisted.sample(n=min(n_wl, len(waitlisted)), random_state=42))

        sampled = pd.concat(parts) if parts else school_df.head(0)

    # Compact encoding: l=lsat, g=gpa, r=A/W/R
    result_map = {'Accepted': 'A', 'Waitlisted': 'W', 'Rejected': 'R'}
    points = []
    for _, row in sampled.iterrows():
        points.append({
            'l': round(float(row['lsat']), 0),
            'g': round(float(row['gpa']), 2),
            'r': result_map[row['result_clean']]
        })
    return points


def compute_holistic_score(school_df):
    """
    Holistic Review Score: measures outcome variance for similar stats.
    Higher = outcomes are less predictable from stats alone = more holistic.
    """
    df = school_df.copy()
    df['is_accepted'] = (df['result_clean'] == 'Accepted').astype(int)
    df['lsat_bin'] = (df['lsat'] // 2) * 2
    df['gpa_bin'] = (df['gpa'] * 10).round() / 10  # Round to nearest 0.1

    variances = []
    weights = []

    for (_, _), group in df.groupby(['lsat_bin', 'gpa_bin']):
        if len(group) < MIN_BIN_SIZE:
            continue
        p = group['is_accepted'].mean()
        variance = p * (1 - p)  # Bernoulli variance, max at p=0.5
        variances.append(variance)
        weights.append(len(group))

    if not variances:
        return 0.0
    return float(np.average(variances, weights=weights))


def compute_splitter_friendliness(school_df, stats):
    """
    Splitter Friendliness: do high-LSAT/low-GPA applicants get in more than expected?
    Returns ratio of splitter acceptance rate to baseline acceptance rate.
    """
    if not stats or 'lsat_75' not in stats or 'gpa_25' not in stats:
        return np.nan

    lsat_75 = stats['lsat_75']
    gpa_25 = stats['gpa_25']

    splitters = school_df[
        (school_df['lsat'] > lsat_75) & (school_df['gpa'] < gpa_25)
    ]

    if len(splitters) < MIN_SPLITTER_COUNT:
        return np.nan

    baseline_rate = (school_df['result_clean'] == 'Accepted').mean()
    if baseline_rate == 0:
        return np.nan

    splitter_rate = (splitters['result_clean'] == 'Accepted').mean()
    return float(splitter_rate / baseline_rate)


def compute_reverse_splitter_friendliness(school_df, stats):
    """
    Reverse Splitter Friendliness: do high-GPA/low-LSAT applicants get in more than expected?
    """
    if not stats or 'gpa_75' not in stats or 'lsat_25' not in stats:
        return np.nan

    gpa_75 = stats['gpa_75']
    lsat_25 = stats['lsat_25']

    reverse_splitters = school_df[
        (school_df['gpa'] > gpa_75) & (school_df['lsat'] < lsat_25)
    ]

    if len(reverse_splitters) < MIN_SPLITTER_COUNT:
        return np.nan

    baseline_rate = (school_df['result_clean'] == 'Accepted').mean()
    if baseline_rate == 0:
        return np.nan

    reverse_rate = (reverse_splitters['result_clean'] == 'Accepted').mean()
    return float(reverse_rate / baseline_rate)


def scores_to_percentiles(scores_dict):
    """Convert raw scores to 0-100 percentile rankings across schools."""
    # Filter out NaN values
    valid = {k: v for k, v in scores_dict.items() if not np.isnan(v)}
    if not valid:
        return {k: None for k in scores_dict}

    sorted_schools = sorted(valid.keys(), key=lambda k: valid[k])
    n = len(sorted_schools)

    percentiles = {}
    for i, school in enumerate(sorted_schools):
        percentiles[school] = round(100 * i / max(n - 1, 1))

    # Schools with NaN get None
    for school in scores_dict:
        if school not in percentiles:
            percentiles[school] = None

    return percentiles


def main():
    df = load_data()
    school_stats = load_school_stats()
    selectivity = pd.read_csv(SELECTIVITY_PATH, index_col=0)
    schools = selectivity.index.tolist()

    print(f"Processing {len(schools)} schools...")

    # Phase 1: Compute raw scores for all schools
    holistic_scores = {}
    splitter_scores = {}
    reverse_scores = {}
    scatter_data = {}
    school_totals = {}

    for i, school in enumerate(schools):
        school_df = df[df['school_name'] == school]
        if len(school_df) < 20:
            print(f"  Skipping {school} ({len(school_df)} records)")
            holistic_scores[school] = np.nan
            splitter_scores[school] = np.nan
            reverse_scores[school] = np.nan
            scatter_data[school] = []
            school_totals[school] = len(school_df)
            continue

        stats = school_stats.get(school, {})

        # Scatter data
        scatter_data[school] = downsample_scatter(school_df)

        # Metrics
        holistic_scores[school] = compute_holistic_score(school_df)
        splitter_scores[school] = compute_splitter_friendliness(school_df, stats)
        reverse_scores[school] = compute_reverse_splitter_friendliness(school_df, stats)
        school_totals[school] = len(school_df)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(schools)} schools")

    # Phase 2: Convert to percentiles
    print("Converting scores to percentiles...")
    holistic_pct = scores_to_percentiles(holistic_scores)
    splitter_pct = scores_to_percentiles(splitter_scores)
    reverse_pct = scores_to_percentiles(reverse_scores)

    # Phase 3: Build output
    profiles = {}
    for school in schools:
        profiles[school] = {
            'scatter': scatter_data[school],
            'metrics': {
                'holistic': holistic_pct[school],
                'splitter': splitter_pct[school],
                'reverse_splitter': reverse_pct[school]
            },
            'n_total': school_totals[school]
        }

    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(profiles, f, separators=(',', ':'))

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"Schools: {len(profiles)}")

    # Print some interesting results
    print("\nTop 10 Most Holistic Schools:")
    top_holistic = sorted(
        [(s, profiles[s]['metrics']['holistic']) for s in schools if profiles[s]['metrics']['holistic'] is not None],
        key=lambda x: x[1], reverse=True
    )
    for school, pct in top_holistic[:10]:
        print(f"  {pct:3d}th pct - {school}")

    print("\nTop 10 Splitter-Friendly Schools:")
    top_splitter = sorted(
        [(s, profiles[s]['metrics']['splitter']) for s in schools if profiles[s]['metrics']['splitter'] is not None],
        key=lambda x: x[1], reverse=True
    )
    for school, pct in top_splitter[:10]:
        print(f"  {pct:3d}th pct - {school}")

    print("\nTop 10 Reverse-Splitter-Friendly Schools:")
    top_reverse = sorted(
        [(s, profiles[s]['metrics']['reverse_splitter']) for s in schools if profiles[s]['metrics']['reverse_splitter'] is not None],
        key=lambda x: x[1], reverse=True
    )
    for school, pct in top_reverse[:10]:
        print(f"  {pct:3d}th pct - {school}")


if __name__ == '__main__':
    main()
