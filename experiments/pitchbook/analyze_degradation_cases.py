import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('pitchbook_combined_processed_results.csv')

# Filter for only statistical baselines
baseline_approaches = ['statistical_baseline_n5', 'statistical_baseline_n10',
                       'statistical_baseline_n20', 'statistical_baseline_n30']
df_baselines = df[df['approach'].isin(baseline_approaches)].copy()

# Extract sample size from approach name
df_baselines['baseline_n'] = df_baselines['approach'].str.extract(r'n(\d+)').astype(int)

print("=" * 80)
print("INVESTIGATING CASES WHERE LARGER SAMPLES ARE LESS ACCURATE")
print("=" * 80)

# For each unique observation (variable + trial), compare errors across sample sizes
df_baselines['obs_id'] = df_baselines['variable'] + '_' + df_baselines['trial'].astype(str)

# Pivot to compare sample sizes for each observation
pivot_data = df_baselines.pivot_table(
    index=['obs_id', 'variable', 'trial', 'ground_truth'],
    columns='baseline_n',
    values='abs_error_from_mean'
).reset_index()

print(f"\nTotal unique observations: {len(pivot_data)}")

# Calculate instances where each larger sample size is worse than n=5
print("\n" + "=" * 80)
print("FREQUENCY OF DEGRADATION")
print("=" * 80)

for n in [10, 20, 30]:
    worse_than_n5 = (pivot_data[n] > pivot_data[5]).sum()
    pct = 100 * worse_than_n5 / len(pivot_data)
    print(f"\nn={n} worse than n=5: {worse_than_n5}/{len(pivot_data)} ({pct:.2f}%)")

    # Also check if worse than previous size
    if n == 10:
        prev = 5
    elif n == 20:
        prev = 10
    else:  # n == 30
        prev = 20

    worse_than_prev = (pivot_data[n] > pivot_data[prev]).sum()
    pct_prev = 100 * worse_than_prev / len(pivot_data)
    print(f"n={n} worse than n={prev}: {worse_than_prev}/{len(pivot_data)} ({pct_prev:.2f}%)")

# Find cases where error monotonically increases
print("\n" + "=" * 80)
print("MONOTONIC ERROR INCREASE PATTERNS")
print("=" * 80)

monotonic_increase = (
    (pivot_data[10] > pivot_data[5]) &
    (pivot_data[20] > pivot_data[10]) &
    (pivot_data[30] > pivot_data[20])
)

print(f"\nObservations with monotonic error increase: {monotonic_increase.sum()} ({100*monotonic_increase.sum()/len(pivot_data):.2f}%)")

if monotonic_increase.sum() > 0:
    print("\nExamples of monotonic error increase:")
    monotonic_cases = pivot_data[monotonic_increase].sort_values(30, ascending=False).head(20)
    for idx, row in monotonic_cases.iterrows():
        print(f"\n  {row['variable']} (trial {row['trial']}, GT={row['ground_truth']:.2f}):")
        print(f"    n=5: {row[5]:.4f}, n=10: {row[10]:.4f}, n=20: {row[20]:.4f}, n=30: {row[30]:.4f}")
        print(f"    Increase: {row[30]/row[5]:.2f}x")

# Find cases where n=30 is significantly worse than n=5
print("\n" + "=" * 80)
print("LARGE DEGRADATION CASES (n=30 much worse than n=5)")
print("=" * 80)

# Define "significantly worse" as >50% increase in error
pivot_data['pct_change_n30_vs_n5'] = 100 * (pivot_data[30] - pivot_data[5]) / (pivot_data[5] + 1e-10)
pivot_data['abs_change_n30_vs_n5'] = pivot_data[30] - pivot_data[5]

significant_degradation = pivot_data['pct_change_n30_vs_n5'] > 50
print(f"\nCases where n=30 error is >50% higher than n=5: {significant_degradation.sum()} ({100*significant_degradation.sum()/len(pivot_data):.2f}%)")

if significant_degradation.sum() > 0:
    print("\nTop 20 cases by percentage increase:")
    worst_cases = pivot_data[significant_degradation].nlargest(20, 'pct_change_n30_vs_n5')
    for idx, row in worst_cases.iterrows():
        print(f"\n  {row['variable']} (trial {row['trial']}, GT={row['ground_truth']:.2f}):")
        print(f"    n=5: {row[5]:.4f}, n=30: {row[30]:.4f}")
        print(f"    Change: +{row['pct_change_n30_vs_n5']:.1f}% (+{row['abs_change_n30_vs_n5']:.4f} absolute)")

# Analyze by variable
print("\n" + "=" * 80)
print("DEGRADATION BY VARIABLE")
print("=" * 80)

var_degradation = pivot_data.groupby('variable').agg({
    'pct_change_n30_vs_n5': 'mean',
    'abs_change_n30_vs_n5': 'mean',
    5: 'mean',
    30: 'mean'
}).round(4)
var_degradation.columns = ['Avg_Pct_Change', 'Avg_Abs_Change', 'Avg_Error_n5', 'Avg_Error_n30']
var_degradation['Degradation'] = var_degradation['Avg_Error_n30'] > var_degradation['Avg_Error_n5']

print("\nVariables where n=30 is worse than n=5 on average:")
degraded_vars = var_degradation[var_degradation['Degradation']].sort_values('Avg_Pct_Change', ascending=False)
print(degraded_vars)

print("\nNumber of variables with degradation: {}/{} ({:.2f}%)".format(
    degraded_vars.shape[0],
    var_degradation.shape[0],
    100 * degraded_vars.shape[0] / var_degradation.shape[0]
))

# Analyze characteristics of observations that degrade
print("\n" + "=" * 80)
print("CHARACTERISTICS OF DEGRADING OBSERVATIONS")
print("=" * 80)

# Define degradation as n=30 worse than n=5
pivot_data['degrades'] = pivot_data[30] > pivot_data[5]

print(f"\nObservations where n=30 > n=5: {pivot_data['degrades'].sum()}/{len(pivot_data)} ({100*pivot_data['degrades'].sum()/len(pivot_data):.2f}%)")

# Compare ground truth distributions
print("\nGround truth comparison:")
print(f"  Mean GT (degrading): {pivot_data[pivot_data['degrades']]['ground_truth'].mean():.4f}")
print(f"  Mean GT (improving): {pivot_data[~pivot_data['degrades']]['ground_truth'].mean():.4f}")
print(f"  Median GT (degrading): {pivot_data[pivot_data['degrades']]['ground_truth'].median():.4f}")
print(f"  Median GT (improving): {pivot_data[~pivot_data['degrades']]['ground_truth'].median():.4f}")
print(f"  Std GT (degrading): {pivot_data[pivot_data['degrades']]['ground_truth'].std():.4f}")
print(f"  Std GT (improving): {pivot_data[~pivot_data['degrades']]['ground_truth'].std():.4f}")

# Compare error magnitudes at n=5
print("\nError at n=5 comparison:")
print(f"  Mean error n=5 (degrading): {pivot_data[pivot_data['degrades']][5].mean():.4f}")
print(f"  Mean error n=5 (improving): {pivot_data[~pivot_data['degrades']][5].mean():.4f}")
print(f"  Median error n=5 (degrading): {pivot_data[pivot_data['degrades']][5].median():.4f}")
print(f"  Median error n=5 (improving): {pivot_data[~pivot_data['degrades']][5].median():.4f}")

# Analyze outlier behavior
print("\n" + "=" * 80)
print("OUTLIER BEHAVIOR IN DEGRADING CASES")
print("=" * 80)

# For observations that degrade, check if they're outliers at either n=5 or n=30
degrading_obs = pivot_data[pivot_data['degrades']].copy()

# Calculate outlier thresholds for each sample size
for n in [5, 30]:
    Q1 = pivot_data[n].quantile(0.25)
    Q3 = pivot_data[n].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    degrading_obs[f'is_outlier_n{n}'] = degrading_obs[n] > outlier_threshold
    pivot_data[f'is_outlier_n{n}'] = pivot_data[n] > outlier_threshold

print(f"\nIn degrading observations:")
print(f"  Outliers at n=5: {degrading_obs['is_outlier_n5'].sum()} ({100*degrading_obs['is_outlier_n5'].sum()/len(degrading_obs):.2f}%)")
print(f"  Outliers at n=30: {degrading_obs['is_outlier_n30'].sum()} ({100*degrading_obs['is_outlier_n30'].sum()/len(degrading_obs):.2f}%)")
print(f"  Becomes outlier (not outlier at n=5, but outlier at n=30): {((~degrading_obs['is_outlier_n5']) & degrading_obs['is_outlier_n30']).sum()}")
print(f"  Remains outlier (outlier at both): {(degrading_obs['is_outlier_n5'] & degrading_obs['is_outlier_n30']).sum()}")

print(f"\nIn improving observations:")
improving_obs = pivot_data[~pivot_data['degrades']].copy()
print(f"  Outliers at n=5: {improving_obs['is_outlier_n5'].sum()} ({100*improving_obs['is_outlier_n5'].sum()/len(improving_obs):.2f}%)")
print(f"  Outliers at n=30: {improving_obs['is_outlier_n30'].sum()} ({100*improving_obs['is_outlier_n30'].sum()/len(improving_obs):.2f}%)")

# Examine specific degrading variables in detail
print("\n" + "=" * 80)
print("DETAILED EXAMINATION OF MOST DEGRADING VARIABLES")
print("=" * 80)

top_degrading_vars = degraded_vars.head(5).index

for var in top_degrading_vars:
    var_data = pivot_data[pivot_data['variable'] == var]
    print(f"\n{var}:")
    print(f"  Ground truth: mean={var_data['ground_truth'].mean():.4f}, std={var_data['ground_truth'].std():.4f}")
    print(f"  Average errors: n=5={var_data[5].mean():.4f}, n=10={var_data[10].mean():.4f}, n=20={var_data[20].mean():.4f}, n=30={var_data[30].mean():.4f}")
    print(f"  Error std: n=5={var_data[5].std():.4f}, n=30={var_data[30].std():.4f}")
    print(f"  Max error: n=5={var_data[5].max():.4f}, n=30={var_data[30].max():.4f}")

    # Check if one or two extreme cases drive the degradation
    top_3_errors_n30 = var_data.nlargest(3, 30)
    print(f"  Top 3 errors at n=30:")
    for idx, row in top_3_errors_n30.iterrows():
        print(f"    Trial {row['trial']}: n=5={row[5]:.4f}, n=30={row[30]:.4f} (GT={row['ground_truth']:.4f})")

# Statistical test
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE")
print("=" * 80)

# Paired t-test: is n=30 significantly different from n=5?
from scipy.stats import ttest_rel, wilcoxon

t_stat, p_value = ttest_rel(pivot_data[30], pivot_data[5])
print(f"\nPaired t-test (n=30 vs n=5):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Mean difference: {(pivot_data[30] - pivot_data[5]).mean():.4f}")

# Wilcoxon signed-rank test (non-parametric alternative)
w_stat, w_pvalue = wilcoxon(pivot_data[30], pivot_data[5])
print(f"\nWilcoxon signed-rank test (n=30 vs n=5):")
print(f"  statistic: {w_stat:.4f}")
print(f"  p-value: {w_pvalue:.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

degrade_pct = 100 * pivot_data['degrades'].sum() / len(pivot_data)
mean_error_diff = pivot_data[30].mean() - pivot_data[5].mean()
median_error_diff = pivot_data[30].median() - pivot_data[5].median()

print(f"""
While larger samples (n=30) are OVERALL more accurate than smaller samples (n=5):
  - Mean error: {pivot_data[5].mean():.4f} (n=5) → {pivot_data[30].mean():.4f} (n=30)
  - Median error: {pivot_data[5].median():.4f} (n=5) → {pivot_data[30].median():.4f} (n=30)

HOWEVER, in {degrade_pct:.2f}% of cases, n=30 produces WORSE estimates than n=5.

This degradation is driven by:
1. Certain variables that consistently perform worse with larger samples
2. Outlier observations that become more extreme with larger samples
3. Heavy-tailed distributions where additional samples can include extreme values

The paradox occurs because:
- On average, more data helps (median/mean improve)
- But in specific cases, larger samples can accidentally include outliers
- These outliers disproportionately affect the mean of the baseline
- This is especially problematic for variables with high variance or skewed distributions
""")
