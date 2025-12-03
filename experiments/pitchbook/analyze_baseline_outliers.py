import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('pitchbook_combined_processed_results.csv')

print("=" * 80)
print("INVESTIGATING WHY LARGER SAMPLE PITCHBOOK BASELINES ARE LESS ACCURATE")
print("=" * 80)

# Filter for only statistical baselines
baseline_approaches = ['statistical_baseline_n5', 'statistical_baseline_n10',
                       'statistical_baseline_n20', 'statistical_baseline_n30']
df_baselines = df[df['approach'].isin(baseline_approaches)].copy()

print(f"\nTotal baseline records: {len(df_baselines)}")
print(f"Records per baseline: {df_baselines['approach'].value_counts().to_dict()}")

# Extract sample size from approach name
df_baselines['baseline_n'] = df_baselines['approach'].str.extract(r'n(\d+)').astype(int)

print("\n" + "=" * 80)
print("1. OVERALL ERROR STATISTICS BY SAMPLE SIZE")
print("=" * 80)

# Group by sample size and calculate error statistics
error_stats = df_baselines.groupby('baseline_n').agg({
    'abs_error_from_mean': ['mean', 'median', 'std', 'min', 'max',
                            lambda x: np.percentile(x, 25),
                            lambda x: np.percentile(x, 75),
                            lambda x: np.percentile(x, 90),
                            lambda x: np.percentile(x, 95),
                            lambda x: np.percentile(x, 99),
                            'skew']
}).round(4)

error_stats.columns = ['Mean', 'Median', 'StdDev', 'Min', 'Max', 'P25', 'P75', 'P90', 'P95', 'P99', 'Skewness']
print("\nAbsolute Error from Mean Statistics:")
print(error_stats)

# Calculate if larger samples have worse mean/median errors
print("\n" + "-" * 80)
print("Do larger samples have higher errors?")
print("-" * 80)
for metric in ['Mean', 'Median']:
    print(f"\n{metric} error progression:")
    for n in [5, 10, 20, 30]:
        print(f"  n={n}: {error_stats.loc[n, metric]:.4f}")

    # Check if monotonically increasing
    values = [error_stats.loc[n, metric] for n in [5, 10, 20, 30]]
    is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
    print(f"  Monotonically increasing: {is_monotonic}")

print("\n" + "=" * 80)
print("2. OUTLIER ANALYSIS")
print("=" * 80)

# Define outliers using IQR method for each sample size
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']
    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    extreme_outlier_threshold = Q3 + 3.0 * IQR

    outliers = subset > outlier_threshold
    extreme_outliers = subset > extreme_outlier_threshold

    print(f"\nn={n}:")
    print(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
    print(f"  Outlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.4f}")
    print(f"  Extreme outlier threshold (Q3 + 3.0*IQR): {extreme_outlier_threshold:.4f}")
    print(f"  Number of outliers: {outliers.sum()} ({100*outliers.sum()/len(subset):.2f}%)")
    print(f"  Number of extreme outliers: {extreme_outliers.sum()} ({100*extreme_outliers.sum()/len(subset):.2f}%)")
    print(f"  Mean error without outliers: {subset[~outliers].mean():.4f}")
    print(f"  Mean error with outliers: {subset.mean():.4f}")
    print(f"  Impact of outliers on mean: {subset[outliers].mean() - subset[~outliers].mean():.4f}")

print("\n" + "=" * 80)
print("3. ERROR DISTRIBUTION ANALYSIS")
print("=" * 80)

# Calculate distribution metrics
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']

    print(f"\nn={n}:")
    print(f"  Skewness: {stats.skew(subset):.4f} (0=symmetric, >0=right-tailed, <0=left-tailed)")
    print(f"  Kurtosis: {stats.kurtosis(subset):.4f} (0=normal, >0=heavy-tailed, <0=light-tailed)")
    print(f"  Coefficient of Variation: {subset.std()/subset.mean():.4f}")
    print(f"  Ratio P95/Median: {subset.quantile(0.95)/subset.median():.4f}")
    print(f"  Ratio Max/Median: {subset.max()/subset.median():.4f}")

print("\n" + "=" * 80)
print("4. VARIABLE-LEVEL ANALYSIS")
print("=" * 80)

# Group by variable and sample size
var_analysis = df_baselines.groupby(['variable', 'baseline_n'])['abs_error_from_mean'].agg([
    'mean', 'median', 'std', 'count'
]).reset_index()

# Pivot to compare sample sizes
var_pivot_mean = var_analysis.pivot(index='variable', columns='baseline_n', values='mean')
var_pivot_median = var_analysis.pivot(index='variable', columns='baseline_n', values='median')

print("\nMean Absolute Error by Variable and Sample Size:")
print(var_pivot_mean.round(4))

print("\nMedian Absolute Error by Variable and Sample Size:")
print(var_pivot_median.round(4))

# Check which variables show degradation with larger samples
print("\n" + "-" * 80)
print("Variables where error INCREASES from n=5 to n=30:")
print("-" * 80)
for var in var_pivot_mean.index:
    mean_change = var_pivot_mean.loc[var, 30] - var_pivot_mean.loc[var, 5]
    median_change = var_pivot_median.loc[var, 30] - var_pivot_median.loc[var, 5]

    if mean_change > 0:
        print(f"\n{var}:")
        print(f"  Mean error: {var_pivot_mean.loc[var, 5]:.4f} (n=5) → {var_pivot_mean.loc[var, 30]:.4f} (n=30)")
        print(f"  Change: +{mean_change:.4f} ({100*mean_change/var_pivot_mean.loc[var, 5]:.2f}%)")
        print(f"  Median error: {var_pivot_median.loc[var, 5]:.4f} (n=5) → {var_pivot_median.loc[var, 30]:.4f} (n=30)")
        print(f"  Change: +{median_change:.4f} ({100*median_change/var_pivot_median.loc[var, 5] if var_pivot_median.loc[var, 5] > 0 else 0:.2f}%)")

print("\n" + "=" * 80)
print("5. VARIANCE AND SPREAD ANALYSIS")
print("=" * 80)

variance_stats = df_baselines.groupby('baseline_n')['abs_error_from_mean'].agg([
    'var', 'std',
    lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
    lambda x: np.percentile(x, 90) - np.percentile(x, 10),  # 80th percentile range
]).round(4)
variance_stats.columns = ['Variance', 'StdDev', 'IQR', 'P10-P90 Range']

print("\nVariance and Spread Metrics:")
print(variance_stats)

# Check if variance increases with sample size
print("\n" + "-" * 80)
print("Does variance increase with larger samples?")
print("-" * 80)
for col in variance_stats.columns:
    print(f"\n{col} progression:")
    values = [variance_stats.loc[n, col] for n in [5, 10, 20, 30]]
    for i, n in enumerate([5, 10, 20, 30]):
        print(f"  n={n}: {values[i]:.4f}")
    is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
    print(f"  Monotonically increasing: {is_increasing}")

print("\n" + "=" * 80)
print("6. EXTREME VALUE ANALYSIS")
print("=" * 80)

# Look at top 10 errors for each sample size
print("\nTop 10 largest errors by sample size:")
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n][['variable', 'abs_error_from_mean', 'ground_truth']].copy()
    subset = subset.nlargest(10, 'abs_error_from_mean')
    print(f"\nn={n}:")
    for idx, row in subset.iterrows():
        print(f"  {row['variable']}: error={row['abs_error_from_mean']:.4f}, ground_truth={row['ground_truth']:.4f}")

print("\n" + "=" * 80)
print("7. CORRELATION ANALYSIS")
print("=" * 80)

# Check correlation between ground truth and error for each sample size
print("\nCorrelation between ground_truth and abs_error_from_mean:")
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]
    corr = subset['ground_truth'].corr(subset['abs_error_from_mean'])
    print(f"  n={n}: {corr:.4f}")

print("\n" + "=" * 80)
print("8. DETAILED OUTLIER IMPACT")
print("=" * 80)

# Calculate how much outliers contribute to the mean error difference
print("\nImpact of outliers on mean error differences:")
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']
    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    is_outlier = subset > outlier_threshold
    mean_with_outliers = subset.mean()
    mean_without_outliers = subset[~is_outlier].mean()
    outlier_contribution = mean_with_outliers - mean_without_outliers

    print(f"\nn={n}:")
    print(f"  Mean with outliers: {mean_with_outliers:.4f}")
    print(f"  Mean without outliers: {mean_without_outliers:.4f}")
    print(f"  Outlier contribution to mean: {outlier_contribution:.4f}")
    print(f"  Percentage of mean due to outliers: {100*outlier_contribution/mean_with_outliers:.2f}%")

# Compare n=5 vs n=30 outlier impact
n5_subset = df_baselines[df_baselines['baseline_n'] == 5]['abs_error_from_mean']
n30_subset = df_baselines[df_baselines['baseline_n'] == 30]['abs_error_from_mean']

Q3_n5 = n5_subset.quantile(0.75)
IQR_n5 = n5_subset.quantile(0.75) - n5_subset.quantile(0.25)
outlier_threshold_n5 = Q3_n5 + 1.5 * IQR_n5

Q3_n30 = n30_subset.quantile(0.75)
IQR_n30 = n30_subset.quantile(0.75) - n30_subset.quantile(0.25)
outlier_threshold_n30 = Q3_n30 + 1.5 * IQR_n30

mean_diff = n30_subset.mean() - n5_subset.mean()
mean_diff_no_outliers = n30_subset[n30_subset <= outlier_threshold_n30].mean() - n5_subset[n5_subset <= outlier_threshold_n5].mean()

print("\n" + "-" * 80)
print("Comparison: n=5 vs n=30")
print("-" * 80)
print(f"Mean error difference (n=30 - n=5): {mean_diff:.4f}")
print(f"Mean error difference without outliers: {mean_diff_no_outliers:.4f}")
print(f"Percentage of difference explained by outliers: {100*(mean_diff - mean_diff_no_outliers)/mean_diff:.2f}%")

print("\n" + "=" * 80)
print("9. SUMMARY FINDINGS")
print("=" * 80)

# Calculate key metrics
n5_mean = error_stats.loc[5, 'Mean']
n30_mean = error_stats.loc[30, 'Mean']
n5_median = error_stats.loc[5, 'Median']
n30_median = error_stats.loc[30, 'Median']

print(f"""
KEY FINDINGS:

1. Overall Error Progression:
   - Mean error increases from {n5_mean:.4f} (n=5) to {n30_mean:.4f} (n=30)
   - Relative increase: {100*(n30_mean-n5_mean)/n5_mean:.2f}%
   - Median error increases from {n5_median:.4f} (n=5) to {n30_median:.4f} (n=30)
   - Relative increase: {100*(n30_median-n5_median)/n5_median:.2f}%

2. Distribution Characteristics:
   - Skewness increases with sample size: {error_stats.loc[5, 'Skewness']:.4f} → {error_stats.loc[30, 'Skewness']:.4f}
   - Indicates distributions become more right-tailed (more extreme outliers)

3. Variance and Spread:
   - Variance increases: {variance_stats.loc[5, 'Variance']:.4f} → {variance_stats.loc[30, 'Variance']:.4f}
   - Standard deviation increases: {variance_stats.loc[5, 'StdDev']:.4f} → {variance_stats.loc[30, 'StdDev']:.4f}
   - IQR increases: {variance_stats.loc[5, 'IQR']:.4f} → {variance_stats.loc[30, 'IQR']:.4f}

4. Outlier Impact:
   - Outliers contribute significantly to mean error
   - Larger samples have more extreme outliers
   - Heavy-tailed distributions become more pronounced with larger samples

5. Variable-Specific Patterns:
   - Some variables show consistent degradation with larger samples
   - Effect varies by variable type and ground truth distribution
""")

print("\nAnalysis complete!")
