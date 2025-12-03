import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the data
df = pd.read_csv('pitchbook_combined_processed_results.csv')

# Filter for only statistical baselines
baseline_approaches = ['statistical_baseline_n5', 'statistical_baseline_n10',
                       'statistical_baseline_n20', 'statistical_baseline_n30']
df_baselines = df[df['approach'].isin(baseline_approaches)].copy()

# Extract sample size from approach name
df_baselines['baseline_n'] = df_baselines['approach'].str.extract(r'n(\d+)').astype(int)

print("=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Distribution of errors by sample size (log scale)
ax1 = plt.subplot(2, 3, 1)
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']
    # Remove extreme outliers for visualization
    subset_clipped = subset[subset < subset.quantile(0.99)]
    ax1.hist(subset_clipped, bins=50, alpha=0.5, label=f'n={n}')
ax1.set_xlabel('Absolute Error from Mean')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Errors by Sample Size (99th percentile clipped)')
ax1.legend()
ax1.set_yscale('log')

# 2. Box plot comparison
ax2 = plt.subplot(2, 3, 2)
data_for_box = [df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean'] for n in [5, 10, 20, 30]]
bp = ax2.boxplot(data_for_box, labels=['n=5', 'n=10', 'n=20', 'n=30'],
                 showfliers=False)  # Hide outliers to see main distribution
ax2.set_ylabel('Absolute Error from Mean')
ax2.set_title('Error Distribution by Sample Size (outliers hidden)')
ax2.grid(True, alpha=0.3)

# 3. Mean and median error progression
ax3 = plt.subplot(2, 3, 3)
sample_sizes = [5, 10, 20, 30]
means = [df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean'].mean() for n in sample_sizes]
medians = [df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean'].median() for n in sample_sizes]
ax3.plot(sample_sizes, means, 'o-', label='Mean', linewidth=2, markersize=8)
ax3.plot(sample_sizes, medians, 's-', label='Median', linewidth=2, markersize=8)
ax3.set_xlabel('Sample Size')
ax3.set_ylabel('Error')
ax3.set_title('Mean and Median Error vs Sample Size')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Outlier count by sample size
ax4 = plt.subplot(2, 3, 4)
outlier_counts = []
outlier_pcts = []
for n in [5, 10, 20, 30]:
    subset = df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']
    Q1 = subset.quantile(0.25)
    Q3 = subset.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    outliers = (subset > outlier_threshold).sum()
    outlier_counts.append(outliers)
    outlier_pcts.append(100 * outliers / len(subset))

ax4.bar(sample_sizes, outlier_pcts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax4.set_xlabel('Sample Size')
ax4.set_ylabel('Percentage of Outliers (%)')
ax4.set_title('Outlier Frequency by Sample Size')
ax4.set_xticks(sample_sizes)
for i, (n, pct) in enumerate(zip(sample_sizes, outlier_pcts)):
    ax4.text(n, pct + 0.2, f'{pct:.1f}%', ha='center', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Percentile analysis
ax5 = plt.subplot(2, 3, 5)
percentiles = [50, 75, 90, 95, 99]
for p in percentiles:
    values = [df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean'].quantile(p/100) for n in sample_sizes]
    ax5.plot(sample_sizes, values, 'o-', label=f'P{p}', linewidth=2, markersize=6)
ax5.set_xlabel('Sample Size')
ax5.set_ylabel('Error at Percentile')
ax5.set_title('Error Percentiles by Sample Size')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# 6. Skewness and Kurtosis
ax6 = plt.subplot(2, 3, 6)
from scipy import stats
skewness = [stats.skew(df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']) for n in sample_sizes]
kurtosis = [stats.kurtosis(df_baselines[df_baselines['baseline_n'] == n]['abs_error_from_mean']) for n in sample_sizes]

ax6_twin = ax6.twinx()
line1 = ax6.plot(sample_sizes, skewness, 'o-', color='blue', label='Skewness', linewidth=2, markersize=8)
line2 = ax6_twin.plot(sample_sizes, kurtosis, 's-', color='red', label='Kurtosis', linewidth=2, markersize=8)

ax6.set_xlabel('Sample Size')
ax6.set_ylabel('Skewness', color='blue')
ax6_twin.set_ylabel('Kurtosis', color='red')
ax6.set_title('Distribution Shape Metrics by Sample Size')
ax6.tick_params(axis='y', labelcolor='blue')
ax6_twin.tick_params(axis='y', labelcolor='red')
ax6.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.savefig('baseline_error_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: baseline_error_analysis.png")

# Second figure: Variable-level analysis
fig2 = plt.figure(figsize=(20, 10))

# Get degrading variables
df_baselines['obs_id'] = df_baselines['variable'] + '_' + df_baselines['trial'].astype(str)
pivot_data = df_baselines.pivot_table(
    index=['obs_id', 'variable', 'trial', 'ground_truth'],
    columns='baseline_n',
    values='abs_error_from_mean'
).reset_index()

var_degradation = pivot_data.groupby('variable').agg({
    5: 'mean',
    30: 'mean'
}).round(4)
var_degradation.columns = ['Avg_Error_n5', 'Avg_Error_n30']
var_degradation['Change'] = var_degradation['Avg_Error_n30'] - var_degradation['Avg_Error_n5']
var_degradation['Pct_Change'] = 100 * var_degradation['Change'] / (var_degradation['Avg_Error_n5'] + 1e-10)
var_degradation = var_degradation.sort_values('Change', ascending=False)

# 1. Top degrading variables
ax1 = plt.subplot(1, 2, 1)
top_degrading = var_degradation.head(15)
y_pos = np.arange(len(top_degrading))
colors = ['red' if x > 0 else 'green' for x in top_degrading['Change']]
ax1.barh(y_pos, top_degrading['Change'], color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_degrading.index, fontsize=8)
ax1.set_xlabel('Change in Error (n=30 - n=5)')
ax1.set_title('Top 15 Variables by Error Change\n(Red = Degradation, Green = Improvement)')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(True, alpha=0.3, axis='x')

# 2. Scatter plot: n=5 error vs n=30 error
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(var_degradation['Avg_Error_n5'], var_degradation['Avg_Error_n30'],
           alpha=0.6, s=100, c=['red' if x > 0 else 'blue' for x in var_degradation['Change']])

# Add diagonal line (y=x)
max_val = max(var_degradation['Avg_Error_n5'].max(), var_degradation['Avg_Error_n30'].max())
ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='No change line')

ax2.set_xlabel('Average Error at n=5')
ax2.set_ylabel('Average Error at n=30')
ax2.set_title('Variable-Level Error Comparison\n(Red = Degradation, Blue = Improvement)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Annotate worst degrading variables
for idx, row in top_degrading.head(5).iterrows():
    if row['Change'] > 0:
        ax2.annotate(idx,
                    xy=(var_degradation.loc[idx, 'Avg_Error_n5'],
                        var_degradation.loc[idx, 'Avg_Error_n30']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('variable_level_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: variable_level_analysis.png")

# Third figure: Outlier behavior analysis
fig3 = plt.figure(figsize=(20, 10))

# Calculate outlier status for each observation
pivot_data['degrades'] = pivot_data[30] > pivot_data[5]

for n in [5, 30]:
    Q1 = pivot_data[n].quantile(0.25)
    Q3 = pivot_data[n].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    pivot_data[f'is_outlier_n{n}'] = pivot_data[n] > outlier_threshold

# 1. Error change distribution
ax1 = plt.subplot(2, 2, 1)
error_change = pivot_data[30] - pivot_data[5]
ax1.hist(error_change, bins=100, alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
ax1.set_xlabel('Error Change (n=30 - n=5)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Error Changes')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Outlier status transitions
ax2 = plt.subplot(2, 2, 2)
transition_counts = {
    'Not outlier → Not outlier': ((~pivot_data['is_outlier_n5']) & (~pivot_data['is_outlier_n30'])).sum(),
    'Outlier → Outlier': (pivot_data['is_outlier_n5'] & pivot_data['is_outlier_n30']).sum(),
    'Not outlier → Outlier': ((~pivot_data['is_outlier_n5']) & pivot_data['is_outlier_n30']).sum(),
    'Outlier → Not outlier': (pivot_data['is_outlier_n5'] & (~pivot_data['is_outlier_n30'])).sum()
}
colors_trans = ['green', 'orange', 'red', 'blue']
ax2.bar(range(len(transition_counts)), list(transition_counts.values()),
       color=colors_trans, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(transition_counts)))
ax2.set_xticklabels(list(transition_counts.keys()), rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Count')
ax2.set_title('Outlier Status Transitions (n=5 → n=30)')
for i, (key, val) in enumerate(transition_counts.items()):
    ax2.text(i, val + 20, str(val), ha='center', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Error at n=5 vs error change
ax3 = plt.subplot(2, 2, 3)
degrading_mask = pivot_data['degrades']
ax3.scatter(pivot_data[~degrading_mask][5],
           pivot_data[~degrading_mask][30] - pivot_data[~degrading_mask][5],
           alpha=0.3, s=20, color='blue', label='Improving')
ax3.scatter(pivot_data[degrading_mask][5],
           pivot_data[degrading_mask][30] - pivot_data[degrading_mask][5],
           alpha=0.3, s=20, color='red', label='Degrading')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Error at n=5')
ax3.set_ylabel('Error Change (n=30 - n=5)')
ax3.set_title('Error Change vs Initial Error')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Degradation by outlier status
ax4 = plt.subplot(2, 2, 4)
outlier_degradation = {
    'Outlier at n=5': (pivot_data['is_outlier_n5'] & pivot_data['degrades']).sum() / pivot_data['is_outlier_n5'].sum() * 100,
    'Not outlier at n=5': ((~pivot_data['is_outlier_n5']) & pivot_data['degrades']).sum() / (~pivot_data['is_outlier_n5']).sum() * 100
}
bars = ax4.bar(range(len(outlier_degradation)), list(outlier_degradation.values()),
              color=['orange', 'lightblue'], alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(outlier_degradation)))
ax4.set_xticklabels(list(outlier_degradation.keys()), fontsize=10)
ax4.set_ylabel('% Degrading')
ax4.set_title('Degradation Rate by Outlier Status at n=5')
ax4.set_ylim([0, 100])
for i, (key, val) in enumerate(outlier_degradation.items()):
    ax4.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outlier_behavior_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: outlier_behavior_analysis.png")

print("\n" + "=" * 80)
print("VISUALIZATION SUMMARY")
print("=" * 80)
print("""
Three visualization files created:

1. baseline_error_analysis.png
   - Shows overall error distributions across sample sizes
   - Demonstrates that outliers are more frequent with larger samples
   - Shows heavy-tailed distributions (high kurtosis)

2. variable_level_analysis.png
   - Identifies specific variables that degrade with larger samples
   - Shows scatter plot comparing n=5 vs n=30 performance

3. outlier_behavior_analysis.png
   - Tracks how observations transition between outlier/non-outlier status
   - Shows that observations starting as non-outliers at n=5 can become outliers at n=30
   - Demonstrates the relationship between initial error and degradation
""")
