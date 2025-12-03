import os 
import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_ground_truth_quartile_distribution_heatmap(results_sets, output_dir="analysis_results"):
    """
    Create heatmaps showing the count of ground truths in each quartile for each model.

    Args:
        results_sets: List of DataFrames containing results for each domain
        output_dir: Directory to save the output plot
    """
    titles = {'nhanes': 'NHANES', 'glassdoor': 'Glassdoor', 'pitchbook': 'Pitchbook'}

    # Model name mapping for pretty printing
    model_name_mapping = {
        'gpt-4o_base_direct_temp0.5': 'GPT 4o',
        'meta-llama-3-70b_base_direct_temp0.5': 'Llama 3 70B',
        'meta-llama-3-8b_base_direct_temp0.5': 'Llama 3 8B',
        'o3-mini_base_direct_tempmedium': 'o3 Mini',
        'o4-mini_base_direct_tempmedium': 'o4 Mini',
        'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': 'Qwen3 235B',
    }

    # Create figure with subplots for each domain
    num_datasets = len(results_sets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 6))

    # Ensure axes is always iterable
    if num_datasets == 1:
        axes = [axes]

    for idx, results in enumerate(results_sets):
        # Filter out statistical baselines
        filtered_results = results[~results['approach'].str.contains('statistical_baseline', case=False, na=False)]

        # Build counts by approach × quartile (1..4)
        counts = (
            filtered_results.groupby(["approach", "quartile_of_gt"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[1, 2, 3, 4], fill_value=0)
        )

        # Map approach names to display names
        counts.index = [model_name_mapping.get(approach, approach) for approach in counts.index]

        # Sort by model name for consistency
        counts = counts.sort_index()

        # Create heatmap
        ax = axes[idx]
        import seaborn as sns

        sns.heatmap(
            counts,
            ax=ax,
            cmap='YlOrRd',
            annot=True,
            fmt='g',
            linewidths=0.5,
            cbar_kws={"label": "Count of Ground Truths"},
            square=False,
            annot_kws={'fontsize': 12}
        )

        # Get dataset name
        dataset_name = results['dataset'].iloc[0]

        # Labels and title
        ax.set_xlabel("Quartile", fontsize=14)
        if idx == 0:
            ax.set_ylabel("Model", fontsize=14)
        else:
            ax.set_ylabel("")
        ax.set_title(titles.get(dataset_name, dataset_name.capitalize()), fontsize=16, fontweight='bold')

        # Tick formatting
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        # Remove colorbar from individual subplots except the last one
        if idx < num_datasets - 1:
            ax.collections[0].colorbar.remove()

    plt.suptitle('Distribution of Ground Truths Across Quartiles by Model',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/ground_truth_quartile_distribution_heatmap.png", "wb") as f:
        plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ece_by_domain(result_sets, output_dir="analysis_results"):
    titles = {'nhanes': 'NHANES', 'glassdoor': 'Glassdoor', 'pitchbook': 'Pitchbook'}
    # Model name mapping for pretty printing
    model_name_mapping = {
        'gpt-4o_base_direct_temp0.5': 'GPT 4o',
        'meta-llama-3-70b_base_direct_temp0.5': 'Llama 3 70B',
        'meta-llama-3-8b_base_direct_temp0.5': 'Llama 3 8B',
        'o3-mini_base_direct_tempmedium': 'o3 Mini',
        'o4-mini_base_direct_tempmedium': 'o4 Mini',
        'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': 'Qwen3 235B',
    }

    result_sets = [results.copy() for results in result_sets]
    # Only keep non-posterior LLM results (filter out statistical baselines, posteriors) 
    for i in range(len(result_sets)):
        result_sets[i] = result_sets[i][~result_sets[i]['approach'].str.contains('stat', case=False, na=False)]
        result_sets[i] = result_sets[i][~result_sets[i]['approach'].str.contains('posterior', case=False, na=False)]
    

    def compute_ece_scores_per_trial(results):
        """Compute ECE scores per trial for each approach, returning mean and std."""
        # Filter out statistical baselines
        results = results[~results['approach'].str.contains('stat', case=False, na=False)]

        # Get dataset name
        dataset_name = results['dataset'].iloc[0]

        # Dictionary to store ECE per trial per approach
        ece_per_trial = {}

        # Get all unique trials
        trials = results['trial'].unique()

        for trial in trials:
            trial_data = results[results['trial'] == trial]

            # Compute counts for this trial
            counts = (
                trial_data.groupby(["approach", "quartile_of_gt"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=[1, 2, 3, 4], fill_value=0)
            )

            # Percentages
            row_totals = counts.sum(axis=1)
            safe_totals = row_totals.replace(0, np.nan)
            pct = counts.div(safe_totals, axis=0) * 100

            # Compute ECE for this trial
            distance_from_perfect = pct.sub(25).abs()
            trial_ece = (distance_from_perfect.sum(axis=1) / 4)

            # Store in dictionary
            for approach in trial_ece.index:
                if approach not in ece_per_trial:
                    ece_per_trial[approach] = []
                ece_per_trial[approach].append(trial_ece[approach])

        # Compute mean and std for each approach
        ece_mean = {}
        ece_std = {}

        for approach, ece_values in ece_per_trial.items():
            display_name = model_name_mapping.get(approach, approach)
            ece_mean[display_name] = np.mean(ece_values)
            ece_std[display_name] = np.std(ece_values, ddof=1) if len(ece_values) > 1 else 0

        ece_mean = pd.Series(ece_mean, name="ECE")
        ece_std = pd.Series(ece_std, name="ECE_std")

        return ece_mean, ece_std, dataset_name

    ece_results = [compute_ece_scores_per_trial(results.copy()) for results in result_sets]

    # Color scheme
    colors = {
        # LLM models - vibrant colors
        'GPT 4o': '#4287f5',
        'Llama 3 70B': '#64b57b',
        'Llama 3 8B': '#fa829c',
        'o3 Mini': '#ffad42',
        'o4 Mini': '#c397fc',
        'Qwen3 235B': '#fa82ee',
        'Perfect Calibration': '#808080',
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, (ece_mean, ece_std, dataset_key) in enumerate(ece_results):
        dataset = titles[dataset_key]

        # Sort by ECE value
        sort_order = ece_mean.sort_values().index
        ece_mean_sorted = ece_mean.loc[sort_order]
        ece_std_sorted = ece_std.loc[sort_order]

        # Get display names and colors
        display_names = ece_mean_sorted.index.tolist()
        bar_colors = [colors.get(name, '#808080') for name in display_names]

        # Create bar plot with error bars
        ax = axes[idx]
        x_pos = np.arange(len(display_names))

        bars = ax.bar(x_pos, ece_mean_sorted.values,
                      yerr=ece_std_sorted.values,
                      capsize=5,
                      alpha=0.8,
                      color=bar_colors,
                      edgecolor='black',
                      linewidth=1.5,
                      error_kw={'elinewidth': 2, 'capthick': 2})

        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=14)
        ax.set_ylabel('Expected Calibration Error (%)' if idx == 0 else '', fontsize=14)
        ax.set_title(dataset, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars, ece_mean_sorted.values, ece_std_sorted.values)):
            height = bar.get_height()
            # Position label above error bar
            label_height = height + std
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=14)
    
    plt.suptitle('Expected Calibration Error by Domain and Approach', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    with open(f"{output_dir}/ece_by_domain.png", "wb") as f: 
        plt.savefig(f, dpi=300, bbox_inches='tight')


def z_score_cdf_plot(results_sets, output_dir="analysis_results"):
    results_sets = [results.copy() for results in results_sets]
    # Only keep non-posterior LLM results (filter out statistical baselines, posteriors) 
    for i in range(len(results_sets)):
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('stat', case=False, na=False)]
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('posterior', case=False, na=False)]
    
    def plot_z_score_subplot(df, ax, max_sigma: float = 10.0, n_points: int = 300, show_legend=False, title="", show_xlabel=False):
        """
        Modified version of plot_uncertainty_accuracy that works with subplots.
        """
        # Define color scheme and model name mapping
        color_map = {
            'gpt-4o': '#4287f5',
            'meta-llama-3-70b': '#64b57b',
            'o3-mini': '#ffad42',
            'o4-mini': '#c397fc',
            'qwen3-235b': '#fa82ee',
            'meta-llama-3-8b': '#fa829c',
        }
        
        # Model name mapping for pretty printing
        model_name_mapping = {
            'gpt-4o': 'GPT-4o',
            'meta-llama-3-70b': 'Llama-3-70B',
            'meta-llama-3-8b': 'Llama-3-8B',
            'o3-mini': 'o3-mini',
            'o4-mini': 'o4-mini',
            'qwen3-235b-a22b-fp8-tput': 'Qwen3-235B',
            'qwen3-235b': 'Qwen3-235B',
        }
        
        # -------- identify model family ---------------------------------
        df = df.copy()
        df["model_family"] = df["approach"].apply(
            lambda x: (x.split("_")[0] if "statistical_baseline" not in x
                    else x.split("_")[2])
        )
        # Filter out statistical baselines
        df = df[~df["approach"].str.contains("statistical_baseline")]


        # -------- pre-compute z-scores ----------------------------------
        df["z"] = df["abs_error_from_mean"] / df["std"]

        o4_mini = df[df['model_family'] == 'o4-mini']['z']
        o4_mini_stats = o4_mini.describe()
        with open (f"{output_dir}/o4_mini_z_score_stats.txt", "w") as f: 
            f.write(f"O4 Mini Z-Score Statistics:\n{o4_mini_stats}\n")
            o4_mini.to_csv(f"{output_dir}/o4_mini_z_score_stats.csv")

        sigmas   = np.linspace(0.0, max_sigma, n_points)
        families = sorted(df["model_family"].unique())

        # coverage[fam] = array of % rows with z ≤ σ for every σ in grid
        coverage = {}
        lines = []
        for fam in families:
            z_vals = df.loc[df["model_family"] == fam, "z"].values
            coverage[fam] = np.array([(z_vals <= s).mean() * 100 for s in sigmas])
            # Use specific color if available, otherwise use default matplotlib color
            color = color_map.get(fam, None)
            # Use pretty name for legend if available
            display_name = model_name_mapping.get(fam, fam)
            line = ax.plot(sigmas, coverage[fam], label=display_name, color=color)
            lines.extend(line)

        # Add perfect calibration line (CDF of standard normal distribution)
        from scipy.stats import norm
        perfect_calibration = norm.cdf(sigmas) * 100
        ax.plot(sigmas, perfect_calibration, 'k--', alpha=0.7, linewidth=2, label='Perfect calibration')

        if show_xlabel:
            ax.set_xlabel("Number of standard deviations", fontsize=16)
        ax.set_ylabel("% ground truths inside interval")
        ax.set_xlim(0, max_sigma)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontweight='bold', fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if show_legend:
            ax.legend(title="Model family", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        return lines

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    titles = {'glassdoor': 'Glassdoor', 'pitchbook': 'Pitchbook', 'nhanes': 'NHANES'}
    lines1 = None
    for i, results in enumerate(results_sets):
        # Only show x-label on the middle subplot (index 1)
        plot = plot_z_score_subplot(results, axes[i], title=titles[results['dataset'].iloc[0]], show_xlabel=(i == 1))
        if lines1 is None:
            lines1 = plot 

    # Create a shared legend using the lines from the first subplot plus the perfect calibration line
    handles = lines1
    labels = [line.get_label() for line in lines1]

    # Add the perfect calibration line to the legend
    from matplotlib.lines import Line2D
    perfect_cal_line = Line2D([0], [0], color='black', linestyle='--', alpha=0.7, linewidth=2)
    handles.append(perfect_cal_line)
    labels.append('Perfect calibration\n (Gaussian)')

    fig.legend(handles, labels, title="Model family", bbox_to_anchor=(0.89, 0.95), loc="upper left")

    # Adjust layout with less space on the right for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    with open(f"{output_dir}/z_score_cdf_plot.png", "wb") as f: 
        plt.savefig(f, dpi=300)
    plt.close(fig)


def plot_uncertainty_accuracy_scatterplots(all_unc_acc_results, output_dir="analysis_results"):
    """
    Create scatterplots for each domain showing uncertainty vs accuracy,
    colored by Spearman correlation strength.
    """
    # Model name mapping for pretty printing
    model_name_mapping = {
        'gpt-4o_base_direct_temp0.5': 'GPT 4o',
        'meta-llama-3-70b_base_direct_temp0.5': 'Llama 3 70B',
        'meta-llama-3-8b_base_direct_temp0.5': 'Llama 3 8B',
        'o3-mini_base_direct_tempmedium': 'o3 Mini',
        'o4-mini_base_direct_tempmedium': 'o4 Mini',
        'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': 'Qwen3 235B',
    }

    # Dataset name mapping for pretty printing
    dataset_name_mapping = {
        'nhanes': 'NHANES',
        'pitchbook': 'Pitchbook',
        'glassdoor': 'Glassdoor'
    }

    # Collect all Spearman correlations to establish global color scale
    all_correlations = []
    for dataset_name, dataset_metrics in all_unc_acc_results.items():
        for approach, metrics in dataset_metrics.items():
            all_correlations.append(metrics['spearman_corr'])

    # Create normalizer for consistent coloring across all plots
    norm = Normalize(vmin=min(all_correlations), vmax=max(all_correlations))
    cmap = cm.RdBu_r  # Red-Blue reversed (blue=negative, red=positive)

    # Determine number of datasets and create figure with that many subplots
    num_datasets = len(all_unc_acc_results)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5))

    # Ensure axes is always iterable (handle case of single subplot)
    if num_datasets == 1:
        axes = [axes]

    # Build datasets list dynamically from all_unc_acc_results
    datasets = [
        (dataset_metrics, dataset_name_mapping.get(dataset_name, dataset_name.capitalize()), axes[i])
        for i, (dataset_name, dataset_metrics) in enumerate(all_unc_acc_results.items())
    ]
    
    for idx, (unc_acc_dict, title, ax) in enumerate(datasets):
        # Extract data for plotting
        approaches = []
        mean_stds = []
        mean_errors = []
        spearman_corrs = []

        for approach, metrics in unc_acc_dict.items():
            approaches.append(model_name_mapping.get(approach, approach))
            mean_stds.append(metrics['mean_std_ratio'])
            mean_errors.append(metrics['mean_error_ratio'])
            spearman_corrs.append(metrics['spearman_corr'])

        # Create scatter plot with colors based on Spearman correlation
        scatter = ax.scatter(
            mean_stds,
            mean_errors,
            c=spearman_corrs,
            cmap=cmap,
            norm=norm,
            s=150,
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5
        )

        # Add dashed reference lines at x=1.0 and y=1.0
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        # Add labels for each point
        for i, approach in enumerate(approaches):
            # Position text to avoid overlap
            x_offset = 0.3
            y_offset = 1.5

            # Adjust offset based on position to avoid clutter
            if i % 2 == 0:
                y_offset = -y_offset

            ax.annotate(
                approach,
                (mean_stds[i], mean_errors[i]),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                fontsize=9,
                ha='left',
                va='bottom' if y_offset > 0 else 'top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, linewidth=0.8)
            )

        # Set labels and title - use dynamic limits based on data with some padding
        x_min, x_max = min(mean_stds), max(mean_stds)
        y_min, y_max = min(mean_errors), max(mean_errors)
        x_padding = (x_max - x_min) * 0.15
        y_padding = (y_max - y_min) * 0.15
        ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        # Only add x-label to the middle subplot
        if idx == num_datasets // 2:
            ax.set_xlabel('Mean Standard Deviation Ratio (relative to N=5)', fontsize=12)
        ax.set_ylabel('Mean Error Ratio (relative to N=5)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Spearman Correlation', rotation=270, labelpad=20, fontsize=12)

    # Add overall title
    fig.suptitle('Uncertainty vs Accuracy by Model (colored by Spearman correlation)', 
                fontsize=16, fontweight='bold', y=1.02, x=.4)

    plt.tight_layout()
    plt.subplots_adjust(right=.75)  
    with open(f"{output_dir}/uncertainty_accuracy_scatterplots.png", "wb") as f:
        plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_error_ratio_by_domain(results_sets, output_dir='analysis_results'):
    """
    Create bar plots showing average error ratio for each approach in each domain.
    Each domain gets its own subplot.
    """
    results_sets = [results.copy() for results in results_sets]

    # Only keep non-posterior LLM results (filter out statistical baselines, posteriors) 
    for i in range(len(results_sets)):
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('stat', case=False, na=False)]
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('posterior', case=False, na=False)]
    
    # Color scheme
    colors = {
        # LLM models - vibrant colors
        'GPT-4o': '#4287f5',
        'gpt-4o': '#4287f5',
        'gpt-4o_base_direct_temp0.5': '#4287f5',
        'meta-llama-3-70b': '#64b57b',
        'llama-3-70b': '#64b57b',
        'meta-llama-3-70b_base_direct_temp0.5': '#64b57b',
        'o3-mini': '#ffad42',
        'o3-mini_base_direct_tempmedium': '#ffad42',
        'o4-mini': '#c397fc',
        'o4-mini_base_direct_tempmedium': '#c397fc',
        'qwen3-235b-a22b-fp8-tput': '#fa82ee',
        'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': '#fa82ee',
        'meta-llama-3-8b': '#fa829c',
        'meta-llama-3-8b_base_direct_temp0.5': '#fa829c',
    }
    
    # Model name mapping for pretty printing
    model_name_mapping = {
        'gpt-4o_base_direct_temp0.5': 'GPT 4o',
        'meta-llama-3-70b_base_direct_temp0.5': 'Llama 3 70B',
        'meta-llama-3-8b_base_direct_temp0.5': 'Llama 3 8B',
        'o3-mini_base_direct_tempmedium': 'o3 Mini',
        'o4-mini_base_direct_tempmedium': 'o4 Mini',
        'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': 'Qwen3 235B',
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = {'nhanes': 'NHANES', 'pitchbook': 'Pitchbook', 'glassdoor': 'Glassdoor'}

    # First pass: calculate error stats for all datasets and find global y-axis limits
    all_error_stats = []
    all_ci_lowers = []
    all_ci_uppers = []

    for idx, results in enumerate(results_sets):
        dataset = results['dataset'].iloc[0]

        # Filter out statistical baselines and llama-3-8b
        llm_results = results[~results['approach'].str.contains('stat', case=False, na=False)]
        llm_results = llm_results[~llm_results['approach'].str.contains('llama-3-8b', case=False, na=False)]

        # Calculate error statistics
        error_stats = llm_results.groupby('approach').agg({
            'error_ratio_mean': ['mean', 'std', 'count']
        }).round(4)

        error_stats.columns = ['mean_error_ratio_mean', 'std_error_ratio_mean', 'n']

        # Calculate 95% confidence interval using t-distribution
        error_stats['ci_lower'] = error_stats.apply(
            lambda row: row['mean_error_ratio_mean'] - stats.t.ppf(0.975, row['n']-1) * row['std_error_ratio_mean'] / np.sqrt(row['n'])
            if row['n'] > 1 else row['mean_error_ratio_mean'],
            axis=1
        )
        error_stats['ci_upper'] = error_stats.apply(
            lambda row: row['mean_error_ratio_mean'] + stats.t.ppf(0.975, row['n']-1) * row['std_error_ratio_mean'] / np.sqrt(row['n'])
            if row['n'] > 1 else row['mean_error_ratio_mean'],
            axis=1
        )

        error_stats = error_stats.sort_values('mean_error_ratio_mean')
        all_error_stats.append(error_stats)
        all_ci_lowers.extend(error_stats['ci_lower'].tolist())
        all_ci_uppers.extend(error_stats['ci_upper'].tolist())

    # Calculate global y-axis limits with padding
    global_ymin = min(all_ci_lowers)
    global_ymax = max(all_ci_uppers)
    y_range = global_ymax - global_ymin
    y_padding = y_range * 0.1  # 10% padding
    global_ylim = (global_ymin - y_padding, global_ymax + y_padding)

    # Second pass: create plots with consistent y-axis
    for idx, results in enumerate(results_sets):
        dataset = results['dataset'].iloc[0]
        error_stats = all_error_stats[idx]
        
        # Map approach names to display names and colors
        display_names = [model_name_mapping.get(approach, approach) for approach in error_stats.index]
        bar_colors = [colors.get(approach, colors.get(approach.split('_')[0], '#808080')) 
                     for approach in error_stats.index]
        
        # Create bar plot
        ax = axes[idx]
        x_pos = np.arange(len(display_names))
        
        # Calculate error bar values (asymmetric)
        yerr_lower = error_stats['mean_error_ratio_mean'] - error_stats['ci_lower']
        yerr_upper = error_stats['ci_upper'] - error_stats['mean_error_ratio_mean']
        yerr = np.array([yerr_lower, yerr_upper])
        
        bars = ax.bar(x_pos, error_stats['mean_error_ratio_mean'],
                      yerr=yerr,
                      capsize=5,
                      alpha=0.8,
                      color=bar_colors,
                      edgecolor='black',
                      error_kw={'elinewidth': 2, 'capthick': 2})

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, error_stats['mean_error_ratio_mean'])):
            height = bar.get_height()
            # Position label above error bar
            label_height = height + yerr_upper.iloc[i]
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=14)

        # Add horizontal line at y=1.0 (baseline)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (n=5)')
        
        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=16)
        ax.set_ylabel('Error Ratio (vs. Baseline n=5)' if idx == 0 else '')
        ax.set_ylim(global_ylim)
        ax.set_title(titles.get(dataset, dataset.capitalize()), fontweight='bold', fontsize=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend()
    
    plt.suptitle('Error Ratios by Domain (95% CI)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    with open(f"{output_dir}/error_ratios_by_domain.png", "wb") as f:
        plt.savefig(f, dpi=300, bbox_inches='tight')


def calibration_heat_map(results_sets, output_dir='analysis_results'): 
    results_sets = [results.copy() for results in results_sets]
    # Only keep non-posterior LLM results (filter out statistical baselines, posteriors) 
    for i in range(len(results_sets)):
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('stat', case=False, na=False)]
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('posterior', case=False, na=False)]
    

    titles = {'nhanes': 'NHANES', 'pitchbook': 'Pitchbook', 'glassdoor': 'Glassdoor'}
    def plot_quartile_heatmap(
        results,
        *,
        sort_by_ece: bool = True,
        annotate: bool = True,
        ax=None,
        # Binomial test controls:
        binomial_alternative: str = "two-sided",   # "two-sided", "greater", or "less"
        star_levels=(0.05, 0.01, 0.001),          # thresholds for *, **, ***
        annot_fontsize: int = 12,
        xtick_fontsize: int = 12,
        ytick_fontsize: int = 12,
        xlabel_fontsize: int = 12,
    ):
        """
        Draw a heatmap that shows, for every approach, the % of ground-truth
        values that fall into each p quartile, and annotate
        each cell with per-quartile binomial test significance stars.

        Per-cell test:
            H0: p = 0.25
            H1: p != 0.25  (two-sided by default; can be "greater" or "less")

        Stars:
            * p < 0.05, ** p < 0.01, *** p < 0.001
        """
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import re
        from scipy.stats import binomtest
        from matplotlib.patches import Rectangle
        import matplotlib.patheffects as path_effects

        _cmap = plt.cm.RdBu_r

        # Filter out statistical baselines
        filtered_results = results[~results['approach'].str.contains('statistical_baseline', case=False, na=False)]
        # Build counts by approach × quartile (1..4)
        counts = (
            filtered_results.groupby(["approach", "quartile_of_gt"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[1, 2, 3, 4], fill_value=0)
        )

        # Percentages
        row_totals = counts.sum(axis=1)
        safe_totals = row_totals.replace(0, np.nan)
        pct = counts.div(safe_totals, axis=0) * 100

        # Pretty names
        model_name_mapping = {
            'gpt-4o_base_direct_temp0.5': 'GPT 4o',
            'gpt-4o_base_direct_tempmedium': 'GPT 4o',
            'meta-llama-3-70b_base_direct_temp0.5': 'Llama 3 70B',
            'meta-llama-3-8b_base_direct_temp0.5': 'Llama 3 8B',
            'o3-mini_base_direct_tempmedium': 'o3 Mini',
            'o4-mini_base_direct_tempmedium': 'o4 Mini',
            'qwen3-235b-a22b-fp8-tput_base_direct_temp0.6': 'Qwen3 235B',
            'qwen3-235b-a22b-fp8-tput_base_direct_tempmedium': 'Qwen3 235B',
            'statistical_baseline_n5': 'Baseline (n=5)',
            'statistical_baseline_n10': 'Baseline (n=10)',
            'statistical_baseline_n20': 'Baseline (n=20)',
            'statistical_baseline_n30': 'Baseline (n=30)',
            'statistical_baseline_nall': 'Baseline (n=all)',
            'statistical_baseline_nALL': 'Baseline (n=all)',
        }

        display_names = {}
        for approach in pct.index:
            display_name = model_name_mapping[approach]
            display_names[approach] = display_name

        pct.index = [display_names[a] for a in pct.index]
        counts.index = pct.index
        row_totals.index = pct.index

        # Add Perfect Calibration row
        perfect_calibration = pd.DataFrame([[25, 25, 25, 25]],
                                        columns=[1, 2, 3, 4],
                                        index=['Perfect Calibration'])
        pct = pd.concat([pct, perfect_calibration])

        # Sort by ECE
        if sort_by_ece:
            ece = (pct.sub(25).abs().sum(axis=1) / 4).rename("ECE")
            order = ece.drop(index='Perfect Calibration').sort_values().index.tolist() + ['Perfect Calibration']
            pct = pct.loc[order]
        else:
            pct_without_perfect = pct.drop('Perfect Calibration')
            pct_without_perfect = pct_without_perfect.sort_index()
            pct = pd.concat([pct_without_perfect, pct.loc[['Perfect Calibration']]])

        # Stars dataframe
        stars_df = pd.DataFrame('', index=pct.index, columns=pct.columns)

        def p_to_stars(p):
            if p < star_levels[2]:
                return '***'
            elif p < star_levels[1]:
                return '**'
            elif p < star_levels[0]:
                return '*'
            else:
                return ''

        for idx in stars_df.index:
            if idx == 'Perfect Calibration':
                continue
            n = int(row_totals.loc[idx]) if pd.notna(row_totals.loc[idx]) else 0
            if n <= 0:
                continue
            for q in [1, 2, 3, 4]:
                k = int(counts.loc[idx, q]) if (idx in counts.index and q in counts.columns) else 0
                res = binomtest(k, n, p=0.25, alternative=binomial_alternative)
                stars_df.loc[idx, q] = p_to_stars(res.pvalue)

        if annotate:
            # Start with the plain numbers as strings
            display_annot = pct.copy()
            display_annot = display_annot.applymap(lambda x: "" if pd.isna(x) else f"{x:.1f}")

            # Instead of appending stars, bold the number when that cell was significant
            for q in [1, 2, 3, 4]:
                for idx in display_annot.index:
                    if stars_df.loc[idx, q] != '':  # i.e., p passed any of your thresholds
                        display_annot.loc[idx, q] = r"$\mathbf{" + display_annot.loc[idx, q] + "}$"

            annot_data = display_annot
            fmt = ""
        else:
            annot_data = False
            fmt = ".1f"

        # ---- Plot ----
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 0.6 * len(pct) + 2))  # bigger cells
        else:
            fig = ax.figure

        hm = sns.heatmap(
            pct,
            ax=ax,
            cmap=_cmap,
            center=25,
            vmin=None, vmax=None,
            annot=annot_data,
            fmt=fmt,
            linewidths=0.5,
            cbar_kws={"label": "% Ground Truth in Quartile"},
            square=False,   # allow rectangular cells
            annot_kws={'fontsize': 12, 'fontweight': 'normal'}  # remove bold
        )

        # Border around Perfect Calibration row
        if 'Perfect Calibration' in pct.index:
            n_cols = len(pct.columns)
            n_rows = len(pct.index)
            rect = Rectangle((0, n_rows - 1), n_cols, 1,
                            linewidth=1, edgecolor='black', facecolor='none',
                            clip_on=False, transform=ax.transData)
            ax.add_patch(rect)

        # Labels and title
        ax.set_xlabel("Quartile", fontsize=xlabel_fontsize)
        #ax.set_ylabel("Approach", fontsize=ylabel_fontsize)
        ax.set_title("Calibration heat-map (red > 25 %, blue < 25 %)", fontsize=14, fontweight='bold', pad=10)

        # Tick formatting
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=xtick_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)

        return ax

    import matplotlib.pyplot as plt

    # Create a figure with 3 subplots arranged horizontally with smaller overall size
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Reduced height from 8 to 4

    i = 0 
    for results in results_sets:
        # Create the three heatmaps as subplots with consistent row ordering
        plot_quartile_heatmap(results, ax=axes[i], sort_by_ece=False)
        axes[i].set_title(titles[results['dataset'].iloc[0]], fontsize=20, fontweight='bold')
        axes[i].collections[0].colorbar.remove()  # Remove colorbar from left plot
        i += 1

    # Remove y-tick labels from middle and right subplots
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    # Create a single shared colorbar positioned further to the right
    mappable = axes[0].collections[0]
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.8, aspect=30, pad=0.05)
    cbar.set_label("% Ground Truths in Quartile")

    # Adjust layout with more space for the colorbar on the right
    plt.subplots_adjust(wspace=0.1, right=0.77)  # Changed from 0.825 to 0.75 to move colorbar further right
    with open(f"{output_dir}/calibration_heatmap.png", "wb") as f:
        plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close(fig)


