import os
import warnings
import numpy as np
import pandas as pd
from utils import load_experiment_results, print_completion_stats
from plotting import (plot_uncertainty_accuracy_scatterplots,
                plot_error_ratio_by_domain, calibration_heat_map, z_score_cdf_plot, plot_ece_by_domain, 
                plot_ground_truth_quartile_distribution_heatmap)


def build_error_comparison_table(posterior_helped_percentages, prior_helped_percentages, dataset, output_dir="analysis_results"):
    """
    Build a table with rows = sample size, columns = posterior error ratio, prior error ratio,
    % of the time prior is better, % of the time posterior is better.
    """
    # Get all sample sizes (excluding 'prior' if present)
    sample_sizes = [s for s in posterior_helped_percentages.keys()]
    sample_sizes = sorted(sample_sizes, key=lambda x: int(x))
    rows = []
    for s in sample_sizes:
        pct_post_better = posterior_helped_percentages[s] * 100
        pct_prior_better = prior_helped_percentages[s] * 100
        rows.append({
            "Sample Size": s,
            "% Prior Better": f"{pct_prior_better:.1f}%",
            "% Posterior Better": f"{pct_post_better:.1f}%"
        })
    df = pd.DataFrame(rows)
    latex_table = df.to_latex(index=False, caption="Prior and Posterior Analysis", label="tab:percent_helped")
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    with open(f"{output_dir}/{dataset}_prior_posterior_comparison_table.tex", "w") as f:
        f.write(latex_table)
    return latex_table


def build_combined_error_comparison_table(all_dataset_results, output_dir="analysis_results"):
    """
    Build a combined LaTeX table with sections for each domain.
    Each section shows sample size vs % prior better vs % posterior better.
    """
    # Dataset name mapping for pretty printing
    dataset_name_mapping = {
        'nhanes': 'NHANES',
        'pitchbook': 'Pitchbook',
        'glassdoor': 'Glassdoor'
    }

    # Start building the LaTeX table manually for better control
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Prior and Posterior Analysis Across Domains}")
    latex_lines.append(r"\label{tab:percent_helped_combined}")
    # Use p{} columns for better width control and @{\extracolsep{\fill}} for spreading
    latex_lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"Sample Size & \% Prior Better & \% Posterior Better \\")
    latex_lines.append(r"\hline")

    # Iterate through each dataset
    first_dataset = True
    for dataset_name in sorted(all_dataset_results.keys()):
        dataset_data = all_dataset_results[dataset_name]
        posterior_helped = dataset_data['posterior_helped']
        prior_helped = dataset_data['prior_helped']

        # Add extra spacing before dataset header (except for first one)
        if not first_dataset:
            latex_lines.append(r"[0.5em]")  # Add vertical space
        first_dataset = False

        # Add dataset section header with better spacing
        pretty_name = dataset_name_mapping.get(dataset_name, dataset_name.capitalize())
        latex_lines.append(r"\multicolumn{3}{l}{\textbf{\large " + pretty_name + r"}} \\[0.3em]")

        # Get all sample sizes and sort them
        sample_sizes = sorted(posterior_helped.keys(), key=lambda x: int(x))

        # Add rows for this dataset
        for s in sample_sizes:
            pct_prior_better = prior_helped[s] * 100
            pct_post_better = posterior_helped[s] * 100
            latex_lines.append(f"{s} & {pct_prior_better:.1f}\\% & {pct_post_better:.1f}\\% \\\\")

        # Add spacing between datasets
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular*}")
    latex_lines.append(r"\end{table}")

    latex_table = "\n".join(latex_lines)

    # Write to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/combined_prior_posterior_comparison_table.tex", "w") as f:
        f.write(latex_table)

    return latex_table


def uncertainty_accuracy_correlation_analysis(results_sets, output_dir="analysis_results"): 
    results_sets = [results.copy() for results in results_sets]
    # Only keep non-posterior LLM results (filter out statistical baselines, posteriors) 
    for i in range(len(results_sets)):
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('stat', case=False, na=False)]
        results_sets[i] = results_sets[i][~results_sets[i]['approach'].str.contains('posterior', case=False, na=False)]
    
    all_acc_unc_results = {}
    for results in results_sets:
        dataset_name = results['dataset'].iloc[0]
        
        uncertainty_accuracy_analysis = {}
        # Compute the correlation between accuracy and uncertainty for each LLM-based approach 
        for approach in results['approach'].unique(): 
            if 'stat' not in approach:
                datapoints = results[results['approach'] == approach]
                spearman_corr = datapoints['error_ratio_mean'].corr(datapoints['std'], method='spearman')
                pearson_corr = datapoints['error_ratio_mean'].corr(datapoints['std'], method='pearson')
                uncertainty_accuracy_analysis[approach] = {
                    'spearman_corr': spearman_corr,
                    'pearson_corr': pearson_corr, 
                    'mean_error_ratio': datapoints['error_ratio_mean'].mean(), 
                    'mean_std_ratio': datapoints['std_ratio'].mean()
            }
        # if dataset_name == 'glassdoor': 
        #     print("Uncertainty-Accuracy Analysis for Glassdoor:") 
        #     print(uncertainty_accuracy_analysis)
            # import pdb; pdb.set_trace()
        all_acc_unc_results[dataset_name] = uncertainty_accuracy_analysis
    plot_uncertainty_accuracy_scatterplots(all_acc_unc_results, output_dir="analysis_results")
    return uncertainty_accuracy_analysis


def compute_llm_prior_win_rates_over_n_sample_baselines(results):
    """
    Compute the percentage of times LLM prior outperforms statistical baseline across different sample sizes.
    The LLM prior is compared against each sample-size baseline separately.
    """
    llm_priors = results[results['approach'].str.contains('o4-mini') & ~results['approach'].str.contains('posterior')]
    sample_sizes = [5, 10, 20, 30]
    llm_prior_helped_percentages = {}

    for sample_size in sample_sizes:
        llm_prior_helped = 0
        num_processed = 0

        for var in llm_priors['variable'].unique():
            llm_prior_for_var = llm_priors[llm_priors['variable'] == var]

            # Determine the fitted distribution type for this variable
            fitted_dist_types = llm_prior_for_var['fitted_distribution_type'].dropna().unique()
            is_lognormal = len(fitted_dist_types) > 0 and fitted_dist_types[0] == 'lognormal'

            # Select the appropriate statistical baseline for this sample size and distribution type
            stat_baseline = results[
                (results['variable'] == var) &
                (results['approach'].str.contains('statistical')) &
                (results['sample_size'] == str(sample_size))
            ]

            # Filter by distribution type
            if is_lognormal:
                stat_baseline_typed = stat_baseline[stat_baseline['fitted_distribution_type'] == 'lognormal']
                if not stat_baseline_typed.empty:
                    stat_baseline = stat_baseline_typed
            else:
                stat_baseline_typed = stat_baseline[stat_baseline['fitted_distribution_type'] != 'lognormal']
                if not stat_baseline_typed.empty:
                    stat_baseline = stat_baseline_typed

            llm_errors = llm_prior_for_var['abs_error_from_mean'].dropna()
            stat_errors = stat_baseline['abs_error_from_mean'].dropna()

            if stat_errors.empty or llm_errors.empty:
                continue

            num_processed += 1
            llm_prior_mae = llm_errors.mean()
            stat_mae = stat_errors.mean()

            if llm_prior_mae < stat_mae:
                llm_prior_helped += 1

        llm_prior_helped_percentage = float(llm_prior_helped) / float(num_processed) if num_processed > 0 else 0.0
        llm_prior_helped_percentages[sample_size] = llm_prior_helped_percentage

    return llm_prior_helped_percentages


def preprocess_posteriors(posteriors, results, variables):
    """
    Preprocess posteriors to compute mean, std, abs_error, error_ratio, and std_ratio.
    Follows the same logic as aggregate_results function.
    """
    from scipy import stats

    # Helper functions to compute mean, median, mode for different distributions
    def compute_beta_mean_median_mode(a, b):
        mean = stats.beta.mean(a, b)
        median = stats.beta.median(a, b)
        # Mode exists only if a > 1 and b > 1
        if a > 1 and b > 1:
            mode = (a - 1) / (a + b - 2)
        else:
            mode = np.nan
        return mean, median, mode

    def compute_lognormal_mean_median_mode(row, mu, sigma):
        # Add overflow protection for very large values
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                mean = np.exp(mu + sigma**2 / 2)
                median = np.exp(mu)
                mode = np.exp(mu - sigma**2)
        except (RuntimeWarning, OverflowError):
            # If overflow occurs, return NaN value
            print('row: {}'.format(row))
            print('mu: ', mu, 'sigma: ', sigma)
            import pdb; pdb.set_trace()
            mean = np.nan
            median = np.nan
            mode = np.nan
        return mean, median, mode

    def compute_normal_mean_median_mode(mu, sigma):
        return mu, mu, mu

    # Helper functions to compute std for different distributions
    def compute_normal_std(mu, sigma):
        return sigma

    def compute_beta_std(a, b):
        return stats.beta.std(a, b)

    def compute_lognormal_std(mu, sigma):
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        return np.sqrt(variance)

    # Extract sample_size from approach name for posteriors (e.g., "posterior_N10" -> "10")
    import re
    def extract_sample_size(approach):
        match = re.search(r'_N(\d+|ALL)', approach)
        if match:
            return match.group(1)
        return np.nan

    posteriors.loc[:, 'sample_size'] = posteriors['approach'].apply(extract_sample_size)

    # Identify distribution types
    beta_vars = posteriors['fitted_distribution_type'] == 'beta'
    lognorm_vars = posteriors['fitted_distribution_type'] == 'lognormal'
    norm_vars = posteriors['fitted_distribution_type'] == 'gaussian'

    # Compute mean, median, mode (use .values to avoid index alignment issues)
    if beta_vars.any():
        beta_results = posteriors.loc[beta_vars].apply(
            lambda row: pd.Series(compute_beta_mean_median_mode(row['a'], row['b'])), axis=1).values
        posteriors.loc[beta_vars, ['mean', 'median', 'mode']] = beta_results

    if lognorm_vars.any():
        lognorm_results = posteriors.loc[lognorm_vars].apply(
            lambda row: pd.Series(compute_lognormal_mean_median_mode(row, row['mu'], row['sigma'])), axis=1).values
        posteriors.loc[lognorm_vars, ['mean', 'median', 'mode']] = lognorm_results

    if norm_vars.any():
        norm_results = posteriors.loc[norm_vars].apply(
            lambda row: pd.Series(compute_normal_mean_median_mode(row['mu'], row['sigma'])), axis=1).values
        posteriors.loc[norm_vars, ['mean', 'median', 'mode']] = norm_results

    # Compute std
    if beta_vars.any():
        posteriors.loc[beta_vars, 'std'] = posteriors.loc[beta_vars].apply(
            lambda row: compute_beta_std(row['a'], row['b']), axis=1).values

    if lognorm_vars.any():
        posteriors.loc[lognorm_vars, 'std'] = posteriors.loc[lognorm_vars].apply(
            lambda row: compute_lognormal_std(row['mu'], row['sigma']), axis=1).values

    if norm_vars.any():
        posteriors.loc[norm_vars, 'std'] = posteriors.loc[norm_vars].apply(
            lambda row: compute_normal_std(row['mu'], row['sigma']), axis=1).values

    # Compute absolute errors
    posteriors.loc[:, 'abs_error_from_mean'] = np.abs(posteriors['ground_truth'] - posteriors['mean'])
    posteriors.loc[:, 'abs_error_from_median'] = np.abs(posteriors['ground_truth'] - posteriors['median'])
    posteriors.loc[:, 'abs_error_from_mode'] = np.abs(posteriors['ground_truth'] - posteriors['mode'])

    # Initialize ratio columns
    posteriors.loc[:, 'error_ratio_mean'] = np.nan
    posteriors.loc[:, 'error_ratio_median'] = np.nan
    posteriors.loc[:, 'error_ratio_mode'] = np.nan
    posteriors.loc[:, 'std_ratio'] = np.nan

    # Compute error ratios by comparing to statistical baselines (match by variable, sample_size, trial, and distribution type)
    for idx, row in posteriors.iterrows():
        if "statistical" in row["approach"]:
            continue

        # Match baseline by variable, sample_size, trial, and distribution type (lognormal vs normal)
        baselines = results[
            (results["approach"].str.contains("statistical", na=False)) &
            (results["sample_size"] == row["sample_size"]) &
            (results["variable"] == row["variable"]) &
            (results["trial"] == row["trial"]) &
            (results["fitted_distribution_type"] == row["fitted_distribution_type"])
        ]

        if len(baselines) == 0:
            raise ValueError(
                f"No matching baseline found for var={row['variable']}, "
                f"sample_size={row['sample_size']}, trial={row['trial']}, "
                f"fitted_dist={row['fitted_distribution_type']}"
            )

        baseline = baselines.iloc[0]
        baseline_error = baseline["abs_error_from_mean"]
        posterior_error = row["abs_error_from_mean"]

        if baseline_error > 0:
            posteriors.at[idx, 'error_ratio_mean'] = posterior_error / baseline_error
        if baseline["abs_error_from_median"] > 0:
            posteriors.at[idx, 'error_ratio_median'] = row["abs_error_from_median"] / baseline["abs_error_from_median"]
        if baseline["abs_error_from_mode"] > 0:
            posteriors.at[idx, 'error_ratio_mode'] = row["abs_error_from_mode"] / baseline["abs_error_from_mode"]
        if baseline["std"] > 0:
            posteriors.at[idx, 'std_ratio'] = row["std"] / baseline["std"]

    return posteriors


def compute_error_ratios_and_helped_percentages(result_sets, output_dir="analysis_results"):
    """
    Compare LLM prior and LLM posterior vs statistical baseline.
    Creates a combined table showing win rates for each sample size.
    """
    import json

    all_rows = []

    for results in result_sets:
        dataset_name = results['dataset'].iloc[0]
        var_file_path = "{}data/variables/{}_variables.json".format(os.environ['OPENESTIMATE_ROOT'], dataset_name)
        variables = json.load(open(var_file_path, 'r'))

        # Load posteriors file
        posterior_file = "{}experiments/{}/{}_combined_processed_results_with_posteriors.csv".format(
            os.environ['OPENESTIMATE_ROOT'], dataset_name, dataset_name)
        posterior_results = pd.read_csv(os.path.expanduser(posterior_file), low_memory=False)

        # Get o4-mini posteriors only
        posteriors_only = posterior_results[
            posterior_results['approach'].str.contains('posterior') &
            posterior_results['approach'].str.contains('o4-mini')
        ].copy()

        # Preprocess posteriors to compute mean, abs_error, and error_ratio
        posteriors_processed = preprocess_posteriors(posteriors_only, results, variables)

        # Compute LLM prior win rates (compares prior against each baseline sample size)
        llm_prior_win_rates = compute_llm_prior_win_rates_over_n_sample_baselines(results)

        sample_sizes = ['5', '10', '20', '30']
        for sample_size in sample_sizes:
            # Prior win rate (from the function that compares against each sample size)
            prior_win_pct = llm_prior_win_rates.get(int(sample_size), np.nan) * 100

            # Posterior win rate
            posteriors_for_n = posteriors_processed[posteriors_processed['sample_size'] == sample_size]
            posterior_win_pct = (posteriors_for_n['error_ratio_mean'] < 1).sum() / len(posteriors_for_n) * 100 if len(posteriors_for_n) > 0 else np.nan

            all_rows.append({
                'Domain': dataset_name.capitalize(),
                'Sample Size': int(sample_size),
                'LLM Prior > Stat. Baseline': f"{prior_win_pct:.1f}%",
                'LLM Posterior > Stat. Baseline': f"{posterior_win_pct:.1f}%"
            })

    # Create DataFrame and save
    combined_df = pd.DataFrame(all_rows)

    # Print table
    print(f"\n{'='*80}")
    print("Combined LLM Prior and Posterior Win Rates vs Statistical Baseline")
    print(f"{'='*80}")
    print(combined_df.to_string(index=False))

    # Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "llm_vs_baseline_win_rates.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return combined_df


def compare_models(datasets, output_dir): 
    experiment_name = 'model_family_comparison'
    print("Datasets: {}".format(datasets))
    results_sets = [load_experiment_results(dataset, experiment_name) for dataset in datasets]
    for results in results_sets: 
        print_completion_stats(results)
    compute_error_ratios_and_helped_percentages(results_sets, output_dir)
    # plot_error_ratio_by_domain(results_sets, output_dir)
    # plot_ece_by_domain(results_sets, output_dir)
    # uncertainty_accuracy_correlation_analysis(results_sets)
    # calibration_heat_map(results_sets, output_dir)
    # z_score_cdf_plot(results_sets, output_dir)


