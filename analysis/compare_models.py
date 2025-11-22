import os
import warnings
import numpy as np
import pandas as pd
from utils import load_experiment_results, print_completion_stats
from plotting import (plot_uncertainty_accuracy_scatterplots,
                plot_error_ratio_by_domain, calibration_heat_map, z_score_cdf_plot, plot_ece_by_domain)


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
    all_acc_unc_results = {}
    for results in results_sets:
        dataset_name = results['dataset'].iloc[0]
        uncertainty_accuracy_analysis = {}
        # Compute the correlation between accuracy and uncertainty for each LLM-based approach 
        for approach in results['approach'].unique(): 
            if 'stat' not in approach:
                datapoints = results[results['approach'] == approach]
                spearman_corr = datapoints['error_ratio'].corr(datapoints['std'], method='spearman')
                pearson_corr = datapoints['error_ratio'].corr(datapoints['std'], method='pearson')
                uncertainty_accuracy_analysis[approach] = {
                    'spearman_corr': spearman_corr,
                    'pearson_corr': pearson_corr, 
                    'mean_error_ratio': datapoints['error_ratio'].mean(), 
                    'mean_std_ratio': datapoints['std_ratio'].mean()
            }
        all_acc_unc_results[dataset_name] = uncertainty_accuracy_analysis
    plot_uncertainty_accuracy_scatterplots(all_acc_unc_results, output_dir="analysis_results")
    return uncertainty_accuracy_analysis


def compute_error_ratios_and_helped_percentages(result_sets, output_dir="analysis_results"):
    result_sets = [results.copy() for results in result_sets]
    all_dataset_results = {}

    for results in result_sets:
        dataset_name = results['dataset'].iloc[0]
        results = pd.read_csv(os.path.expanduser(f"~/openestimate/experiments/{dataset_name}/results_with_posteriors.csv"))
        posteriors = results[results['approach'].str.contains('posterior') | results['approach'].str.contains('statistical')]
        print(f"Dataset: {dataset_name}, Total rows in posteriors: {len(posteriors[posteriors['approach'].str.contains('posterior')])}")
        o4_mini_posteriors = posteriors[posteriors['approach'].str.contains('o4-mini') | posteriors['approach'].str.contains('statistical')]
        sample_sizes = [5, 10, 20, 30]
        all_posterior_helped_percentages = {}
        all_prior_helped_percentages = {}
        o4_mini_priors = results[results['approach'].str.contains('o4-mini') & ~results['approach'].str.contains('posterior')]
        for sample_size in sample_sizes:
            posteriors_with_N = o4_mini_posteriors[o4_mini_posteriors['approach'].str.contains(f'N{sample_size}') | o4_mini_posteriors['approach'].str.contains(f'n{sample_size}')]
            num_skipped = 0
            llm_posterior_helped = 0
            llm_prior_helped = 0
            num_processed = 0
            for var in posteriors_with_N['variable'].unique():
                posteriors_with_n = posteriors_with_N[posteriors_with_N['variable'] == var]
                stat_baseline = posteriors_with_n[posteriors_with_n['approach'].str.contains('statistical')]
                llm_baseline = posteriors_with_n[posteriors_with_n['approach'].str.contains('o4-mini')]
                llm_prior = o4_mini_priors[o4_mini_priors['variable'] == var]
                stat_means = stat_baseline['abs_error'].dropna()
                llm_means = llm_baseline['abs_error'].dropna()
                if stat_means.empty or llm_means.empty:
                    num_skipped += 1
                    continue
                num_processed += 1
                llm_posterior_mae = llm_means.mean()
                llm_prior_mae = llm_prior['abs_error'].mean()
                stat_mae = stat_means.mean()
                if llm_posterior_mae < stat_mae:
                    llm_posterior_helped += 1
                if llm_prior_mae < stat_mae:
                    llm_prior_helped += 1
            llm_prior_helped_percentage = float(llm_prior_helped) / float(num_processed) if num_processed > 0 else 0.0
            llm_posterior_helped_percentage = float(llm_posterior_helped) / float(num_processed) if num_processed > 0 else 0.0
            all_posterior_helped_percentages[sample_size] = llm_posterior_helped_percentage
            all_prior_helped_percentages[sample_size] = llm_prior_helped_percentage

        # Store results for this dataset
        all_dataset_results[dataset_name] = {
            'posterior_helped': all_posterior_helped_percentages,
            'prior_helped': all_prior_helped_percentages
        }
    # Build combined table with all datasets
    build_combined_error_comparison_table(all_dataset_results, output_dir=output_dir)  


def compare_models(datasets, output_dir): 
    experiment_name = 'model_family_comparison'
    results_sets = [load_experiment_results(dataset, experiment_name) for dataset in datasets]
    for results in results_sets: 
        print_completion_stats(results)
    compute_error_ratios_and_helped_percentages(results_sets, output_dir)
    plot_error_ratio_by_domain(results_sets, output_dir)
    plot_ece_by_domain(results_sets, output_dir)
    uncertainty_accuracy_correlation_analysis(results_sets)
    calibration_heat_map(results_sets, output_dir)
    z_score_cdf_plot(results_sets, output_dir)


