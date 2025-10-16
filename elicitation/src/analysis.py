import json
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm as normal_dist
from load import load_data, determine_quartile_of_gt


def generate_analysis(dataset):
    """
    Generate analysis results for a specific dataset.
    
    Args:
        dataset (str): Name of the dataset (e.g., 'nhanes', 'labor', 'pitchbook')
        
    Returns:
        pd.DataFrame: Combined and processed results
    """
    # Define dataset-specific paths
    dataset_configs = {
        'nhanes': {
            'results_dirs': [
                f'../{dataset}/part1_eval/part1_eval/trial_0_results',
                f'../{dataset}/part1_eval/part1_eval/trial_1_results',
                f'../{dataset}/part1_eval/part1_eval/trial_2_results',
                f'../{dataset}/part1_eval/part1_eval/trial_3_results',
                f'../{dataset}/part1_eval/part1_eval/trial_4_results'
            ],
            'var_file_paths': [f'../{dataset}/variables_by_difficulty.json'] * 5,
            'datasets': [dataset] * 5,
            'baseline_file_paths': [f'../{dataset}/baselines_by_var.json'] * 5
        },
        'glassdoor': {
            'results_dirs': [
                f'../{dataset}/part1_eval/part1_eval/trial_0_results',
                f'../{dataset}/part1_eval/part1_eval/trial_1_results',
                f'../{dataset}/part1_eval/part1_eval/trial_2_results',
                f'../{dataset}/part1_eval/part1_eval/trial_3_results', 
                f'../{dataset}/part1_eval/part1_eval/trial_4_results'
            ],
            'var_file_paths': [f'../{dataset}/variables_by_difficulty.json'] * 1,
            'datasets': [dataset] * 5,  
            'baseline_file_paths': [f'../{dataset}/baselines_by_var.json'] * 1
        },
        'pitchbook': {
            'results_dirs': [
                f'../{dataset}/part1_eval/part1_eval/trial_0_results',
                f'../{dataset}/part1_eval/part1_eval/trial_1_results',
                f'../{dataset}/part1_eval/part1_eval/trial_2_results',
                f'../{dataset}/part1_eval/part1_eval/trial_3_results',
                f'../{dataset}/part1_eval/part1_eval/trial_4_results'
            ],
            'var_file_paths': [f'../{dataset}/variables_by_difficulty.json'] * 5,
            'datasets': [dataset] * 5,
            'baseline_file_paths': [f'../{dataset}/baselines_by_var.json'] * 5
        }
    }
    
    if dataset not in dataset_configs:
        raise ValueError(f"Dataset '{dataset}' not supported. Supported datasets: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset]
    
    return aggregate_results(
        dataset,
        config['results_dirs'],
        config['var_file_paths'],
        config['datasets'],
        config['baseline_file_paths']
    )


def aggregate_results(dataset, results_dirs, var_file_paths, datasets, baseline_file_paths):
    all_results = []
    variables = json.load(open(var_file_paths[0], 'r'))

    for results_dir, _, dataset in zip(results_dirs, var_file_paths, datasets):
        results = load_data(results_dir)
        results['dataset'] = dataset
        results['trial'] = results_dir.split('/')[-1].split('_')[1]
        print("Number of results: ", len(results))
        all_results.append(results)
    
    # Combine all results
    results = pd.concat(all_results, ignore_index=True)
    print("Final number of results: ", len(results))

    # Process beta variables to compute normalized mean and std
    beta_vars = results[results['ground_truth_distribution_type'] == 'beta'].copy()

    for idx, row in beta_vars.iterrows():
        alpha = row['a'] 
        beta_param = row['b']  

        # Check for None or NaN values
        if pd.isna(alpha) or pd.isna(beta_param) or alpha is None or beta_param is None:
            mean = row['mean']
            std = row['std']
        else: 
            alpha = float(alpha)
            beta_param = float(beta_param)
            mean = beta_dist.mean(alpha, beta_param)
            std = beta_dist.std(alpha, beta_param)
        beta_vars.loc[idx, 'mean'] = mean
        beta_vars.loc[idx, 'std'] = std

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    stat_baselines = json.load(open(baseline_file_paths[0], 'r'))

    stat_baselines_metrics_by_var = {}
    for var, info in stat_baselines.items(): 
        stat_baselines_metrics_by_var[var] = {}
        for n, baseline_infos in info.items(): 
            var_name = variables[var]['variable']
            
            # Store each resampling's metrics separately
            resampling_results = []
            
            # Process each resampling
            trial = 0 
            for baseline_info in baseline_infos:
                var_type = 'beta' if 'alpha' in baseline_info else 'gaussian'
                if var_type == 'beta': 
                    mean = beta_dist.mean(baseline_info['alpha'], baseline_info['beta'])
                    std = beta_dist.std(baseline_info['alpha'], baseline_info['beta'])
                elif var_type == 'gaussian': 
                    mean = baseline_info['mu']
                    std = baseline_info['sigma']

                var_difficulty = 'easy' if 'easy' in var else 'medium' if 'medium' in var else 'hard' if 'hard' in var else 'base'
                ground_truth = variables[var]['mean']
               
                # Store each resampling's results separately
                resampling_results.append({
                    'variable_name': var_name,
                    'variable': var,
                    'ground_truth': ground_truth,
                    'ground_truth_distribution_type': variables[var]['ground_truth_distribution_type'],
                    'mean': mean,
                    'std': std,
                    'a': baseline_info['alpha'] if 'alpha' in baseline_info else None,
                    'b': baseline_info['beta'] if 'beta' in baseline_info else None,
                    'trial': trial,
                    'difficulty': var_difficulty
                })
                trial += 1
            # Store all resampling results for this sample size
            stat_baselines_metrics_by_var[var][n] = resampling_results
    
    flattened_metrics = []
    for var, baselines in stat_baselines_metrics_by_var.items():
        for n, resamplings in baselines.items():
            for resampling in resamplings:
                resampling_copy = resampling.copy()
                resampling_copy['approach'] = f'statistical_baseline_n{n}'  # Add approach name
                flattened_metrics.append(resampling_copy)

    results_df = pd.DataFrame(flattened_metrics)

    combined_results = pd.concat([results, results_df], ignore_index=True)
    combined_results = determine_quartile_of_gt(combined_results)

    # Initialize the abs_error column first
    combined_results['abs_error'] = 0.0

    # Standardize all variables for the sake of downstream relative comparisons 
    for idx, row in combined_results.iterrows():
        var_info = variables[row['variable']]

        if var_info['ground_truth_distribution_type'] == 'normal':
            # For normal variables, use raw prediction
            combined_results.loc[idx, 'abs_error'] = abs(row['mean'] - var_info['mean'])
            
        elif var_info['ground_truth_distribution_type'] == 'beta':
            mean = beta_dist.mean(row['a'], row['b'])
            std = beta_dist.std(row['a'], row['b'])
            # write mean/std into the DataFrame so downstream code and the CSV have them
            combined_results.loc[idx, 'mean'] = mean
            combined_results.loc[idx, 'std'] = std
            combined_results.loc[idx, 'abs_error'] = abs(mean - var_info['mean'])

    for idx, row in combined_results.iterrows(): 
        gt_value = row['ground_truth']
        var_info = variables[row['variable']]
        if row['ground_truth_distribution_type'] == 'beta': 
            # For beta distribution, compute log prob of ground truth under fitted distribution
            log_prob_gt_under_prior = beta_dist.logpdf(gt_value, row['a'], row['b'])
            jeffreys = (1.0/2.0) * np.log((gt_value) * (1 - gt_value))
            log_prob = log_prob_gt_under_prior + jeffreys
            combined_results.loc[idx, 'ground_truth_log_prob'] = log_prob
        elif row['ground_truth_distribution_type'] == 'normal': 
            log_prob_gt_under_prior = normal_dist.logpdf(gt_value, loc=row['mean'], scale=row['std'])
            jeffreys = np.log(var_info['std']) 
            log_prob = log_prob_gt_under_prior + jeffreys
            combined_results.loc[idx, 'ground_truth_log_prob'] = log_prob
        else: 
            raise ValueError(f"Unknown distribution type: {row['ground_truth_distribution_type']}")
        
    combined_results.to_csv(f"{dataset}_combined_processed_results.csv")
    print(f"Results saved to: {dataset}_combined_processed_results.csv")
    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate analysis results for a specific dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['nhanes', 'glassdoor', 'pitchbook'],
                       help='Dataset to analyze (nhanes, labor, or pitchbook)')
    
    args = parser.parse_args()
    
    results = generate_analysis(args.dataset)
    print(f"Analysis completed for dataset: {args.dataset}")
   