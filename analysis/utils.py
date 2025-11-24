import os
import re 
import tqdm
import glob
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, List
from scipy.stats import beta as beta_dist, norm as normal_dist


def dataframe_to_latex(df, escape=True, float_fmt="%.4f", caption="", label="", **kwargs):
    """
    Convert DataFrame to LaTeX format with error handling.
    
    Args:
        df: DataFrame to convert
        escape: Whether to escape special characters
        float_fmt: Format string for float values
        caption: Table caption
        label: Table label
        **kwargs: Additional arguments for to_latex()
    
    Returns:
        str: LaTeX formatted string
    """
    try:
        latex_str = df.to_latex(escape=escape, float_format=float_fmt, **kwargs)
        
        # Add caption and label if provided
        if caption or label:
            # Find the end of the table and insert caption/label
            lines = latex_str.split('\n')
            new_lines = []
            in_table = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.strip() == '\\begin{table}':
                    in_table = True
                elif in_table and line.strip() == '\\end{table}':
                    # Insert caption and label before closing table
                    if label:
                        new_lines.insert(-1, f"\\label{{{label}}}")
                    if caption:
                        new_lines.insert(-1, f"\\caption{{{caption}}}")
                    in_table = False
            
            latex_str = '\n'.join(new_lines)
        
        return latex_str
    except Exception as e:
        warnings.warn(f"Error converting DataFrame to LaTeX: {e}")
        return f"% Error generating LaTeX: {e}\n% DataFrame shape: {df.shape}\n% Columns: {list(df.columns)}"


def compute_ground_truth_percentile(df):
    """Compute which percentile of the elicited distribution contains the ground truth."""
    new_df = df.copy()
    for index, row in df.iterrows():
        if row['ground_truth_distribution_type'] == 'gaussian' or row['ground_truth_distribution_type'] == 'normal':
            # Uses norm.cdf to get the percentile from the Gaussian CDF
            new_df.at[index, 'ground_truth_percentile'] = normal_dist.cdf(row['ground_truth'], row['mean'], row['std']) * 100
        elif row['ground_truth_distribution_type'] == 'beta':
            # Uses beta.cdf to get the percentile from the Beta CDF
            try:
                new_df.at[index, 'ground_truth_percentile'] = beta_dist.cdf(row['ground_truth'], row['alpha'], row['beta']) * 100
            except:
                new_df.at[index, 'ground_truth_percentile'] = beta_dist.cdf(row['ground_truth'], row['a'], row['b']) * 100
        else:
            print(f"Warning: Unknown distribution type: {row['ground_truth_distribution_type']} for variable {row['variable_name']}")
            new_df.at[index, 'ground_truth_percentile'] = None
    return new_df


def find_experts_file(results_dir, experts_filename):
    """
    Look for experts file in subdirectories of results_dir and in sibling directories.
    
    Args:
        results_dir: Path to the results directory
        experts_filename: Name of the experts file to find (e.g. 'nhanes_gpt-4o_conservative_temp0.0_experts.json')
        
    Returns:
        Path to the experts file if found
    
    Raises:
        FileNotFoundError if the file cannot be found
    """
    # Check if OPENESTIMATE_ROOT is set and use it to resolve paths
    openestimate_root = os.environ.get('OPENESTIMATE_ROOT')
    if openestimate_root:
        # Try to find the file using OPENESTIMATE_ROOT
        root_experts_path = os.path.join(openestimate_root, experts_filename)
        if os.path.exists(root_experts_path):
            return root_experts_path
            
        # Also check in common subdirectories under OPENESTIMATE_ROOT
        common_dirs = ['src/experiments', 'data', 'experiments']
        for common_dir in common_dirs:
            common_path = os.path.join(openestimate_root, common_dir, experts_filename)
            if os.path.exists(common_path):
                return common_path
    
    # First try to find in subdirectories (original approach)
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path):
            experts_path = os.path.join(subdir_path, experts_filename)
            if os.path.exists(experts_path):
                # print(f"Found {experts_filename} in {subdir_path}")
                return experts_path
    
    # If not found, check sibling directories
    # Convert to absolute path first to avoid empty parent_dir
    abs_results_dir = os.path.abspath(results_dir)
    parent_dir = os.path.dirname(abs_results_dir)
    
    if not parent_dir:
        raise FileNotFoundError(f"Could not find experts file {experts_filename} in any subdirectory of {results_dir}")
        
    # print(f"Looking for {experts_filename} in sibling directories of {results_dir}...")
    
    for sibling_dir in os.listdir(parent_dir):
        sibling_path = os.path.join(parent_dir, sibling_dir)
        # Skip if not a directory or if it's the original results dir
        if not os.path.isdir(sibling_path) or sibling_path == abs_results_dir:
            continue
            
        # Check in the sibling directory root
        experts_path = os.path.join(sibling_path, experts_filename)
        if os.path.exists(experts_path):
            # print(f"Found {experts_filename} in sibling directory {sibling_path}")
            return experts_path
            
        # Also check in subdirectories of each sibling
        for subdir in os.listdir(sibling_path):
            sub_path = os.path.join(sibling_path, subdir)
            if not os.path.isdir(sub_path):
                continue
                
            experts_path = os.path.join(sub_path, experts_filename)
            if os.path.exists(experts_path):
                # print(f"Found {experts_filename} in {sub_path}")
                return experts_path
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError(f"Could not find experts file {experts_filename} in any subdirectory of {results_dir} or in sibling directories")


def get_quartiles_from_gaussian(mean, std):
    return normal_dist.ppf([0.25, 0.5, 0.75], mean, std)


def get_quartiles_from_beta(alpha, beta_param):
    res = beta_dist.ppf([0.25, 0.5, 0.75], alpha, beta_param)
    return res



def load_data(results_dir):
    """Load and process experimental results from multiple EXPERTS directories."""
    
    # Find all directories starting with 'EXPERTS-'
    expert_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('EXPERTS-')]    
    # Create an empty DataFrame to store all results
    all_results = pd.DataFrame()

    # Process each expert directory
    for dir_name in expert_dirs:
        csv_path = os.path.join(results_dir, dir_name, 'processed_results.csv')
        exp_spec_files = glob.glob(os.path.join(results_dir, dir_name, '*exp-spec*.json'))
        convo_file_path = os.path.join(results_dir, dir_name, 'elicited_priors.json')
        
        # Check if required files exist
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            continue
            
        if not exp_spec_files:
            print(f"Exp Spec file not found in: {results_dir}/{dir_name}")
            continue
            
        exp_spec_file = exp_spec_files[0]
        
        # Load experiment specification and results
        exp_spec = json.load(open(exp_spec_file, 'r'))
        results = pd.read_csv(csv_path, index_col=False)
        
        # Extract metadata from exp_spec
        vars_file = exp_spec['variables']
        expert_filename = exp_spec['experts_spec'].split('/')[-1]
        sysprompt_type = expert_filename.split('_')[2]
        
        # Extract protocol name from path (similar to main.py line 44)
        protocol_path = exp_spec['protocol_spec']['individual_elicitation_protocol']
        if 'direct' in protocol_path:
            elicitation_protocol = 'direct'
        else:
            elicitation_protocol = protocol_path.split("/")[-1].split(".")[0]
        
        # Extract model from experiment name or expert filename
        model = exp_spec['experiment_name'].split('_')[0]  # e.g., "o3-mini" from "o3-mini_base_direct_temp0.2"
        
        # Load experts info and extract temperature
        experts_info_path = exp_spec['experts_spec']
        # Check if OPENESTIMATE_ROOT is set and use it to resolve paths
        openestimate_root = os.environ.get('OPENESTIMATE_ROOT')
        if openestimate_root:
            # Replace everything up to and including "openestimate" with OPENESTIMATE_ROOT
            if 'openestimate' in experts_info_path:
                # Find the position of "openestimate" in the path
                openestimate_pos = experts_info_path.find('openestimate')
                # Get everything after "openestimate" in the path
                path_after_openestimate = experts_info_path[openestimate_pos + len('openestimate'):]
                # Construct new path using OPENESTIMATE_ROOT
                experts_info_path = openestimate_root + path_after_openestimate
        else:
            # Fall back to original behavior
            experts_info_path = os.path.expanduser(experts_info_path)
            
        experts_info = json.load(open(experts_info_path, 'r'))
        
        if 'model_kwargs' not in experts_info or 'temperature' not in experts_info['model_kwargs']:
            raise ValueError(f"Temperature not found in experts file {expert_filename}")
            
        temperature = experts_info['model_kwargs']['temperature']
        
        # Add metadata to results
        results['temperature'] = temperature
        results['variables_file'] = vars_file
        results['sysprompt_type'] = sysprompt_type
        results['elicitation_protocol'] = elicitation_protocol  
        results['model'] = model  
        results['convo_file_path'] = convo_file_path
        
        # Append to all_results
        all_results = pd.concat([all_results, results], ignore_index=True)

    # Clean up unnecessary columns
    if 'Unnamed: 0' in all_results.columns:
        all_results.drop(columns=['Unnamed: 0'], inplace=True)

    # Expand variables into individual rows and sort
    expanded_results = all_results.explode('variable_name')
    sorted_results = expanded_results.sort_values(by=['variable', 'variable_name', 'elicitation_protocol', 'model'])

    # Add derived columns
    # sorted_results = determine_quartile_of_gt(sorted_results)
    # sorted_results = compute_ground_truth_percentile(sorted_results)
    
    sorted_results['approach'] = sorted_results[['model', 'sysprompt_type', 'elicitation_protocol', 'temperature']].apply(
        lambda row: f"{row['model']}_{row['sysprompt_type']}_{row['elicitation_protocol']}_temp{row['temperature']}", 
        axis=1
    )
  
    return sorted_results


def compute_error_ratios_and_std_ratios(results): 
    # Compute error ratio for all LLM-based approaches 
    results = results.copy()
    for approach in results['approach'].unique(): 
        if 'stat' not in approach:
            for var in results['variable'].unique():
                five_sample_stat_baseline_mae = results[
                    (results['variable'] == var) &
                    (results['approach'].str.contains('statistical_baseline_n5'))
                ]['abs_error'].mean()
                llm_mae = results[(results['approach'] == approach) & (results['variable'] == var)]['abs_error'].mean()
                prior_error_ratio = llm_mae / five_sample_stat_baseline_mae 
                results.loc[
                    (results['approach'] == approach) &
                    (results['variable'] == var),
                    'error_ratio'
                ] = prior_error_ratio

                five_sample_stat_baseline_std = results[
                    (results['variable'] == var) &
                    (results['approach'].str.contains('statistical_baseline_n5'))
                ]['std'].mean()
                llm_std = results[(results['approach'] == approach) & (results['variable'] == var)]['std'].mean()
                std_ratio = llm_std / five_sample_stat_baseline_std 
                results.loc[
                    (results['approach'] == approach) &
                    (results['variable'] == var),
                    'std_ratio'
                ] = std_ratio
    return results 


def compute_normal_mean_median_mode(mu, sigma):
    return {'mean': mu, 'median': mu, 'mode': mu}


def compute_beta_mean_median_mode(a, b): 
    # Mean of Beta(a, b) = a / (a + b)
    mean = a / (a + b)
    
    # Median - no closed form, use scipy
    median = stats.beta.median(a, b)
    
    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
    elif a == 1 and b == 1:
        mode = np.nan  # uniform distribution
    elif a <= 1 and b > 1:
        mode = 0.0
    elif a > 1 and b <= 1:
        mode = 1.0
    else:  # a < 1 and b < 1 (bimodal)
        mode = np.nan  # or could return [0.0, 1.0

    return {
        'mean': mean,
        'median': median,
        'mode': mode  
    }


def compute_lognormal_mean_median_mode(mu, sigma):
    """
    Compute mean, median, and mode of a Lognormal distribution.
    
    Parameters:
    -----------
    mu : float
        Mean of the underlying normal distribution
    sigma : float
        Standard deviation of the underlying normal distribution, must be > 0
    
    Returns:
    --------
    dict with keys 'mean', 'median', 'mode'
    """
    import numpy as np
    
    # Mean of Lognormal(mu, sigma) = exp(mu + sigma^2/2)
    mean = np.exp(mu + sigma**2 / 2)
    
    # Median of Lognormal(mu, sigma) = exp(mu)
    median = np.exp(mu)
    
    # Mode of Lognormal(mu, sigma) = exp(mu - sigma^2)
    mode = np.exp(mu - sigma**2)
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode
    }


def get_quartiles_from_gaussian(mean, std):
    return normal_dist.ppf([0.25, 0.5, 0.75], mean, std)


def get_quartiles_from_beta(alpha, beta_param):
    res = beta_dist.ppf([0.25, 0.5, 0.75], alpha, beta_param)
    return res


def get_quartiles_from_lognormal(mu, sigma):
    return np.exp(stats.norm.ppf([0.25, 0.5, 0.75], mu, sigma))


def determine_quartile_of_gt(results):
    # Initialize the 'quartile_of_gt' column with default values
    results['quartile_of_gt'] = np.nan 

    for index, row in results.iterrows():
        if row['fitted_distribution_type'] == 'normal' or row['fitted_distribution_type'] == 'gaussian':
            if pd.notna(row['mu']) and pd.notna(row['sigma']):
                quartiles = get_quartiles_from_gaussian(row['mu'], row['sigma'])
        elif row['fitted_distribution_type'] == 'beta' or row['fitted_distribution_type'] == 'binomial':
            try: 
                alpha = row['alpha']
                beta = row['beta']
            except:
                alpha = row['a']
                beta = row['b']

            if alpha == 0: 
                alpha = 0.00001
            if beta == 0:
                beta = 0.00001
            if alpha == 0:
                alpha = 0.00001
            quartiles = get_quartiles_from_beta(alpha, beta)
        elif row['fitted_distribution_type'] == 'lognormal':
            if 'sigma' in row:
                quartiles = get_quartiles_from_lognormal(row['mu'], row['sigma'])
            else: 
                print(row)
                quartiles = get_quartiles_from_lognormal(row['mu'], row['std']) 
        else:
            if row['ground_truth_distribution_type'] == 'gaussian':
                quartiles = get_quartiles_from_gaussian(row['mu'], row['sigma'])
            elif row['ground_truth_distribution_type'] == 'beta':
                try: 
                    alpha = row['alpha']
                    beta = row['beta']
                except:
                    alpha = row['a']
                    beta = row['b']

                if alpha == 0: 
                    alpha = 0.00001
                if beta == 0:
                    beta = 0.00001
                quartiles = get_quartiles_from_beta(alpha, beta)
            else:
                raise ValueError("Unknown distribution type: ", row['ground_truth_distribution_type'])
            
        if row['ground_truth'] <= quartiles[0]:
            quartile_of_gt = 1
        elif row['ground_truth'] <= quartiles[1]:
            quartile_of_gt = 2
        elif row['ground_truth'] <= quartiles[2]:
            quartile_of_gt = 3
        else:
            quartile_of_gt = 4
        results.at[index, 'quartile_of_gt'] = quartile_of_gt
    return results


def aggregate_results(dataset, results_dirs, var_file_path, baselines_file_path):
    all_results = []
    variables = json.load(open(var_file_path, 'r'))
    for results_dir in results_dirs:
        results_dir = str(results_dir)
        results = load_data(results_dir)
        results['dataset'] = dataset
        results['trial'] = results_dir.split('/')[-1].split('_')[1]
        all_results.append(results)
    
    # Combine all results
    if len(all_results) > 1:
        results = pd.concat(all_results, ignore_index=True)
    else:
        results = all_results[0]

    stat_baselines = json.load(open(baselines_file_path, 'r'))

    stat_baselines_metrics_by_var = {}
    for var, info in stat_baselines.items():
        stat_baselines_metrics_by_var[var] = {}
        for n, baseline_infos in info.items():
            var_name = variables[var]['variable']

            # Store each resampling's metrics separately
            resampling_results = []

            # Determine if this is a lognormal baseline
            is_lognormal = 'lognorm' in str(n)
            # Extract the numeric sample size (e.g., "5" from "5_lognorm" or "5" from "5")
            n_numeric = str(n).replace('_lognorm', '')

            # Process each resampling
            trial = 0
            for baseline_info in baseline_infos:
                var_type = 'beta' if 'alpha' in baseline_info else 'gaussian'
                var_difficulty = 'easy' if 'easy' in var else 'medium' if 'medium' in var else 'hard' if 'hard' in var else 'base'
                ground_truth = variables[var]['mean']

                # Determine ground truth distribution type based on the baseline type
                if is_lognormal:
                    ground_truth_dist = 'lognormal'
                elif var_type == 'gaussian':
                    ground_truth_dist = 'gaussian'
                else:
                    ground_truth_dist = 'beta'

                # Store each resampling's results separately
                resampling_results.append({
                    'variable_name': var_name,
                    'variable': var,
                    'ground_truth': ground_truth,
                    'ground_truth_distribution_type': ground_truth_dist,
                    'fitted_distribution_type': ground_truth_dist,
                    'mu': baseline_info.get('mu') if var_type == 'gaussian' else None,
                    'sigma': baseline_info.get('sigma', baseline_info.get('std')) if var_type == 'gaussian' else None,
                    'a': baseline_info['alpha'] if 'alpha' in baseline_info else None,
                    'b': baseline_info['beta'] if 'beta' in baseline_info else None,
                    'trial': trial,
                    'difficulty': var_difficulty,
                    'sample_size': n_numeric
                })
                trial += 1
            # Store all resampling results for this sample size
            stat_baselines_metrics_by_var[var][n] = resampling_results

    flattened_metrics = []
    for var, baselines in stat_baselines_metrics_by_var.items():
        for n, resamplings in baselines.items():
            # Extract numeric sample size (works for both "5" and "5_lognorm")
            n_numeric = str(n).replace('_lognorm', '')

            for resampling in resamplings:
                resampling_copy = resampling.copy()
                resampling_copy['approach'] = f'statistical_baseline_n{n_numeric}'
                flattened_metrics.append(resampling_copy)

    results_df = pd.DataFrame(flattened_metrics)

    results = pd.concat([results, results_df], ignore_index=True)

    lognorm_mask = results['fitted_distribution_type'] == 'lognormal'
    results.loc[lognorm_mask, 'ground_truth'] = results.loc[lognorm_mask].apply(
        lambda row: variables[row['variable']].get('mean_lognormal', variables[row['variable']]['mean']), 
        axis=1
    )

    beta_mask = results['fitted_distribution_type'] == 'beta'
    results.loc[beta_mask, 'ground_truth'] = results.loc[beta_mask].apply(
        lambda row: variables[row['variable']].get('mean'), 
        axis=1
    )

    norm_mask = results['fitted_distribution_type'] == 'gaussian'
    results.loc[norm_mask, 'ground_truth'] = results.loc[norm_mask].apply(
        lambda row: variables[row['variable']].get('mean'), 
        axis=1
    )

    beta_vars = results['fitted_distribution_type'] == 'beta'
    lognorm_vars = results['fitted_distribution_type'] == 'lognormal'
    norm_vars = results['fitted_distribution_type'] == 'gaussian'

    results['mean'] = np.nan
    results['median'] = np.nan 
    results['mode'] = np.nan 

    results.loc[beta_vars, ['mean', 'median', 'mode']] = results.loc[beta_vars].apply(
        lambda row: pd.Series(compute_beta_mean_median_mode(row['a'], row['b'])), axis=1)

    results.loc[lognorm_vars, ['mean', 'median', 'mode']] = results.loc[lognorm_vars].apply(
        lambda row: pd.Series(compute_lognormal_mean_median_mode(row['mu'], row['sigma'])), axis=1)

    results.loc[norm_vars, ['mean', 'median', 'mode']] = results.loc[norm_vars].apply(
        lambda row: pd.Series(compute_normal_mean_median_mode(row['mu'], row['sigma'])), axis=1)

    results['abs_error_from_mean'] = np.abs(results['ground_truth'] - results['mean'])
    results['abs_error_from_median'] = np.abs(results['ground_truth'] - results['median'])
    results['abs_error_from_mode'] = np.abs(results['ground_truth'] - results['mode'])

    results = determine_quartile_of_gt(results)
    # Compute std for each distribution type
    def compute_normal_std(mu, sigma):
        return sigma

    def compute_beta_std(a, b):
        return stats.beta.std(a, b)

    def compute_lognormal_std(mu, sigma):
        """
        Compute std of a Lognormal distribution.
        
        For Lognormal(mu, sigma):
        Var(X) = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        Std(X) = sqrt(Var(X))
        """
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        return np.sqrt(variance)

    # Compute std for all rows
    results['std'] = np.nan

    results.loc[beta_vars, 'std'] = results.loc[beta_vars].apply(
        lambda row: compute_beta_std(row['a'], row['b']), axis=1)

    results.loc[lognorm_vars, 'std'] = results.loc[lognorm_vars].apply(
        lambda row: compute_lognormal_std(row['mu'], row['sigma']), axis=1)

    results.loc[norm_vars, 'std'] = results.loc[norm_vars].apply(
        lambda row: compute_normal_std(row['mu'], row['sigma']), axis=1)

    fallback_count = 0
    exact_match_count = 0

    results['error_ratio_mean'] = np.nan
    results['error_ratio_median'] = np.nan
    results['error_ratio_mode'] = np.nan
    results['std_ratio'] = np.nan
    results['associated_baseline_error_mean'] = np.nan
    results['associated_baseline_error_median'] = np.nan
    results['associated_baseline_error_mode'] = np.nan
    results['associated_baseline_std'] = np.nan

    for idx, row in results.iterrows():
        if "statistical" not in row["approach"]:
            # First try: Match baselines with same variable, sample_size=5, and distribution type
            baselines = results[
                (results["approach"].str.contains("statistical", na=False)) & 
                (results["sample_size"] == "5") &
                (results["variable"] == row["variable"]) & 
                (results["ground_truth_distribution_type"] == row["fitted_distribution_type"])
            ]
            
            # Fallback: If no exact match, use any available baseline for this variable
            if len(baselines) == 0:
                baselines = results[
                    (results["approach"].str.contains("statistical", na=False)) & 
                    (results["sample_size"] == "5") &
                    (results["variable"] == row["variable"])
                ]
                if len(baselines) > 0:
                    fallback_count += 1
            else:
                exact_match_count += 1
            
            if len(baselines) == 0:
                continue
                
            baseline_avg_error_mean = baselines["abs_error_from_mean"].mean()
            baseline_avg_error_median = baselines["abs_error_from_median"].mean()
            baseline_avg_error_mode = baselines["abs_error_from_mode"].mean()
            baseline_avg_std = baselines["std"].mean()

            error_ratio_mean = row["abs_error_from_mean"] / baseline_avg_error_mean 
            error_ratio_median = row["abs_error_from_median"] / baseline_avg_error_median
            error_ratio_mode = row["abs_error_from_mode"] / baseline_avg_error_mode
            std_ratio = row["std"] / baseline_avg_std

            results.at[idx, 'associated_baseline_error_mean'] = baseline_avg_error_mean
            results.at[idx, 'associated_baseline_error_median'] = baseline_avg_error_median
            results.at[idx, 'associated_baseline_error_mode'] = baseline_avg_error_mode
            results.at[idx, 'associated_baseline_std'] = baseline_avg_std
            results.at[idx, 'error_ratio_mean'] = error_ratio_mean
            results.at[idx, 'error_ratio_median'] = error_ratio_median
            results.at[idx, 'error_ratio_mode'] = error_ratio_mode
            results.at[idx, 'std_ratio'] = std_ratio
        
    print(f"\nExact distribution type matches: {exact_match_count}")
    print(f"Fallback to any available baseline: {fallback_count}")


    results.to_csv(os.path.expanduser("{}experiments/{dataset}/{dataset}_combined_processed_results.csv".format(os.environ['OPENESTIMATE_ROOT'], dataset=dataset)), index=False)
    print(f"Results saved to: {os.path.expanduser('{}experiments/{dataset}/{dataset}_combined_processed_results.csv'.format(os.environ['OPENESTIMATE_ROOT'], dataset=dataset))}")
    return results, variables


def load_experiment_results(dataset, experiment_name): 
    # Get project root (assuming this file is in openestimate/analysis/)
    print("loading results for dataset: ", dataset)
    project_root = Path(__file__).parent.parent
    base_path = project_root / 'experiments' / dataset
    results_dir = base_path / experiment_name / dataset / experiment_name
    results_dirs = list(results_dir.glob('trial_*'))
    print('Number of trials found: ', len(results_dirs))
    var_file_path = project_root / 'data' / 'variables' / f'{dataset}_variables.json'
    baseline_file_path = project_root / 'data' / 'baselines' / f'{dataset}_baselines.json'
    results, variables = aggregate_results(dataset, results_dirs, var_file_path, baseline_file_path) 
    return results


def print_completion_stats(results): 
    print("Completion Statistics")
    signal_col = 'fitted_distribution_type'  # Column indicating if a response was given 
    completion_stats = (
        results
            .assign(answered=results[signal_col].notna())   # True / False per row
            .groupby(['approach'])['answered']     # group then take boolean col
            .mean()                                         # fraction answered
            .mul(100)                                       # convert to %
            .reset_index(name='completion_rate')            # nice tidy frame
    )
    print("\nCompletion-rate by model (%):")
    print("\nApproach                                   Completion %")
    print("-" * 65)
    for _, row in completion_stats.sort_values(['approach']).iterrows():
        print(f"{row['approach']:<40} {row['completion_rate']:>8.1f}")