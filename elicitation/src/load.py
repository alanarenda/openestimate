import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, beta


def compute_ground_truth_percentile(df):
    """Compute which percentile of the elicited distribution contains the ground truth."""
    new_df = df.copy()
    for index, row in df.iterrows():
        if row['ground_truth_distribution_type'] == 'gaussian' or row['ground_truth_distribution_type'] == 'normal':
            # Uses norm.cdf to get the percentile from the Gaussian CDF
            new_df.at[index, 'ground_truth_percentile'] = norm.cdf(row['ground_truth'], row['mean'], row['std']) * 100
        elif row['ground_truth_distribution_type'] == 'beta':
            # Uses beta.cdf to get the percentile from the Beta CDF
            try:
                new_df.at[index, 'ground_truth_percentile'] = beta.cdf(row['ground_truth'], row['alpha'], row['beta']) * 100
            except:
                new_df.at[index, 'ground_truth_percentile'] = beta.cdf(row['ground_truth'], row['a'], row['b']) * 100
        else:
            print(f"Warning: Unknown distribution type: {row['ground_truth_distribution_type']} for variable {row['variable_name']}")
            new_df.at[index, 'ground_truth_percentile'] = None
    return new_df


def determine_difficulty(variable_name):
    """Determine difficulty level from variable name."""
    if 'easy' in variable_name.lower():
        return 'easy'
    elif 'medium' in variable_name.lower():
        return 'medium'
    elif 'hard' in variable_name.lower():
        return 'hard'
    elif 'base' in variable_name.lower():
        return 'base'
    else:
        # Instead of returning 'base' as default, let's print a warning
        return 'base'  # Default to base if we can't determine difficulty


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
    return norm.ppf([0.25, 0.5, 0.75], mean, std)


def get_quartiles_from_beta(alpha, beta_param):
    res = beta.ppf([0.25, 0.5, 0.75], alpha, beta_param)
    return res


def determine_quartile_of_gt(sorted_results):
    # Initialize the 'quartile_of_gt' column with default values
    sorted_results['quartile_of_gt'] = None

    for index, row in sorted_results.iterrows():
        if row['ground_truth_distribution_type'] == 'normal' or row['ground_truth_distribution_type'] == 'gaussian':
            if row['mean'] is not None:
                quartiles = get_quartiles_from_gaussian(row['mean'], row['std'])
        elif row['ground_truth_distribution_type'] == 'beta' or row['ground_truth_distribution_type'] == 'binomial':
            # print("Alpha: ", row['alpha'], "Beta: ", row['beta'])
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
        else:
            raise ValueError("Unknown distribution type: ", row['ground_truth_distribution_type'])
            
        if row['ground_truth'] < quartiles[0]: 
            quartile_of_gt = 1
        elif row['ground_truth'] > quartiles[0] and row['ground_truth'] < quartiles[1]:
            quartile_of_gt = 2
        elif row['ground_truth'] > quartiles[1] and row['ground_truth'] < quartiles[2]:
            quartile_of_gt = 3
        elif row['ground_truth'] > quartiles[2]:
            quartile_of_gt = 4
        else: 
            print("Row: ", row)
            print("Quartiles: ", quartiles)
            print("Found NAN for approach: ", row['model'])
            quartile_of_gt = None
        sorted_results.at[index, 'quartile_of_gt'] = quartile_of_gt
    return sorted_results


def load_data(results_dir):
    """Load and process experimental results from multiple EXPERTS directories."""
    
    # Find all directories starting with 'EXPERTS-'
    expert_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('EXPERTS-')]
    print(expert_dirs)
    
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
        print("Experts info path: ", experts_info_path)
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

    print(all_results.columns)
    # Expand variables into individual rows and sort
    expanded_results = all_results.explode('variable_name')
    sorted_results = expanded_results.sort_values(by=['variable', 'variable_name', 'elicitation_protocol', 'model'])

    # Add derived columns
    sorted_results = determine_quartile_of_gt(sorted_results)
    sorted_results = compute_ground_truth_percentile(sorted_results)
    sorted_results['difficulty'] = sorted_results['variable'].apply(determine_difficulty)
    
    sorted_results['approach'] = sorted_results[['model', 'sysprompt_type', 'elicitation_protocol', 'temperature']].apply(
        lambda row: f"{row['model']}_{row['sysprompt_type']}_{row['elicitation_protocol']}_temp{row['temperature']}", 
        axis=1
    )
  
    return sorted_results


if __name__ == "__main__":
    results_dir = "../nhanes/full_results"
    results = load_data(results_dir)
    results.to_csv("all_processed_results.csv")