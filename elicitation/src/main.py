import json
import shutil
import argparse
from pathlib import Path
from fit_priors import assign_prior_direct, process_priors, fit_prior_quantile, fit_prior_mean_variance
from elicitation import elicit_priors


def get_safe_filename(var_id):
    """Convert variable ID to safe filename by replacing directory separators."""
    safe_id = var_id.replace('/', '_')
    return f"{safe_id}_prior.json"


def load_json_file(file_path):
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def generate_results_name(experiment_spec, experts):
    """Generate standardized results directory name."""
    experts_name = Path(experiment_spec["experts_spec"]).stem
    variables_name = Path(experiment_spec["variables"]).stem
    protocol_name = Path(experiment_spec["protocol_spec"]["individual_elicitation_protocol"]).stem
    temperature = experts["model_kwargs"]["temperature"]
    
    return f"EXPERTS-{experts_name}-VARIABLES-{variables_name}-PROTOCOL-{protocol_name}-TEMP-{temperature}"


def get_protocol_content(experiment_spec):
    """Get protocol content or None for direct protocols."""
    protocol_path = experiment_spec["protocol_spec"]["individual_elicitation_protocol"]
    
    if 'direct' in protocol_path:
        return None
    
    protocol_file = Path(protocol_path).expanduser()
    try:
        with open(protocol_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")


def elicit_all_priors(protocol, variables, experts, cache_dir):
    """Elicit priors for all variables with caching support."""
    all_elicited_priors = {}
    num_new_elicited = 0
    
    for var_id, variable in variables.items():
        var_name = variable["variable"]
        print(f"\nEliciting prior for variable: {var_name}")
        
        cache_file = Path(cache_dir) / get_safe_filename(var_id)
        
        if cache_file.exists():
            # Load from cache
            elicited_prior = load_json_file(cache_file)
        else:
            # Elicit new prior
            elicited_prior = elicit_priors(protocol, [variable], experts)
            num_new_elicited += 1
            
            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(elicited_prior, f, indent=2)
        
        all_elicited_priors[var_name] = elicited_prior
    
    print(f"Elicited {num_new_elicited} new priors")
    return all_elicited_priors


def fit_priors_by_protocol(all_elicited_priors, variables, experiment_spec):
    """Fit priors based on the specified protocol."""
    protocol_name = experiment_spec["protocol_spec"]["individual_elicitation_protocol"]
    
    if 'direct' in protocol_name:
        return assign_prior_direct(all_elicited_priors, variables)
    elif 'quantile' in protocol_name:
        return fit_prior_quantile(all_elicited_priors, variables)
    elif 'mean-variance' in protocol_name:
        return fit_prior_mean_variance(all_elicited_priors, variables)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")


def save_results(results, all_elicited_priors, variables, results_path):
    """Save all experiment results."""
    # Save elicited priors
    elicited_priors_file = Path(results_path) / "elicited_priors.json"
    with open(elicited_priors_file, "w") as f:
        json.dump(all_elicited_priors, f, indent=2)
    
    # Process and save results CSV
    processed_df = process_priors(results, variables)
    csv_file = Path(results_path) / "processed_results.csv"
    processed_df.to_csv(csv_file, index=False)


def copy_config_files(experiment_spec, config_path, results_path):
    """Copy configuration files to results directory."""
    results_dir = Path(results_path)
    
    # Copy experts file
    experts_path = Path(experiment_spec["experts_spec"]).expanduser()
    shutil.copy(experts_path, results_dir / experts_path.name)
    
    # Copy experiment config
    config_file = Path(config_path)
    shutil.copy(config_file, results_dir / config_file.name)


def main():
    parser = argparse.ArgumentParser(description="Run prior elicitation experiment")
    parser.add_argument("--experiment_config", type=str, required=True,
                       help="Path to experiment configuration file")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for results")
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.experiment_config).expanduser()
        experiment_spec = load_json_file(config_path)
        
        # Load experts and variables
        experts_path = Path(experiment_spec["experts_spec"]).expanduser()
        variables_path = Path(experiment_spec["variables"]).expanduser()
        
        experts = load_json_file(experts_path)
        variables = load_json_file(variables_path)
        
        # Get protocol
        protocol = get_protocol_content(experiment_spec)
        
        # Create results directory
        results_name = generate_results_name(experiment_spec, experts)
        results_path = Path(args.output_dir) / results_name
        create_directory(results_path)
        
        # Create cache directory
        cache_dir = results_path / "elicitation_cache"
        create_directory(cache_dir)
        
        # Elicit priors
        all_elicited_priors = elicit_all_priors(protocol, variables, experts, cache_dir)
        
        # Fit priors
        results = fit_priors_by_protocol(all_elicited_priors, variables, experiment_spec)
        
        # Save results
        save_results(results, all_elicited_priors, variables, results_path)
        
        # Copy configuration files
        copy_config_files(experiment_spec, config_path, results_path)
        
        print("Experiment completed successfully!")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()