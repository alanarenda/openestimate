from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
from utils import (
    SUBSAMPLE_SIZES, RESAMPLES_PER_N, ALPHA0, BETA0, MU0, SIGMA0,
    save_sample_to_csv, gaussian_posterior, beta_posterior
)
from pitchbook_generation import (
    target_variables_boolean,
    apply_conditions_get_data_subset,
    pitchbook_load
)


def compute_baselines_for_variables(
    df: pd.DataFrame,
    variables: Dict[str, Dict],
    dataset_name: str = 'pitchbook'
) -> Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]]:
    """
    Compute baselines for a given set of variables.

    Args:
        df: The full dataset (preprocessed)
        variables: Dictionary of variable specifications with structure:
            {
                'var_key': {
                    'base_variable': str,
                    'conditions': List[Tuple[str, int]],
                    ...
                },
                ...
            }
        dataset_name: Name of the dataset (used for saving samples)

    Returns:
        Dictionary of baselines with structure:
            {
                'var_key': {
                    'n': [{'alpha': ..., 'beta': ...}] or [{'mu': ..., 'sigma': ...}],
                    ...
                },
                ...
            }
    """
    baselines: Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]] = {}

    for var_key, spec in variables.items():
        base_var = spec["base_variable"]
        conditions = spec.get("conditions", [])

        # Get the subset of data matching the conditions
        subset, descriptions = apply_conditions_get_data_subset(df, base_var, conditions)
        if subset is None or subset.empty:
            print(f"Warning: No valid subset for variable '{var_key}'. Skipping.")
            continue

        # Keep only the target variable, drop missing
        subset = subset[[base_var]].dropna(subset=[base_var])
        baselines[var_key] = {}

        is_boolean = base_var in target_variables_boolean

        # Compute baselines for each subsample size
        for n in SUBSAMPLE_SIZES:
            if len(subset) < n:
                continue

            trials: List[Dict[str, float]] = []
            lognorm_trials: List[Dict[str, float]] = []

            for trial_idx in range(RESAMPLES_PER_N):
                samp = subset.sample(
                    n=n,
                    replace=False,
                    random_state=None,
                )

                # Save the sample to CSV for reproducibility
                save_sample_to_csv(dataset_name, samp, var_key, n, trial_idx)

                n_eff = len(samp)

                if is_boolean:
                    # Boolean variable: Beta-Bernoulli update
                    p_hat = float(samp[base_var].mean())
                    s_eff = p_hat * n_eff
                    alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                    trials.append({"alpha": alpha, "beta": beta})

                else:
                    # Continuous variable: Normal update
                    mean_hat = float(samp[base_var].mean())
                    pop_sd = float(samp[base_var].std())
                    mu_n, sig_n = gaussian_posterior(
                        MU0, SIGMA0 * pop_sd, n_eff, mean_hat, pop_sd
                    )

                    # Lognormal update
                    log_values = np.log(samp[base_var] + 1e-6)  # avoid log(0)
                    mean_log_hat = float(log_values.mean())
                    pop_log_sd = float(log_values.std())
                    # Use log-space priors
                    MU0_LOG = 0.0
                    SIGMA0_LOG = 10.0 * pop_log_sd
                    mu_n_log, sig_n_log = gaussian_posterior(
                        MU0_LOG, SIGMA0_LOG, n_eff, mean_log_hat, pop_log_sd
                    )

                    trials.append({"mu": mu_n, "sigma": sig_n})
                    lognorm_trials.append({"mu": mu_n_log, "sigma": sig_n_log})

            baselines[var_key][str(n)] = trials
            if len(lognorm_trials) > 0:
                baselines[var_key][str(n) + "_lognorm"] = lognorm_trials

        # Compute ALL-examples baseline (using all rows)
        all_trials: List[Dict[str, float]] = []
        all_lognorm_trials: List[Dict[str, float]] = []

        # Save the full subset as "ALL" sample
        save_sample_to_csv(dataset_name, subset, var_key, "ALL", 0)

        n_eff_all = len(subset)
        if is_boolean:
            p_hat_all = float(subset[base_var].mean())
            s_eff_all = p_hat_all * n_eff_all
            alpha_all, beta_all = beta_posterior(ALPHA0, BETA0, s_eff_all, n_eff_all)
            all_trials.append({"alpha": alpha_all, "beta": beta_all})
        else:
            mean_all = float(subset[base_var].mean())
            sd_all = float(subset[base_var].std())
            mu_all, sig_all = gaussian_posterior(
                MU0, SIGMA0 * sd_all, n_eff_all, mean_all, sd_all
            )

            # Lognormal update for ALL
            log_values_all = np.log(subset[base_var] + 1e-6)
            mean_log_all = float(log_values_all.mean())
            sd_log_all = float(log_values_all.std())
            MU0_LOG = 0.0
            SIGMA0_LOG = 10.0 * sd_log_all
            mu_all_log, sig_all_log = gaussian_posterior(
                MU0_LOG, SIGMA0_LOG, n_eff_all, mean_log_all, sd_log_all
            )

            all_trials.append({"mu": mu_all, "sigma": sig_all})
            all_lognorm_trials.append({"mu": mu_all_log, "sigma": sig_all_log})

        baselines[var_key]["ALL"] = all_trials
        if len(all_lognorm_trials) > 0:
            baselines[var_key]["ALL_lognorm"] = all_lognorm_trials

    return baselines


def load_variables_from_json(filepath: str) -> Dict[str, Dict]:
    """
    Load variable specifications from a JSON file.

    Args:
        filepath: Path to JSON file containing variable specifications

    Returns:
        Dictionary of variable specifications
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_baselines_to_json(baselines: Dict, filepath: str):
    """
    Save baselines to a JSON file.

    Args:
        baselines: Dictionary of baseline results
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(baselines, f, indent=2)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Compute baselines for a set of variables')
    parser.add_argument('--variables', type=str, default='variables/pitchbook_variables.json',
                        help='Path to JSON file containing variable specifications')
    parser.add_argument('--data', type=str, default='Company.csv',
                        help='Path to raw company data CSV')
    parser.add_argument('--output', type=str, default='baselines_output.json',
                        help='Path to save output baselines JSON')
    parser.add_argument('--dataset-name', type=str, default='pitchbook',
                        help='Name of the dataset (for saving samples)')

    args = parser.parse_args()

    # Load and preprocess data
    print(f"Loading data from {args.data}...")
    company_raw = pd.read_csv(args.data)
    df = pitchbook_load(company_raw)
    print(f"Loaded {len(df)} companies")

    # Load variable specifications
    print(f"Loading variables from {args.variables}...")
    variables = load_variables_from_json(args.variables)
    print(f"Loaded {len(variables)} variables")

    # Compute baselines
    print("Computing baselines...")
    baselines = compute_baselines_for_variables(df, variables, args.dataset_name)

    # Save results
    print(f"Saving baselines to {args.output}...")
    save_baselines_to_json(baselines, args.output)

    print("Done!")
