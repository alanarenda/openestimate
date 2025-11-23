from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
from statsmodels.stats.weightstats import DescrStatsW
from utils import (
    SUBSAMPLE_SIZES, RESAMPLES_PER_N, ALPHA0, BETA0, MU0, SIGMA0,
    gaussian_posterior, beta_posterior, kish_effn
)
from nhanes_generation import (
    target_variables_boolean,
    target_variables_continuous,
    load_and_preprocess_nhanes_data,
    apply_conditions_get_data_subset,
    get_conditions,
    combine_conditions_to_mask
)


def compute_baselines_for_variables_nhanes(
    df: pd.DataFrame,
    variables: Dict[str, Dict],
    dataset_name: str = 'nhanes'
) -> Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]]:
    """
    Compute baselines for a given set of variables (NHANES dataset).

    Args:
        df: The full dataset (preprocessed NHANES data with weights)
        variables: Dictionary of variable specifications with structure:
            {
                'var_key': {
                    'base_variable': str,
                    'conditions': List[Tuple[str, any]],
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

    base_dir = "baseline_data_samples"

    for var_key, spec in variables.items():
        base_var = spec["base_variable"]
        conditions = spec.get("conditions", [])

        # Get the subset of data matching the conditions
        subset = apply_conditions_get_data_subset(df, base_var, conditions)
        if subset is None or subset.empty:
            print(f"Warning: No valid subset for variable '{var_key}'. Skipping.")
            continue

        # Keep only the variable and weights, drop missing
        subset = subset[[base_var, "WTMEC2YR"]].dropna(subset=[base_var])

        # Create directory to save individual resamples for this variable
        sample_var_dir = Path("baselines") / Path(dataset_name) / Path(base_dir) / var_key
        sample_var_dir.mkdir(parents=True, exist_ok=True)

        baselines[var_key] = {}

        is_boolean = base_var in target_variables_boolean

        # Compute baselines for each subsample size
        for n in SUBSAMPLE_SIZES:
            if len(subset) < n:
                continue

            trials: List[Dict[str, float]] = []
            lognorm_trials: List[Dict[str, float]] = []

            for trial_idx in range(RESAMPLES_PER_N):
                # Draw until Kish effective n >= target n (or exhausted)
                sample_size = n
                max_sample_size = len(subset)

                while True:
                    samp = subset.sample(
                        n=sample_size,
                        weights="WTMEC2YR",
                        replace=False,
                        random_state=None,
                    )
                    w = samp["WTMEC2YR"]
                    n_eff = kish_effn(w)

                    # Success condition
                    if n_eff >= n or sample_size == max_sample_size:
                        break

                    # Otherwise, increase actual sample size by one and try again
                    sample_size += 1

                # Save this sample to disk for reproducibility
                sample_path = sample_var_dir / f"n{n}_trial{trial_idx}.csv"
                samp.reset_index(drop=False).to_csv(sample_path, index=False)

                # Posterior update
                if is_boolean:
                    # Weighted success proportion
                    p_hat = float(np.average(samp[base_var], weights=w))
                    s_eff = p_hat * n_eff
                    alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                    trials.append({"alpha": alpha, "beta": beta})

                else:
                    # Continuous variable: Normal update
                    dsw = DescrStatsW(samp[base_var], weights=w, ddof=0)
                    mean_hat = float(dsw.mean)
                    pop_sd = float(dsw.std)

                    # Use small epsilon for zero or NaN variance
                    if pop_sd == 0 or np.isnan(pop_sd):
                        pop_sd = 1e-6

                    mu_n, sig_n = gaussian_posterior(
                        MU0, SIGMA0 * pop_sd, n_eff, mean_hat, pop_sd
                    )
                    trials.append({"mu": mu_n, "sigma": sig_n})

                    # Lognormal update
                    log_values = np.log(samp[base_var] + 1e-6)
                    dsw_log = DescrStatsW(log_values, weights=w, ddof=0)
                    mean_hat_log = float(dsw_log.mean)
                    pop_sd_log = float(dsw_log.std)

                    # Use small epsilon for zero or NaN log variance
                    if pop_sd_log == 0 or np.isnan(pop_sd_log):
                        pop_sd_log = 1e-6

                    # Use log-space priors
                    MU0_LOG = 0.0
                    SIGMA0_LOG = 10.0 * pop_sd_log
                    mu_n_log, sig_n_log = gaussian_posterior(
                        MU0_LOG, SIGMA0_LOG, n_eff, mean_hat_log, pop_sd_log
                    )
                    lognorm_trials.append({"mu": mu_n_log, "sigma": sig_n_log})

            baselines[var_key][str(n)] = trials
            if len(lognorm_trials) > 0:
                baselines[var_key][str(n) + "_lognorm"] = lognorm_trials

        # Compute ALL-examples baseline (using all rows)
        all_trials: List[Dict[str, float]] = []
        all_lognorm_trials: List[Dict[str, float]] = []

        # Save the full subset as "ALL" sample
        sample_path = sample_var_dir / f"nALL_trial0.csv"
        subset.reset_index(drop=False).to_csv(sample_path, index=False)

        # Compute posterior using ALL available data
        w = subset["WTMEC2YR"]
        n_eff_all = kish_effn(w)

        if is_boolean:
            # Weighted success proportion
            p_hat_all = float(np.average(subset[base_var], weights=w))
            s_eff_all = p_hat_all * n_eff_all
            alpha_all, beta_all = beta_posterior(ALPHA0, BETA0, s_eff_all, n_eff_all)
            all_trials.append({"alpha": alpha_all, "beta": beta_all})

        else:
            # Continuous variable
            dsw_all = DescrStatsW(subset[base_var], weights=w, ddof=0)
            mean_all = float(dsw_all.mean)
            sd_all = float(dsw_all.std)

            # Use small epsilon for zero or NaN variance
            if sd_all == 0 or np.isnan(sd_all):
                sd_all = 1e-6

            mu_all, sig_all = gaussian_posterior(
                MU0, SIGMA0 * sd_all, n_eff_all, mean_all, sd_all
            )
            all_trials.append({"mu": mu_all, "sigma": sig_all})

            # Lognormal update for ALL
            log_values_all = np.log(subset[base_var] + 1e-6)
            dsw_log_all = DescrStatsW(log_values_all, weights=w, ddof=0)
            mean_log_all = float(dsw_log_all.mean)
            sd_log_all = float(dsw_log_all.std)

            # Use small epsilon for zero or NaN log variance
            if sd_log_all == 0 or np.isnan(sd_log_all):
                sd_log_all = 1e-6

            MU0_LOG = 0.0
            SIGMA0_LOG = 10.0 * sd_log_all
            mu_all_log, sig_all_log = gaussian_posterior(
                MU0_LOG, SIGMA0_LOG, n_eff_all, mean_log_all, sd_log_all
            )
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

    parser = argparse.ArgumentParser(description='Compute baselines for NHANES variables')
    parser.add_argument('--variables', type=str, default='variables/nhanes_variables.json',
                        help='Path to JSON file containing variable specifications')
    parser.add_argument('--output', type=str, default='nhanes_baselines_output.json',
                        help='Path to save output baselines JSON')
    parser.add_argument('--dataset-name', type=str, default='nhanes',
                        help='Name of the dataset (for saving samples)')

    args = parser.parse_args()

    # Load NHANES data (preprocessed with weights)
    print("Loading NHANES data...")
    df = load_and_preprocess_nhanes_data()
    print(f"Loaded {len(df)} individuals")

    # Load variable specifications
    print(f"Loading variables from {args.variables}...")
    variables = load_variables_from_json(args.variables)
    print(f"Loaded {len(variables)} variables")

    # Compute baselines
    print("Computing baselines...")
    baselines = compute_baselines_for_variables_nhanes(df, variables, args.dataset_name)

    # Save results
    print(f"Saving baselines to {args.output}...")
    save_baselines_to_json(baselines, args.output)

    print("Done!")
