import os 
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from utils import (gaussian_posterior, beta_posterior, complex_gaussian_posterior, 
                   complex_beta_posterior, kish_effn) 
from statsmodels.stats.weightstats import DescrStatsW


_SAMPLE_RE = re.compile(r"n(?P<n>ALL|\d+)_trial(?P<trial>\d+)\.csv")


def load_priors_and_variables(dataset, variables_path):     
    results_file_path = os.path.expanduser("~/openestimate/experiments/{dataset}/{dataset}_combined_processed_results.csv".format(dataset=dataset))
    results = pd.read_csv(results_file_path)
    llm_df = load_llm_rows(results)

    # Normalize temperature to string for robust comparisons (CSV may mix types)
    temp_str = llm_df["temperature"].astype(str)

    # Base include mask: only LLM rows with desired prompt/protocol
    base_mask = (
        ~llm_df["approach"].fillna("").str.contains("statistical_baseline", na=False) &
        ~llm_df["approach"].fillna("").str.contains("base_guess",           na=False) &
        (llm_df["sysprompt_type"] == "base") &
        (llm_df["elicitation_protocol"] == "direct")
    )

    # Model-family specific masks; temperature checks done on normalized strings
    keep_mask = (
        ((temp_str == "0.5")    & llm_df["approach"].str.contains("gpt-4o",   na=False)) |
        ((temp_str == "medium") & llm_df["approach"].str.contains("o4-mini",  na=False)) |
        ((temp_str == "medium") & llm_df["approach"].str.contains("o3-mini",  na=False)) |
        ((temp_str == "0.5")    & llm_df["approach"].str.contains("meta",     na=False)) |
        ((temp_str == "0.6")    & llm_df["approach"].str.contains("qwen",     na=False))
    )

    model_mask = base_mask & keep_mask
   
    llm_df = llm_df[model_mask].copy()
    var_specs = load_variable_specs(variables_path)
    return results, llm_df, var_specs


def compute_llm_posteriors_complex(dataset, variables_path, baseline_samples_dir, posterior_file_path):
    baseline_samples_dir = Path(baseline_samples_dir)
    results, llm_df, var_specs = load_priors_and_variables(dataset, variables_path)

    # Pre-compute baseline MAE for each variable for error_ratio calculation
    baseline_mae_by_var = {}
    for var in results['variable'].unique():
        baseline_data = results[
            (results['variable'] == var) &
            (results['approach'].str.contains('statistical_baseline_n5', na=False))
        ]
        if len(baseline_data) > 0:
            baseline_mae_by_var[var] = baseline_data['abs_error'].mean()

    new_rows: List[pd.Series] = []
    for row in llm_df.itertuples(index=False):
        var_key: str = getattr(row, "variable")
        base_var = var_specs.get(var_key, {}).get("base_variable", var_key)
        dist_type: str = getattr(row, "ground_truth_distribution_type")
        approach_orig: str = getattr(row, "approach")
        var_sample_dir = baseline_samples_dir / var_key
      
        # ------------------------------------------------------------------
        # Prior parameters from the LLM row
        # ------------------------------------------------------------------
        if dist_type == "normal":
            mu0 = float(getattr(row, "mean"))
            sigma0 = float(getattr(row, "std"))
            if np.isnan(sigma0) or sigma0 <= 0:
                sigma0 = 100_000.0
        else:  # beta
            alpha0 = float(getattr(row, "a"))
            beta0 = float(getattr(row, "b"))
            if (alpha0 <= 0) or (beta0 <= 0):
                alpha0, beta0 = 1.0, 1.0

        # Iterate over stored samples
        for sample_fp in sorted(glob.glob(str(var_sample_dir / "n*_trial*.csv"))):
            m = _SAMPLE_RE.search(Path(sample_fp).name)
            if m is None:
                continue
            N_label = m.group("n")
            trial_idx = int(m.group("trial"))

            samp = pd.read_csv(sample_fp)
            if base_var not in samp.columns:
                continue
            w = samp["WTMEC2YR"]
            y = samp[base_var]
            n_eff = kish_effn(w)

            # Posterior update
            if dist_type == "normal":
                dsw = DescrStatsW(y, weights=w, ddof=0)
                mean_hat = float(dsw.mean)
                pop_sd = float(dsw.std)
                mu_n, sig_n = complex_gaussian_posterior(mu0, sigma0, n_eff, mean_hat, pop_sd)
            else:
                p_hat = float(np.average(y, weights=w))
                s_eff = p_hat * n_eff
                alpha_n, beta_n = complex_beta_posterior(alpha0, beta0, s_eff, n_eff)

            # Create new row based on original
            row_dict = row._asdict()
            row_dict["trial"] = trial_idx
            row_dict["approach"] = f"{approach_orig}_posterior_N{N_label}"
            # Update distribution parameters
            if dist_type == "normal":
                row_dict["mean"] = mu_n
                row_dict["std"] = sig_n
            else:
                row_dict["a"] = alpha_n
                row_dict["b"] = beta_n

            # Recompute abs_error based on updated distribution parameters
            gt_value = float(getattr(row, "ground_truth"))
            if dist_type == "normal":
                row_dict["abs_error"] = abs(mu_n - gt_value)
            else:
                # For beta distribution, compute mean from updated alpha/beta
                mean_beta = alpha_n / (alpha_n + beta_n)
                row_dict["abs_error"] = abs(mean_beta - gt_value)

            # Compute error_ratio relative to n=5 baseline
            row_dict["error_ratio"] = row_dict["abs_error"] / baseline_mae_by_var[var_key]

            new_rows.append(pd.Series(row_dict))

    if new_rows:
        print("Adding new rows")
        df_aug = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new rows to add")
        df_aug = results.copy()

    with open(posterior_file_path, 'w') as fh:
        df_aug.to_csv(fh, index=False)

    return df_aug 


def load_variable_specs(variable_specs_path):
    with open(variable_specs_path, 'r') as fh:
        return json.load(fh)


def load_llm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that correspond to LLM priors (exclude statistical baselines)."""
    # mask_model = ~df["model"].str.contains("statistical_baseline")
    mask_approach = ~df["approach"].str.contains("statistical_baseline")
    return df[mask_approach].copy()


def compute_llm_posteriors_regular(dataset, variables_path, baseline_samples_dir, posterior_file_path):
    baseline_samples_dir = Path(baseline_samples_dir)
    results, llm_df, var_specs = load_priors_and_variables(dataset, variables_path)

    # Pre-compute baseline MAE for each variable for error_ratio calculation
    baseline_mae_by_var = {}
    for var in results['variable'].unique():
        baseline_data = results[
            (results['variable'] == var) &
            (results['approach'].str.contains('statistical_baseline_n5', na=False))
        ]
        if len(baseline_data) > 0:
            baseline_mae_by_var[var] = baseline_data['abs_error'].mean()

    new_rows: List[pd.Series] = []

    for row in llm_df.itertuples(index=False):
        var_key: str = getattr(row, "variable")
        var_info = var_specs.get(var_key, {})
        base_var = var_info.get("base_variable", var_key)
        dist_type: str = getattr(row, "ground_truth_distribution_type")
        approach_orig: str = getattr(row, "approach")

        var_sample_dir = baseline_samples_dir / var_key

        # ------------------------------------------------------------------
        # Prior parameters from the LLM row
        # ------------------------------------------------------------------
        if dist_type == "normal":
            mu0 = float(getattr(row, "mean"))
            sigma0 = float(getattr(row, "std"))
            if np.isnan(sigma0) or sigma0 <= 0:
                sigma0 = 100_000.0
        else:  # beta
            alpha0 = float(getattr(row, "a"))
            beta0 = float(getattr(row, "b"))
            if (alpha0 <= 0) or (beta0 <= 0):
                alpha0, beta0 = 1.0, 1.0

        # Iterate over stored samples
        for sample_fp in sorted(glob.glob(str(var_sample_dir / "n*_trial*.csv"))):
            m = _SAMPLE_RE.search(Path(sample_fp).name)
            if m is None:
                continue
            N_label = m.group("n")
            trial_idx = int(m.group("trial"))

            samp = pd.read_csv(sample_fp)
            if base_var not in samp.columns:
                continue
            # Unweighted updates: use simple sample statistics
            y = samp[base_var].dropna()
            n_eff = int(len(y))
            if n_eff <= 0:
                continue

            # Posterior update
            if dist_type == "normal":
                mean_hat = float(y.mean())
                sd_hat = float(y.std())
                mu_n, sig_n = gaussian_posterior(mu0, sigma0, n_eff, mean_hat, sd_hat)
            else:
                p_hat = float(y.mean())
                s_eff = p_hat * n_eff
                alpha_n, beta_n = beta_posterior(alpha0, beta0, s_eff, n_eff)

            # Create new row based on original
            row_dict = row._asdict()
            row_dict["trial"] = trial_idx
            row_dict["approach"] = f"{approach_orig}_posterior_N{N_label}"
            # Update distribution parameters
            if dist_type == "normal":
                row_dict["mean"] = mu_n
                row_dict["std"] = sig_n
            else:
                row_dict["a"] = alpha_n
                row_dict["b"] = beta_n

            # Recompute abs_error based on updated distribution parameters
            gt_value = float(getattr(row, "ground_truth"))
            if dist_type == "normal":
                row_dict["abs_error"] = abs(mu_n - gt_value)
            else:
                # For beta distribution, compute mean from updated alpha/beta
                mean_beta = alpha_n / (alpha_n + beta_n)
                row_dict["abs_error"] = abs(mean_beta - gt_value)

            # Compute error_ratio relative to n=5 baseline
            row_dict["error_ratio"] = row_dict["abs_error"] / baseline_mae_by_var[var_key]

            new_rows.append(pd.Series(row_dict))

    if new_rows:
        print("Adding new rows")
        df_aug = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        print("No new rows to add")
        df_aug = results.copy()

    with open(posterior_file_path, 'w') as fh:
        df_aug.to_csv(fh, index=False)

    return df_aug 


def parse_args():
    parser = argparse.ArgumentParser(description='Compute posteriors for datasets with LLM priors and baseline samples.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Datasets to process (e.g., glassdoor, pitchbook, nhanes)')
    datasets = parser.parse_args().datasets[0].split(',')
    datasets = [ds.strip() for ds in datasets]
    return datasets


posterior_functions = {'glassdoor': compute_llm_posteriors_regular, 'nhanes': compute_llm_posteriors_complex, 'pitchbook': compute_llm_posteriors_regular}


def main(datasets): 
    # Take a set of dataset names as input and compute posteriors for each. We assume we have elicited priors for all of the datasets passed in. 
    # Store functions to compute posteriors for each dataset in a dictionary since the methodology may differ between them. 
    for dataset in datasets:
        var_file_path = os.path.expanduser('~/openestimate/data/variables/{dataset}_variables.json'.format(dataset=dataset))
        baseline_samples_dir = os.path.expanduser('~/openestimate/data/baselines/{dataset}/baseline_data_samples'.format(dataset=dataset))
        posterior_file_path = os.path.expanduser('~/openestimate/experiments/{dataset}/results_with_posteriors.csv'.format(dataset=dataset))
        compute_posteriors = posterior_functions.get(dataset)
        if compute_posteriors is None:
            raise ValueError(f"No posterior computation function defined for dataset: {dataset}")
        else: 
            compute_posteriors(dataset, var_file_path, baseline_samples_dir, posterior_file_path)
    return 


if __name__ == "__main__":
    datasets = parse_args()
    main(datasets)



