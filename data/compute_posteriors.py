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
    results_file_path = os.path.expanduser("{}experiments/{}/{}_combined_processed_results.csv".format(os.environ['OPENESTIMATE_ROOT'], dataset, dataset))
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

    new_rows: List[pd.Series] = []
    for row in llm_df.itertuples(index=False):
        var_key: str = getattr(row, "variable")
        base_var = var_specs.get(var_key, {}).get("base_variable", var_key)
        ground_truth_dist: str = getattr(row, "ground_truth_distribution_type")
        fitted_dist: str = getattr(row, "fitted_distribution_type", ground_truth_dist)
        approach_orig: str = getattr(row, "approach")
        var_sample_dir = baseline_samples_dir / var_key

        # ------------------------------------------------------------------
        # Prior parameters from the LLM row
        # ------------------------------------------------------------------
        if ground_truth_dist == "normal":
            mu0 = float(getattr(row, "mu"))
            sigma0 = float(getattr(row, "sigma"))
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

            # Create new row based on original
            row_dict = row._asdict()
            row_dict["trial"] = trial_idx
            row_dict["approach"] = f"{approach_orig}_posterior_N{N_label}"
            row_dict["sample_size"] = N_label

            # Clear stale computed columns (will be recomputed from new mu/sigma/a/b)
            for col in ['mean', 'median', 'mode', 'std', 'abs_error_from_mean',
                        'abs_error_from_median', 'abs_error_from_mode', 'error_ratio_mean',
                        'error_ratio_median', 'error_ratio_mode', 'std_ratio',
                        'associated_baseline_error_mean', 'associated_baseline_error_median',
                        'associated_baseline_error_mode', 'associated_baseline_std']:
                if col in row_dict:
                    row_dict[col] = np.nan

            # Posterior update based on fitted distribution type
            if ground_truth_dist == "normal":
                if fitted_dist == "lognormal":
                    # Lognormal prior: mu0 and sigma0 are already in log-space
                    # Transform data to log-space
                    y_positive = y[y > 0]
                    w_positive = w[y > 0]
                    if len(y_positive) == 0:
                        continue
                    log_values = np.log(y_positive)
                    n_eff = kish_effn(w_positive)
                    dsw_log = DescrStatsW(log_values, weights=w_positive, ddof=0)
                    mean_hat_log = float(dsw_log.mean)
                    pop_sd_log = float(dsw_log.std)
                    if pop_sd_log == 0 or np.isnan(pop_sd_log):
                        pop_sd_log = 1e-6
                    mu_n, sig_n = complex_gaussian_posterior(mu0, sigma0, n_eff, mean_hat_log, pop_sd_log)
                else:
                    # Normal prior: standard Gaussian update
                    dsw = DescrStatsW(y, weights=w, ddof=0)
                    mean_hat = float(dsw.mean)
                    pop_sd = float(dsw.std)
                    if pop_sd == 0 or np.isnan(pop_sd):
                        pop_sd = 1e-6
                    mu_n, sig_n = complex_gaussian_posterior(mu0, sigma0, n_eff, mean_hat, pop_sd)

                row_dict["mu"] = mu_n
                row_dict["sigma"] = sig_n
            else:
                # Beta posterior
                p_hat = float(np.average(y, weights=w))
                s_eff = p_hat * n_eff
                alpha_n, beta_n = complex_beta_posterior(alpha0, beta0, s_eff, n_eff)
                row_dict["a"] = alpha_n
                row_dict["b"] = beta_n

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

    new_rows: List[pd.Series] = []

    for row in llm_df.itertuples(index=False):
        var_key: str = getattr(row, "variable")
        var_info = var_specs.get(var_key, {})
        base_var = var_info.get("base_variable", var_key)
        ground_truth_dist: str = getattr(row, "ground_truth_distribution_type")
        fitted_dist: str = getattr(row, "fitted_distribution_type", ground_truth_dist)
        approach_orig: str = getattr(row, "approach")

        var_sample_dir = baseline_samples_dir / var_key

        # ------------------------------------------------------------------
        # Prior parameters from the LLM row
        # ------------------------------------------------------------------
        if ground_truth_dist == "normal":
            mu0 = float(getattr(row, "mu"))
            sigma0 = float(getattr(row, "sigma"))
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

            # Create new row based on original
            row_dict = row._asdict()
            row_dict["trial"] = trial_idx
            row_dict["approach"] = f"{approach_orig}_posterior_N{N_label}"
            row_dict["sample_size"] = N_label

            # Clear stale computed columns (will be recomputed from new mu/sigma/a/b)
            for col in ['mean', 'median', 'mode', 'std', 'abs_error_from_mean',
                        'abs_error_from_median', 'abs_error_from_mode', 'error_ratio_mean',
                        'error_ratio_median', 'error_ratio_mode', 'std_ratio',
                        'associated_baseline_error_mean', 'associated_baseline_error_median',
                        'associated_baseline_error_mode', 'associated_baseline_std']:
                if col in row_dict:
                    row_dict[col] = np.nan

            # Posterior update based on fitted distribution type
            if ground_truth_dist == "normal":
                if fitted_dist == "lognormal":
                    # Lognormal prior: mu0 and sigma0 are already in log-space
                    # Transform data to log-space
                    y_positive = y[y > 0]
                    if len(y_positive) == 0:
                        continue
                    log_values = np.log(y_positive)
                    n_eff = int(len(y_positive))
                    mean_hat_log = float(log_values.mean())
                    pop_sd_log = float(log_values.std())
                    if pop_sd_log == 0 or np.isnan(pop_sd_log):
                        pop_sd_log = 1e-6
                    mu_n, sig_n = gaussian_posterior(mu0, sigma0, n_eff, mean_hat_log, pop_sd_log)
                else:
                    # Normal prior: standard Gaussian update
                    mean_hat = float(y.mean())
                    pop_sd = float(y.std())
                    if pop_sd == 0 or np.isnan(pop_sd):
                        pop_sd = 1e-6
                    mu_n, sig_n = gaussian_posterior(mu0, sigma0, n_eff, mean_hat, pop_sd)

                row_dict["mu"] = mu_n
                row_dict["sigma"] = sig_n
            else:
                # Beta posterior
                p_hat = float(y.mean())
                s_eff = p_hat * n_eff
                alpha_n, beta_n = beta_posterior(alpha0, beta0, s_eff, n_eff)
                row_dict["a"] = alpha_n
                row_dict["b"] = beta_n

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
        var_file_path = os.path.expanduser('{}data/variables/{}_variables.json'.format(os.environ['OPENESTIMATE_ROOT'], dataset))
        baseline_samples_dir = os.path.expanduser('{}data/baselines/{}/baseline_data_samples'.format(os.environ['OPENESTIMATE_ROOT'], dataset))
        posterior_file_path = os.path.expanduser('{}experiments/{}/{}_combined_processed_results_with_posteriors.csv'.format(os.environ['OPENESTIMATE_ROOT'], dataset, dataset))
        compute_posteriors = posterior_functions.get(dataset)
        if compute_posteriors is None:
            raise ValueError(f"No posterior computation function defined for dataset: {dataset}")
        else: 
            compute_posteriors(dataset, var_file_path, baseline_samples_dir, posterior_file_path)
    return 


if __name__ == "__main__":
    datasets = parse_args()
    main(datasets)



