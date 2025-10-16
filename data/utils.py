import numpy as np 
import pandas as pd 
from pathlib import Path

SUBSAMPLE_SIZES = [5, 10, 20, 30]   
RESAMPLES_PER_N = 25                     
ALPHA0, BETA0 = 1.0, 1.0
MU0, SIGMA0   = 0.0, 100_000.0     


def check_difference_threshold_proportion(a, b, se, difference_threshold):
    dist = abs(a - b)
    if dist > difference_threshold and dist > se:
        return True
    return False


def check_difference_threshold_continuous(a, b, se, difference_threshold):
    dist = abs(a - b)
    if dist > difference_threshold * a and dist > se:
        return True
    return False


def compute_mean_continuous(data_df, var): 
    col = data_df[var].dropna()  # Remove NaN values for accurate count
    mean_value = col.mean()
    std_value = col.std()
    se_value = std_value / np.sqrt(len(col))
    return mean_value, std_value, se_value


def gaussian_posterior(mu0: float, sigma0: float,
                       n_eff: float, mean_hat: float, pop_sd: float):
    """
    Normal–Normal conjugate update  (variance known = pop_sd²).

    Returns posterior (mu_n, sigma_n).
    """
    sig2 = pop_sd ** 2
    sig02 = sigma0 ** 2

    mu_n = (sig2 * mu0 + n_eff * sig02 * mean_hat) / (sig2 + n_eff * sig02)
    sig_n = np.sqrt((sig2 * sig02) / (sig2 + n_eff * sig02))
    return float(mu_n), float(sig_n)


def beta_posterior(alpha0: float, beta0: float,
                   s_eff: float, n_eff: float):
    """Beta–Bernoulli update:  α′ = α₀ + s,  β′ = β₀ + (n − s)."""
    return float(alpha0 + s_eff), float(beta0 + (n_eff - s_eff))


def save_sample_to_csv(dataset, sample_df, variable, n, trial_idx, base_dir="baseline_data_samples"):
    """
    Drop-in function to save a sample DataFrame to CSV in organized directory structure.
    
    Args:
        sample_df: DataFrame containing the sample
        variable: Variable name (creates subdirectory)
        n: Sample size (int or "ALL")
        trial_idx: Trial index
        base_dir: Base directory for samples
    
    Returns:
        Path to saved file
    """
    sample_dir = Path("baselines") / Path(dataset) / Path(base_dir) /  variable
    sample_dir.mkdir(parents=True, exist_ok=True)
    file_path = sample_dir / f"n{n}_trial{trial_idx}.csv"
    
    try:
        sample_df.reset_index(drop=True).to_csv(file_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save sample {file_path}: {e}")
    
    return file_path


def kish_effn(w: pd.Series | np.ndarray) -> float:
    """Kish effective sample size:  (Σw)² / Σw²."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def complex_gaussian_posterior(
    mu0: float,
    sigma0: float,
    n_eff: float,
    mean_hat: float,
    pop_sd: float,
):
    """
    Normal–Normal conjugate update (variance known = pop_sd²).

    Returns posterior (mu_n, sigma_n).  See Gelman et al. (BDA) §2.5.
    """
    sig2 = pop_sd**2
    sig02 = sigma0**2

    mu_n = (sig2 * mu0 + n_eff * sig02 * mean_hat) / (sig2 + n_eff * sig02)
    sig_n = np.sqrt((sig2 * sig02) / (sig2 + n_eff * sig02))
    return float(mu_n), float(sig_n)


def complex_beta_posterior(alpha0: float, beta0: float, s_eff: float, n_eff: float):
    """Beta–Bernoulli update:  α′ = α₀ + s,  β′ = β₀ + (n − s)."""
    return float(alpha0 + s_eff), float(beta0 + (n_eff - s_eff))

