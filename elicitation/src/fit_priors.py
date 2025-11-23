import re
import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from scipy.optimize import minimize
from utils import get_variable_name, convert_number_to_float


def coerce_float(x, variable_name=None, enable_conversion=True):
    if x is None: 
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    mult = 1.0
    if s.endswith('%'):
        mult *= 0.01
        s = s[:-1]
    s = s.replace('$', '').replace(',', '').replace(' ', '')
    
    try:
        return float(s) * mult
    except ValueError:
        return None


def fit_gaussian_prior(q1, median, q3, plausible_range):
    def objective(params):
        mean, std = params
        q1_fit, median_fit, q3_fit, p1_fit, p99_fit = norm.ppf([0.25, 0.5, 0.75, 0.01, 0.99], loc=mean, scale=std)
        return ((q1 - q1_fit)**2 + (median - median_fit)**2 + (q3 - q3_fit)**2 + 
                (plausible_range[0] - p1_fit)**2 + (plausible_range[1] - p99_fit)**2)

    # Initial guesses
    initial_guess = [median, (q3 - q1) / 1.349]

    # Minimize
    result = minimize(objective, initial_guess, bounds=[(plausible_range[0], plausible_range[1]), (1e-5, None)])
    mean, std = result.x

    # Validate the fit
    print(f"Fitted mean: {mean}, Fitted std: {std}")

    return {'mean': mean, 'std': std, 'type': 'gaussian'}
    
    
def fit_beta_prior(q1, median, q3, plausible_range):
    """
    Fit a beta distribution to quantiles in [0,1] scale.
    General solution that works for any valid set of quantiles.
    """
    lower, upper = plausible_range
    
    # Target quantiles (x, cumulative probability)
    target_quantiles = [
        (lower, 0.01),
        (q1, 0.25),
        (median, 0.50),
        (q3, 0.75),
        (upper, 0.99)
    ]

    def objective(params):
        a, b = params
        if a <= 1 or b <= 1:  # Ensure parameters allow interior mode
            return 1e10
        
        error = 0.0
        # Fit CDF values
        for x, p in target_quantiles:
            try:
                predicted_p = beta.cdf(x, a, b)
                error += (predicted_p - p) ** 2
            except:
                return 1e10
        
        # Add small penalty for extreme parameter values
        if a > 100 or b > 100:
            error += 0.1 * ((max(a - 100, 0))**2 + (max(b - 100, 0))**2)
            
        return error

    # Estimate initial parameters using method of moments
    sample_mean = np.mean([q1, median, q3])  # Rough mean estimate
    sample_var = np.var([q1, median, q3])    # Rough variance estimate
    
    if sample_var == 0:
        # Handle degenerate case
        init_a = init_b = 2.0
    else:
        # Method of moments estimates
        temp = sample_mean * (1 - sample_mean) / sample_var - 1
        init_a = max(1.1, sample_mean * temp)
        init_b = max(1.1, (1 - sample_mean) * temp)

    # Multiple initial guesses including method of moments and some alternatives
    initial_guesses = [
        [init_a, init_b],          # Method of moments
        [2.0, 2.0],               # Symmetric
        [max(1.1, init_a * 0.5), max(1.1, init_b * 0.5)],  # Spread out
        [max(1.1, init_a * 2), max(1.1, init_b * 2)]       # More concentrated
    ]
    
    best_result = None
    best_error = float('inf')
    
    for guess in initial_guesses:
        try:
            # Bounds ensure interior mode but allow wide range of shapes
            bounds = [(1.001, 100.0), (1.001, 100.0)]
            result = minimize(objective, guess, bounds=bounds, method='L-BFGS-B')
            if result.success and result.fun < best_error:
                best_error = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None or not best_result.success:
        raise ValueError(f"Optimization failed for all initial guesses")

    a_opt, b_opt = best_result.x
    
    # Print diagnostics
    print("\nFitted Beta Distribution:")
    print(f"Parameters: a={a_opt:.3f}, b={b_opt:.3f}")
    mean = a_opt/(a_opt + b_opt)
    print(f"Mean: {mean:.3f}")
    mode = (a_opt - 1)/(a_opt + b_opt - 2)
    print(f"Mode: {mode:.3f}")
    
    print("\nQuantile Verification:")
    for x, p in target_quantiles:
        fitted_p = beta.cdf(x, a_opt, b_opt)
        print(f"x={x:.3f}: target_p={p:.3f}, fitted_p={fitted_p:.3f}")

    return {'a': a_opt, 'b': b_opt, 'type': 'beta'}



def assign_prior_direct(elicited_priors, variables):
    """
    Assign priors from direct elicitation.

    Supports two modes:
    1. Traditional mode: Uses ground truth distribution type from variables
    2. Unified mode: LLM chooses distribution type (checks for 'distribution_type' in output)
    """
    for var_name, info in elicited_priors.items():
        var = get_variable_name(var_name, variables)
        output = info[var_name]['var_output']

        # Check if LLM chose the distribution type (unified prompt)
        if 'distribution_type' in output:
            chosen_type = output['distribution_type'].lower().strip()

            if chosen_type == 'normal':
                try:
                    info['fitted_prior'] = {
                        'type': 'gaussian',
                        'mu': coerce_float(output['mu'], variable_name=var_name),
                        'sigma': coerce_float(output['sigma'], variable_name=var_name)
                    }
                except Exception as e:
                    print(f"Error extracting Normal parameters for {var_name}: {e}")
                    info['fitted_prior'] = {
                        'type': 'gaussian',
                        'mu': None,
                        'sigma': None
                    }

            elif chosen_type == 'lognormal':
                try:
                    info['fitted_prior'] = {
                        'type': 'lognormal',
                        'mu': coerce_float(output['mu'], variable_name=var_name),
                        'sigma': coerce_float(output['sigma'], variable_name=var_name)
                    }
                except Exception as e:
                    print(f"Error extracting Lognormal parameters for {var_name}: {e}")
                    info['fitted_prior'] = {
                        'type': 'lognormal',
                        'mu': None,
                        'sigma': None
                    }

            elif chosen_type == 'beta':
                try:
                    info['fitted_prior'] = {
                        'type': 'beta',
                        'a': float(output['alpha']),
                        'b': float(output['beta'])
                    }
                except Exception as e:
                    print(f"Error extracting Beta parameters for {var_name}: {e}")
                    info['fitted_prior'] = {
                        'type': 'beta',
                        'a': None,
                        'b': None
                    }
            else:
                print(f"Warning: Unknown distribution type '{chosen_type}' for {var_name}")
                info['fitted_prior'] = {
                    'type': 'unknown',
                    'error': f"Unknown distribution type: {chosen_type}"
                }

        else:
            # Traditional mode: use ground truth distribution type
            distr = variables[var]['ground_truth_distribution_type']

            if distr == 'normal':
                try:
                    info['fitted_prior'] = {
                        'type': 'gaussian',
                        'mu': coerce_float(output['mu'], variable_name=var_name),
                        'sigma': coerce_float(output['sigma'], variable_name=var_name)
                    }
                except Exception as e:
                    info['fitted_prior'] = {
                        'type': 'gaussian',
                        'mu': None,
                        'sigma': None
                    }
            elif distr == 'beta' or distr == 'binomial':
                try:
                    info['fitted_prior'] = {
                        'type': 'beta',
                        'a': float(output['alpha']),
                        'b': float(output['beta'])
                    }
                except Exception as e:
                    info['fitted_prior'] = {
                        'type': 'beta',
                        'a': None,
                        'b': None
                    }
            else:
                raise ValueError("Unknown distribution type: ", distr)

    return elicited_priors


def process_priors(elicited_priors, variables):
    """
    Process elicited priors into a standardized DataFrame format.

    Handles Normal, Lognormal, and Beta distributions.
    """
    all_processed_results = []

    for var_name, info in elicited_priors.items():
        var = get_variable_name(var_name, variables)
        ground_truth_distr = variables[var]['ground_truth_distribution_type']

        # Get the fitted prior type (may differ from ground truth if using unified prompt)
        fitted_type = info['fitted_prior'].get('type', 'unknown')

        # Build processed result based on fitted distribution type
        processed_result = {
            'variable_name': var_name,
            'variable': var,
            'ground_truth_distribution_type': ground_truth_distr,
            'fitted_distribution_type': fitted_type,
        }

        try:
            if fitted_type == 'gaussian':
                mu = float(info['fitted_prior']['mu'])
                sigma = float(info['fitted_prior']['sigma'])

                processed_result.update({
                    'mu': mu,
                    'sigma': sigma,
                    'a': None,
                    'b': None
                })

            elif fitted_type == 'lognormal':
                mu = float(info['fitted_prior']['mu'])
                sigma = float(info['fitted_prior']['sigma'])

                processed_result.update({
                    'mu': mu,
                    'sigma': sigma,
                    'a': None,
                    'b': None
                })

            elif fitted_type == 'beta':
                a = float(info['fitted_prior']['a'])
                b = float(info['fitted_prior']['b'])
                processed_result.update({
                    'a': a,
                    'b': b,
                    'mu': None,
                    'sigma': None
                })

            else:
                # Unknown or failed extraction
                processed_result.update({
                    'mu': None,
                    'sigma': None,
                    'a': None,
                    'b': None
                })

        except Exception as e:
            print(f"Error processing {fitted_type} prior for {var_name}: {e}")
            processed_result.update({
                'mu': None,
                'sigma': None,
                'a': None,
                'b': None
            })

        info['processed_results'] = processed_result
        all_processed_results.append(processed_result)

    processed_results_df = pd.DataFrame(all_processed_results)
    return processed_results_df


def process_beta_prior(q1, median, q3, plausible_range):
    """
    Convert percentage values to probabilities for beta distribution.
    
    Args:
        q1, median, q3: Quantile values
        plausible_range: Tuple of (lower, upper) bounds
        
    Returns:
        Tuple of converted values: (q1, median, q3, plausible_range)
    """
    plausible_range_lower = plausible_range[0]
    plausible_range_upper = plausible_range[1]
    q1 = q1 / 100.0
    median = median / 100.0
    q3 = q3 / 100.0
    plausible_range_lower = plausible_range[0] / 100.0
    plausible_range_upper = plausible_range[1] / 100.0
    return q1, median, q3, [plausible_range_lower, plausible_range_upper]


def fit_beta_prior_from_mean_variance(mean, variance):
    """
    Fit a Beta distribution from mean and variance.
    
    Parameters:
    mean (float): Desired mean (0 < mean < 1)
    variance (float): Desired variance (0 < variance < mean*(1-mean))
    
    Returns:
    tuple: (alpha, beta) parameters for Beta distribution
    """
    # Check constraints
    if not (0 < mean < 1):
        raise ValueError("Mean must be between 0 and 1")
    
    if variance <= 0:
        raise ValueError("Variance must be positive")
    
    max_var = mean * (1 - mean)
    if variance >= max_var:
        # Use a small epsilon to avoid nu = -1
        variance = max_var * 0.999  # or raise an error instead
        print(f"Warning: Variance {variance} >= max possible {max_var}, clipping to {variance}")

    # Calculate concentration parameter
    nu = (mean * (1 - mean)) / variance - 1
    
    # Ensure nu is positive (additional safety check)
    if nu <= 0:
        raise ValueError(f"Invalid nu={nu}. Variance {variance} too close to maximum {max_var}")
    
    # Calculate shape parameters
    alpha = mean * nu
    beta = (1 - mean) * nu
    
    return alpha, beta


def fit_prior_quantile(elicited_priors, variables):
    """
    Fit priors using quantile method for all variables.
    
    Args:
        elicited_priors: Dictionary of elicited priors
        variables: Dictionary of variable specifications
        
    Returns:
        Updated elicited_priors with fitted_prior added to each variable
    """
    results = elicited_priors.copy()
    for var in variables.items():
        var_name = var[1]['variable']
        print("Processing ", var_name)
        elicited_prior = elicited_priors[var_name][var_name]['var_output']
        print("Elicited prior: ", elicited_prior)
        
        p5 = convert_number_to_float(elicited_prior['q5'])
        p25 = convert_number_to_float(elicited_prior['q25'])
        p50 = convert_number_to_float(elicited_prior['q50'])
        p75 = convert_number_to_float(elicited_prior['q75'])
        p95 = convert_number_to_float(elicited_prior['q95'])
        plausible_range = (p5, p95)
        median = p50
        q1 = p25
        q3 = p75
        
        if var[1]['ground_truth_distribution_type'] == 'normal':
            print("Fitting Gaussian prior for ", var_name)
            distr_params = fit_gaussian_prior(q1, median, q3, plausible_range)
        elif var[1]['ground_truth_distribution_type'] in ['binomial', 'beta']:
            # Check if plausible range contains q1, median, q3
            if not (plausible_range[0] <= q1 <= plausible_range[1] and 
                    plausible_range[0] <= median <= plausible_range[1] and
                    plausible_range[0] <= q3 <= plausible_range[1]):
                raise ValueError(f"Var name: {var_name}, Values must be within plausible range [{plausible_range[0]}, {plausible_range[1]}]. Got q1={q1}, median={median}, q3={q3}")
            
            # Check if all values are already between 0 and 1
            if (0 <= q1 <= 1 and 0 <= median <= 1 and 0 <= q3 <= 1 and 
                0 <= plausible_range[0] <= 1 and 0 <= plausible_range[1] <= 1):
                print("Values already in probability range, skipping processing")
            else:
                print("Converting values to probabilities")
                q1, median, q3, plausible_range = process_beta_prior(q1, median, q3, plausible_range)

            distr_params = fit_beta_prior(q1, median, q3, plausible_range)
        else:
            raise ValueError("Unknown distribution type: ", var[1]['ground_truth_distribution_type'])
        
        results[var_name]['fitted_prior'] = distr_params
        print("--------------------------------")
        print("\n")
    return results


def fit_prior_mean_variance(elicited_priors, variables):
    """
    Fit priors using mean-variance method for all variables.
    
    Args:
        elicited_priors: Dictionary of elicited priors
        variables: Dictionary of variable specifications
        
    Returns:
        Updated elicited_priors with fitted_prior added to each variable
    """
    results = elicited_priors.copy()
    for var in variables.items():
        var_name = var[1]['variable']
        print("Processing ", var_name)
        elicited_prior = elicited_priors[var_name][var_name]['var_output']
        mean = convert_number_to_float(elicited_prior['mean'])
        stdev = convert_number_to_float(elicited_prior['std_dev'])
        
        if var[1]['ground_truth_distribution_type'] == 'normal':
            if mean is None or stdev is None:
                distr_params = {'type': 'gaussian', 'mean': None, 'std': None}
            else:
                distr_params = {
                    'type': 'gaussian',
                    'mean': mean,
                    'std': stdev
                }
        elif var[1]['ground_truth_distribution_type'] == 'beta':
            if mean is None or stdev is None:
                distr_params = {'type': 'beta', 'a': None, 'b': None, 'mean': None, 'std': None}
            else:
                # Handle percentage inputs correctly
                if mean > 1.0:
                    mean = mean / 100.0
                    stdev = stdev / 100.0
                variance = stdev**2

                alpha, beta = fit_beta_prior_from_mean_variance(mean, variance)
                if alpha == 0 or beta == 0:
                    raise ValueError("Alpha or beta is 0, which is not possible for a beta distribution")
                distr_params = {
                    'type': 'beta',
                    'a': alpha,
                    'b': beta, 
                    'mean': mean,
                    'std': stdev
                }
        else:
            raise ValueError("Unknown distribution type: ", var[1]['ground_truth_distribution_type'])
        results[var_name]['fitted_prior'] = distr_params
    return results


def fit_priors(elicited_priors, variables):
    """
    Legacy function for fitting priors using quartile method.
    This appears to be an older version of fit_prior_quantile.
    """
    results = elicited_priors.copy()
    for var in variables.items():
        var_name = var[1]['variable']
        print("Processing ", var_name)
    
        elicited_prior = elicited_priors[var_name][var_name]['var_output']
        plausible_range = (
            convert_number_to_float(elicited_prior['lower_bound']),
            convert_number_to_float(elicited_prior['upper_bound'])
        )
        median = convert_number_to_float(elicited_prior['median'])
        q1 = convert_number_to_float(elicited_prior['lower_quartile'])
        q3 = convert_number_to_float(elicited_prior['upper_quartile'])

        if var[1]['ground_truth_distribution_type'] == 'normal':
            print("Fitting Gaussian prior for ", var_name)
            distr_params = fit_gaussian_prior(q1, median, q3, plausible_range)
        elif var[1]['ground_truth_distribution_type'] == 'beta':
            # Check if plausible range contains q1, median, q3
            if not (plausible_range[0] <= q1 <= plausible_range[1] and 
                    plausible_range[0] <= median <= plausible_range[1] and
                    plausible_range[0] <= q3 <= plausible_range[1]):
                raise ValueError(f"Var name: {var_name}, Values must be within plausible range [{plausible_range[0]}, {plausible_range[1]}]. Got q1={q1}, median={median}, q3={q3}")

            # Check if all values are already between 0 and 1
            if (0 <= q1 <= 1 and 0 <= median <= 1 and 0 <= q3 <= 1 and 
                0 <= plausible_range[0] <= 1 and 0 <= plausible_range[1] <= 1):
                print("Values already in probability range, skipping processing")
            else:
                print("Converting values to probabilities")
                q1, median, q3, plausible_range = process_beta_prior(q1, median, q3, plausible_range)

            distr_params = fit_beta_prior(q1, median, q3, plausible_range)
        else:
            raise ValueError("Unknown distribution type: ", var[1]['ground_truth_distribution_type'])
        results[var_name]['fitted_prior'] = distr_params
    return results
