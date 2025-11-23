from __future__ import annotations
import os
import random 
import numpy as np
import pandas as pd 
from scipy import stats
from pathlib import Path
from typing import Dict, List, Union
from statsmodels.stats.weightstats import DescrStatsW
from nhanes.load import load_NHANES_data, load_NHANES_metadata
from utils import complex_beta_posterior, complex_gaussian_posterior, kish_effn, check_difference_threshold_continuous, check_difference_threshold_proportion
random.seed(42)

target_variables_boolean = [
    "DoctorToldYouHaveDiabetes",
    "TakeMedicationForDepression",
    "CoveredByHealthInsurance",
    "EverToldYouHadCancerOrMalignancy",
    "EverToldYouHadHeartAttack",
    'HaveSeriousDifficultyConcentrating',
    'TakingInsulinNow'
]

target_variables_continuous = [
    "WaistCircumferenceCm",
    "TotalCholesterolMgdl",
    "BloodLeadUgdl",
    "BloodCadmiumUgl",
    "BloodMercuryTotalUgl",
    "BloodManganeseUgl",
    "WeightKg"
]

base_phrases = { 
    "TakeMedicationForDepression": "The probability that the average adult (over 18) in the US population takes medication for depression",
    "CoveredByHealthInsurance": "The probability that the average adult (over 18) in the US population or their spouse has health insurance",
    "EverToldYouHadCancerOrMalignancy": "The probability that a member of the US population who is over 18 has ever been diagnosed with cancer by a doctor",
    "EverToldYouHadHeartAttack": "The probability that a member of the US population who is over 18 has ever had a heart attack that was confirmed by a doctor",
    "WaistCircumferenceCm": "The average waist circumference among adult (over 18) individuals in cm in the general US population",
    "TotalCholesterolMgdl": "The average total cholesterol among adult (over 18) individuals in units of mg/dL in the general US population",
    "BloodLeadUgdl": "The average blood lead level among adult (over 18) individuals in units of ug/dL in the general US population",
    "DoctorToldYouHaveDiabetes": "The probability that the average adult (over 18) in the US population has ever been diagnosed with diabetes by a doctor",
    "HaveSeriousDifficultyConcentrating": "The probability that the average adult (over 18) in the US population has ever had serious difficulty concentrating",
    "TakingInsulinNow": "The probability that the average adult (over 18) in the US population is currently taking insulin",
    "BloodCadmiumUgl": "The average blood cadmium level among adult (over 18) individuals in units of ug/L in the general US population",
    "BloodMercuryTotalUgl": "The average blood mercury level among adult (over 18) individuals in units of ug/L in the general US population",
    "BloodManganeseUgl": "The average blood manganese level among adult (over 18) individuals in units of ug/L in the general US population",
    "WeightKg": "The average weight among adult (over 18) individuals in the general US population in kilograms",
}

cond_phrases = {
    "TakeMedicationForDepression": {True: "they take medication for depression", False: "that they do not take medication for depression"},
    "CoveredByHealthInsurance": {True: "they have health insurance", False: "they do not have health insurance"},
    "EverToldYouHadCancerOrMalignancy": {True: "they have ever been diagnosed with cancer by a doctor", False: "they have not been diagnosed with cancer by a doctor"},
    "EverToldYouHadHeartAttack": {True: "they have ever had a heart attack that was confirmed by a doctor", False: "they have not had a heart attack that was confirmed by a doctor"},
    "HaveSeriousDifficultyConcentrating": {True: "they have serious difficulty concentrating", False: "they do not have serious difficulty concentrating"},
    "TakingInsulinNow": {True: "they are currently taking insulin", False: "they are not currently taking insulin"},
    "WaistCircumferenceCm": "their waist circumference in centimeters is",
    "TotalCholesterolMgdl": "their total cholesterol in mg/dL is",
    "BloodLeadUgdl": "their blood lead level in ug/dL is",
    "BloodCadmiumUgl": "their blood cadmium level in ug/L is",
    "BloodMercuryTotalUgl": "their blood mercury level in ug/L is",
    "BloodManganeseUgl": "their blood manganese level in ug/L is",
    "WeightKg": "their weight in kilograms is",
    "DoctorToldYouHaveDiabetes": {True: "they have ever been diagnosed with diabetes by a doctor", False: "they have not been diagnosed with diabetes by a doctor"},
}


def load_and_preprocess_nhanes_data(year='2017-2018'):
    year="2017-2018"

    data_df = load_NHANES_data(year=year)
    meta_df = load_NHANES_metadata(year=year)

    # attach weights & design variables
    weights_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/nhanes_sample_weights.csv"))
    weights_df = weights_df[["SEQN", "WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU"]]
    data_df = data_df.merge(weights_df, on="SEQN", how="left")

    # simple recodes
    data_df["DoctorToldYouHaveDiabetes"] = data_df["DoctorToldYouHaveDiabetes"].apply(
        lambda x: 1 if (x == "1") or (x == "Borderline") else 0
    )
    data_df["DoYouNowSmokeCigarettes"] = data_df["DoYouNowSmokeCigarettes"].apply(
        lambda x: 1 if (x == "Every day" or x == "Some days") else 0
    )

    age = 'AgeInYearsAtScreening'
    data_df = data_df[data_df[age] >= 18]

    data_df = data_df[
        ["SEQN", "WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU"]
        + target_variables_boolean
        + target_variables_continuous
    ]
    return data_df


def combine_conditions_to_mask(data_df, conditions):
    """Combine multiple conditions into a single mask while preserving index alignment"""
    if not conditions:
        return pd.Series([True]*len(data_df), index=data_df.index)
    
    # Start with all True and progressively apply conditions
    mask = pd.Series(True, index=data_df.index)
    for cond in conditions:
        if isinstance(cond, pd.Series):
            mask &= cond
        else:  # Handle array-like conditions
            mask &= pd.Series(cond, index=data_df.index)
    return mask


def nhanes_domain_mean_and_variance(df, var_name, domain_var=None, 
                        weight_name='WTMEC2YR',
                        strata_name='SDMVSTRA', 
                        psu_name='SDMVPSU'):
    """
    Calculate weighted mean AND standard deviation for a domain/subpopulation 
    in NHANES following CDC guidelines.
    """
    # Keep ALL observations - critical for variance estimation
    work_df = df.copy()
    
    # Create domain indicator
    if domain_var is None:
        work_df['_domain'] = 1
    else:
        work_df['_domain'] = domain_var.astype(int)
    
    # Remove ONLY rows with missing design variables
    design_cols = [weight_name, strata_name, psu_name]
    work_df = work_df.dropna(subset=design_cols)
    
    # Handle missing values in analysis variable
    work_df['_nonmissing'] = (~work_df[var_name].isna()).astype(int)
    work_df['_y'] = work_df[var_name].fillna(0)
    
    # Domain AND non-missing indicator
    work_df['_in_analysis'] = work_df['_domain'] * work_df['_nonmissing']
    
    # Calculate weighted totals
    weighted_sum = (work_df['_y'] * work_df['_in_analysis'] * work_df[weight_name]).sum()
    weight_sum = (work_df['_in_analysis'] * work_df[weight_name]).sum()

    if var_name in target_variables_continuous:
        # Log-transform the values (add small constant to avoid log(0))
        work_df['_log_y'] = np.log(work_df['_y'] + 1e-6)

        # Calculate weighted mean on log scale
        weighted_log_sum = (work_df['_log_y'] * work_df['_in_analysis'] * work_df[weight_name]).sum()
        log_mean = weighted_log_sum / weight_sum if weight_sum > 0 else np.nan

        # Calculate weighted variance on log scale
        work_df['_log_squared_dev'] = (work_df['_log_y'] - log_mean) ** 2
        weighted_log_sq_dev = (work_df['_log_squared_dev'] *
                            work_df['_in_analysis'] *
                            work_df[weight_name]).sum()
        log_variance = weighted_log_sq_dev / weight_sum if weight_sum > 0 else np.nan
        log_std = np.sqrt(log_variance)

        # Convert log-scale parameters to original scale using lognormal formulas
        mean_lognormal = np.exp(log_mean + 0.5 * log_std ** 2)
        std_lognormal = np.sqrt((np.exp(log_std ** 2) - 1) * np.exp(2 * log_mean + log_std ** 2))
    else: 
        mean_lognormal = np.nan
        std_lognormal = np.nan
    
    if weight_sum == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'se_mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_domain': 0,
            'weighted_n': 0,
            'df': 0, 
            'mean_lognormal': np.nan, 
            'std_lognormal':np.nanargmax
        }
    
    # Domain mean
    domain_mean = weighted_sum / weight_sum
    
    # ========================================
    # WEIGHTED STANDARD DEVIATION CALCULATION
    # ========================================
    
    # Calculate weighted variance (for the population, not the mean)
    # This is the second moment about the mean
    work_df['_squared_dev'] = (work_df['_y'] - domain_mean) ** 2
    
    # Weighted sum of squared deviations
    weighted_sq_dev = (work_df['_squared_dev'] * 
                      work_df['_in_analysis'] * 
                      work_df[weight_name]).sum()
    
    # Weighted variance 
    # Note: Using N weighting (not N-1) as weights represent population
    weighted_variance = weighted_sq_dev / weight_sum
    
    # Weighted standard deviation
    weighted_std = np.sqrt(weighted_variance)
    
    # ========================================
    # STANDARD ERROR OF THE MEAN (Taylor linearization)
    # ========================================
    
    # Create linearized variable for ratio (for SE of mean)
    work_df['_lin'] = work_df['_in_analysis'] * (work_df['_y'] - domain_mean)
    
    # Calculate variance of the mean using PSUs and strata
    variance_sum = 0
    n_strata_in_domain = 0
    n_psu_in_domain = 0
    
    for stratum, stratum_data in work_df.groupby(strata_name):
        stratum_domain_n = stratum_data['_in_analysis'].sum()
        
        if stratum_domain_n > 0:
            n_strata_in_domain += 1
            psu_totals = []
            
            for psu, psu_data in stratum_data.groupby(psu_name):
                if psu_data['_in_analysis'].sum() > 0:
                    n_psu_in_domain += 1
                    psu_total = (psu_data['_lin'] * psu_data[weight_name]).sum()
                    psu_totals.append(psu_total)
            
            n_psu_stratum = len(psu_totals)
            
            if n_psu_stratum > 1:
                psu_totals = np.array(psu_totals)
                psu_mean = psu_totals.mean()
                stratum_variance = (n_psu_stratum / (n_psu_stratum - 1)) * \
                                   np.sum((psu_totals - psu_mean) ** 2)
                variance_sum += stratum_variance
    
    # Standard error of the mean
    se_mean = np.sqrt(variance_sum) / weight_sum if weight_sum > 0 else np.nan
    
    # Degrees of freedom
    df = n_psu_in_domain - n_strata_in_domain
    
    # Confidence intervals for the mean
    if df > 0:
        t_critical = stats.t.ppf(0.975, df)
    else:
        t_critical = 1.96
    
    ci_lower = domain_mean - t_critical * se_mean
    ci_upper = domain_mean + t_critical * se_mean
    
    # Additional statistics
    n_domain = work_df['_in_analysis'].sum()

    # Calculate percentiles (weighted)
    domain_data = work_df[work_df['_in_analysis'] == 1].copy()
    if len(domain_data) > 0:
        # Sort by value
        domain_data = domain_data.sort_values(var_name)
        domain_data['_cumweight'] = domain_data[weight_name].cumsum()
        total_weight = domain_data[weight_name].sum()

        # Find weighted percentiles
        def weighted_percentile(percent):
            threshold = total_weight * percent / 100
            idx = (domain_data['_cumweight'] >= threshold).idxmax()
            return domain_data.loc[idx, var_name]

        p25 = weighted_percentile(25)
        p50 = weighted_percentile(50)  # median
        p75 = weighted_percentile(75)
    else:
        p25 = p50 = p75 = np.nan

    # ========================================
    # LOGNORMAL PARAMETERS
    # ========================================

    # Log-transform the values (add small constant to avoid log(0))
    work_df['_log_y'] = np.log(work_df['_y'] + 1e-6)

    # Calculate weighted mean on log scale
    weighted_log_sum = (work_df['_log_y'] * work_df['_in_analysis'] * work_df[weight_name]).sum()
    log_mean = weighted_log_sum / weight_sum if weight_sum > 0 else np.nan

    # Calculate weighted variance on log scale
    work_df['_log_squared_dev'] = (work_df['_log_y'] - log_mean) ** 2
    weighted_log_sq_dev = (work_df['_log_squared_dev'] *
                          work_df['_in_analysis'] *
                          work_df[weight_name]).sum()
    log_variance = weighted_log_sq_dev / weight_sum if weight_sum > 0 else np.nan
    log_std = np.sqrt(log_variance)

    # Standard error of log mean using Taylor linearization
    work_df['_log_lin'] = work_df['_in_analysis'] * (work_df['_log_y'] - log_mean)

    log_variance_sum = 0
    for stratum, stratum_data in work_df.groupby(strata_name):
        stratum_domain_n = stratum_data['_in_analysis'].sum()

        if stratum_domain_n > 0:
            psu_totals = []

            for psu, psu_data in stratum_data.groupby(psu_name):
                if psu_data['_in_analysis'].sum() > 0:
                    psu_total = (psu_data['_log_lin'] * psu_data[weight_name]).sum()
                    psu_totals.append(psu_total)

            n_psu_stratum = len(psu_totals)

            if n_psu_stratum > 1:
                psu_totals = np.array(psu_totals)
                psu_mean = psu_totals.mean()
                stratum_variance = (n_psu_stratum / (n_psu_stratum - 1)) * \
                                   np.sum((psu_totals - psu_mean) ** 2)
                log_variance_sum += stratum_variance

    se_log_mean = np.sqrt(log_variance_sum) / weight_sum if weight_sum > 0 else np.nan

    # Convert log-scale parameters to original scale using lognormal formulas
    mean_lognormal = np.exp(log_mean + 0.5 * log_std ** 2)
    std_lognormal = np.sqrt((np.exp(log_std ** 2) - 1) * np.exp(2 * log_mean + log_std ** 2))

    return {
        'mean': domain_mean,
        'stdev': weighted_std,  # Population standard deviation
        'se': se_mean,   # Standard error of the mean
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'median': p50,
        'q1': p25,
        'q3': p75,
        'iqr': p75 - p25,
        'cv': (weighted_std / domain_mean * 100) if domain_mean != 0 else np.nan,  # Coefficient of variation
        'mean_lognormal': mean_lognormal,
        'std_lognormal': std_lognormal,
        'se_lognormal': se_log_mean,
        'n_domain': n_domain,
        'weighted_n': weight_sum,
        'df': df,
        'n_strata_in_domain': n_strata_in_domain,
        'n_psu_in_domain': n_psu_in_domain
    }


def get_variable_name(var_name):
    metadata_df = load_NHANES_metadata(year="2017-2018")
    return metadata_df.loc[var_name, "SASLabel"]


def get_quartile_df(data_df, cond_var, ind):
    """Returns (mask, description) tuple for a quartile condition"""
    # Calculate dynamic quartiles based on current data_df
    filtered_values = data_df[cond_var].dropna()
    quartiles = np.percentile(filtered_values, [1, 25, 50, 75, 99])
    
    # Create mask with original DataFrame's index
    mask = pd.Series(False, index=data_df.index)
    if ind == 0:
        quartile_mask = (filtered_values <= quartiles[1])
    elif ind == 1:
        quartile_mask = (filtered_values > quartiles[1]) & (filtered_values <= quartiles[2])
    elif ind == 2:
        quartile_mask = (filtered_values > quartiles[2]) & (filtered_values <= quartiles[3])
    elif ind == 3:
        quartile_mask = (filtered_values > quartiles[3])
    
    mask.loc[quartile_mask.index] = quartile_mask
    
    # Create natural language description
    var_name = get_variable_name(cond_var)
    if ind == 0:
        desc = f"{var_name} ≤ {quartiles[1]:.1f}"
    elif ind == 1:
        desc = f"{quartiles[1]:.1f} < {var_name} ≤ {quartiles[2]:.1f}"
    elif ind == 2:
        desc = f"{quartiles[2]:.1f} < {var_name} ≤ {quartiles[3]:.1f}"
    elif ind == 3:
        desc = f"{var_name} > {quartiles[3]:.1f}"
    
    return mask, desc  # Now returns tuple of (mask, description)


def get_conditions(data_df, conditions):
    """Progressive filtering to determine bounds, but create masks on full dataset"""
    conds = []
    descriptions = []
    working_df = data_df.copy()  # For determining bounds only
    
    # Store the bounds for each condition
    condition_bounds = []
    
    # First pass: determine all bounds using progressive filtering
    for cond_var, ind in conditions:
        if cond_var in target_variables_boolean:
            condition_bounds.append(('boolean', cond_var, ind))
            # Apply this condition for next quartile calculation
            mask = working_df[cond_var] == ind
            working_df = working_df[mask].copy()
        else:
            # Calculate quartiles on current filtered dataset
            filtered_values = working_df[cond_var].dropna()
            
            # Check if we have enough data for quartiles
            if len(filtered_values) < 4:
                # Not enough data for meaningful quartiles - reject this variable combination
                return [], []
            
            quartiles = np.percentile(filtered_values, [1, 25, 50, 75, 99])
            
            # Store the actual bounds
            if ind == 0:
                condition_bounds.append(('continuous', cond_var, 'le', quartiles[1]))
            elif ind == 1:
                condition_bounds.append(('continuous', cond_var, 'between', quartiles[1], quartiles[2]))
            elif ind == 2:
                condition_bounds.append(('continuous', cond_var, 'between', quartiles[2], quartiles[3]))
            elif ind == 3:
                condition_bounds.append(('continuous', cond_var, 'gt', quartiles[3]))
            
            # Apply this condition for next quartile calculation
            if ind == 0:
                mask = working_df[cond_var] <= quartiles[1]
            elif ind == 1:
                mask = (working_df[cond_var] > quartiles[1]) & (working_df[cond_var] <= quartiles[2])
            elif ind == 2:
                mask = (working_df[cond_var] > quartiles[2]) & (working_df[cond_var] <= quartiles[3])
            elif ind == 3:
                mask = working_df[cond_var] > quartiles[3]
            
            # Check if this condition results in empty dataset
            if mask.sum() == 0:
                # No observations meet this condition - reject this variable combination
                return [], []
                
            working_df = working_df[mask].copy()
    
    # Second pass: create masks on FULL dataset using determined bounds
    for bound in condition_bounds:
        if bound[0] == 'boolean':
            _, cond_var, ind = bound
            cond = data_df[cond_var] == ind
            description = f"{cond_var} == {ind}"
            conds.append(cond)
            descriptions.append(description)
        else:
            if bound[2] == 'le':
                _, cond_var, _, threshold = bound
                cond = data_df[cond_var] <= threshold
                var_name = get_variable_name(cond_var)
                description = f"{var_name} ≤ {threshold:.1f}"
            elif bound[2] == 'between':
                _, cond_var, _, lower, upper = bound
                cond = (data_df[cond_var] > lower) & (data_df[cond_var] <= upper)
                var_name = get_variable_name(cond_var)
                description = f"{lower:.1f} < {var_name} ≤ {upper:.1f}"
            elif bound[2] == 'gt':
                _, cond_var, _, threshold = bound
                cond = data_df[cond_var] > threshold
                var_name = get_variable_name(cond_var)
                description = f"{var_name} > {threshold:.1f}"
            
            conds.append(cond)
            descriptions.append(description)
    
    return conds, descriptions


def condition_signature(base_variable, conditions):
    """Return an order-insensitive signature for a set of conditions on a base variable.

    Ignores the third numeric value per condition and only keeps (cond_var, index).
    This allows us to detect semantic duplicates across generated variables.
    """
    if not conditions:
        return (base_variable, ())
    reduced = tuple(sorted((cond[0], float(cond[1])) for cond in conditions))
    return (base_variable, reduced)


def apply_conditions(data_df, var, conditions):
    conds, descriptions = get_conditions(data_df, conditions)
    subpop_mask = combine_conditions_to_mask(data_df, conds)

    if subpop_mask.sum() < 30:
        return None
    
    mean_result = nhanes_domain_mean_and_variance(
        data_df,
        var,
        weight_name='WTMEC2YR',
        strata_name='SDMVSTRA',
        psu_name='SDMVPSU',
        domain_var=subpop_mask
    )
    if mean_result is None:
        return None

    return mean_result['mean'], mean_result['stdev'], mean_result['se'], mean_result['mean_lognormal'], mean_result['std_lognormal']


def sample_conditions(data_df, var, num_conditions, all_conditions):
    # Pick random keys from all_conditions that aren't the base variable
    available_vars = [k for k in all_conditions.keys() if k != var]
    
    # If we don't have enough distinct variables, return None
    if len(available_vars) < num_conditions:
        return None, None
        
    # Sample distinct condition variables
    cond_vars = random.sample(available_vars, num_conditions)
    
    all_conds = []
    used_vars = set()
    
    for cond_var in cond_vars:
        if cond_var in used_vars:
            continue
            
        conds = all_conditions[cond_var]
        # Randomly sample one condition from the available conditions
        sampled_cond = random.choice(conds)
        all_conds.append(sampled_cond)
        used_vars.add(cond_var)
        
    res = apply_conditions(data_df, var, all_conds)
    return res, all_conds
    


# Create variables with different numbers of conditionals by randomly sample conditionals on a base variable and check if 
# they shift the point estimate for that variable by more than a predefined threshold, keep if they do, resample if they don't 
def create_variables_by_difficulty(data_df, ground_truths, all_conditions, num_easy, num_medium, num_hard, difference_threshold):
    variables_by_difficulty = {}
    seen_signatures = set()
    # Initialize base variables
    for var, info in ground_truths.items(): 
        paraphrase = base_phrases[var]
        info['variable'] = paraphrase
        info['base_variable'] = var
        info['conditions'] = []
        if var in target_variables_boolean: 
            ground_truth_distribution_type = 'beta'
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
        info['ground_truth_distribution_type'] = ground_truth_distribution_type
        variables_by_difficulty[var] = info
        seen_signatures.add(condition_signature(var, []))
        
    easy_vars = 0
    attempts = 0  
    while easy_vars < num_easy:
        attempts += 1

        var = random.choice(list(ground_truths.keys()))
        res, conds = sample_conditions(data_df, var, 1, all_conditions)
        if res is None:
            continue
        mean, stdev, se, mean_lognormal, std_lognormal = res

        if (var in target_variables_boolean and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        varname = 'easy_{}'.format(easy_vars)

        signature = condition_signature(var, conds)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if var in target_variables_boolean:
            ground_truth_distribution_type = 'beta'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'difficulty': 'easy', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': mean_lognormal, 'std_lognormal': std_lognormal, 'base_variable': var, 'conditions': conds, 'difficulty': 'easy', 'ground_truth_distribution_type': ground_truth_distribution_type}

        easy_vars += 1

    medium_vars = 0
    attempts = 0  
    while medium_vars < num_medium:
        attempts += 1
        var = random.choice(list(ground_truths.keys()))
        res, conds = sample_conditions(data_df, var, 2, all_conditions)
        if res is None:
            continue
        mean, stdev, se, mean_lognormal, std_lognormal = res

        if (var in target_variables_boolean and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        varname = 'medium_{}'.format(medium_vars)

        signature = condition_signature(var, conds)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if var in target_variables_boolean:
            ground_truth_distribution_type = 'beta'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'difficulty': 'medium', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': mean_lognormal, 'std_lognormal': std_lognormal, 'base_variable': var, 'conditions': conds, 'difficulty': 'medium', 'ground_truth_distribution_type': ground_truth_distribution_type}

        medium_vars += 1

    hard_vars = 0   
    attempts = 0  
    while hard_vars < num_hard:
        attempts += 1
        var = random.choice(list(ground_truths.keys()))
        res, conds = sample_conditions(data_df, var, 3, all_conditions)
        if res is None:
            continue
        mean, stdev, se, mean_lognormal, std_lognormal = res

        if (var in target_variables_boolean and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        varname = 'hard_{}'.format(hard_vars)
        signature = condition_signature(var, conds)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if var in target_variables_boolean:
            ground_truth_distribution_type = 'beta'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'difficulty': 'hard', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
            variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': mean_lognormal, 'std_lognormal': std_lognormal, 'base_variable': var, 'conditions': conds, 'difficulty': 'hard', 'ground_truth_distribution_type': ground_truth_distribution_type}

        hard_vars += 1
    return variables_by_difficulty  


# Create natural language descriptions of all variables
def get_quartile_df_nat_langs(data_df, cond_var, quartile):
    # Calculate quartiles dynamically based on the current filtered dataset
    unique_values = data_df[cond_var].dropna()
    quartiles = np.percentile(unique_values, [1, 25, 50, 75, 99])
    var_name = cond_phrases[cond_var]
    if quartile == 0:
        nat_lang = "less than or equal to {}".format(quartiles[1])
    elif quartile == 1:
        nat_lang = "greater than {} and less than or equal to {}".format(quartiles[1], quartiles[2])
    elif quartile == 2:
        nat_lang = "greater than {} and less than or equal to {}".format(quartiles[2], quartiles[3])
    elif quartile == 3:
        nat_lang = "greater than {}".format(quartiles[3])
    else: 
        raise ValueError("Quartile must be 0, 1, 2, or 3.")
    var_name = cond_phrases[cond_var] + " " + nat_lang
    return var_name


def paraphrase_conditions(data_df, base, conditions):
    if base in target_variables_boolean and len(conditions) == 0:
        txt = base_phrases[base]
        return txt
    elif base in target_variables_continuous and len(conditions) == 0:
        txt = base_phrases[base]
        return txt
    else:
        base_txt = base_phrases[base]
        nat_langs = []
        
        # Start with the full dataset for the first condition
        working_df = data_df.copy()
        
        for cond in conditions:
            cond_var, ind = cond
            if cond_var in target_variables_boolean: 
                var_name = get_variable_name(cond_var)
                if ind == 1.0: 
                    cond_phrase = cond_phrases[cond_var][True]
                else: 
                    cond_phrase = cond_phrases[cond_var][False]
                nat_langs.append(cond_phrase)
                mask = working_df[cond_var] == ind
                working_df = working_df[mask].copy()
            elif cond_var in target_variables_continuous: 
                # Use current state of working_df for dynamic quartile calculation
                nat_lang = get_quartile_df_nat_langs(working_df, cond_var, ind)
                nat_langs.append(nat_lang)
                
                # Apply this quartile condition to working dataset for next iteration
                cond_mask, _ = get_quartile_df(working_df, cond_var, ind)
                working_df = working_df[cond_mask].copy()

        # Combine base text with conditions
        conditions_text = ", given " + ", and ".join(nat_langs)
        full_text = base_txt + conditions_text
        return full_text


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


def save_sample_to_csv(sample_df, variable, n, trial_idx, base_dir="baseline_data_samples"):
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
    sample_dir = Path(base_dir) / variable
    sample_dir.mkdir(parents=True, exist_ok=True)
    file_path = sample_dir / f"n{n}_trial{trial_idx}.csv"
    
    try:
        sample_df.reset_index(drop=True).to_csv(file_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save sample {file_path}: {e}")
    
    return file_path


def get_conditions(data_df, conditions):
    """Progressive filtering to determine bounds, but create masks on full dataset"""
    conds = []
    descriptions = []
    working_df = data_df.copy()  # For determining bounds only
    
    # Store the bounds for each condition
    condition_bounds = []
    
    # First pass: determine all bounds using progressive filtering
    for cond_var, ind in conditions:
        if cond_var in target_variables_boolean:
            condition_bounds.append(('boolean', cond_var, ind))
            # Apply this condition for next quartile calculation
            mask = working_df[cond_var] == ind
            working_df = working_df[mask].copy()
        else:
            # Calculate quartiles on current filtered dataset
            filtered_values = working_df[cond_var].dropna()
            
            # Check if we have enough data for quartiles
            if len(filtered_values) < 4:
                # Not enough data for meaningful quartiles - reject this variable combination
                return [], []
            
            quartiles = np.percentile(filtered_values, [1, 25, 50, 75, 99])
            
            # Store the actual bounds
            if ind == 0:
                condition_bounds.append(('continuous', cond_var, 'le', quartiles[1]))
            elif ind == 1:
                condition_bounds.append(('continuous', cond_var, 'between', quartiles[1], quartiles[2]))
            elif ind == 2:
                condition_bounds.append(('continuous', cond_var, 'between', quartiles[2], quartiles[3]))
            elif ind == 3:
                condition_bounds.append(('continuous', cond_var, 'gt', quartiles[3]))
            
            # Apply this condition for next quartile calculation
            if ind == 0:
                mask = working_df[cond_var] <= quartiles[1]
            elif ind == 1:
                mask = (working_df[cond_var] > quartiles[1]) & (working_df[cond_var] <= quartiles[2])
            elif ind == 2:
                mask = (working_df[cond_var] > quartiles[2]) & (working_df[cond_var] <= quartiles[3])
            elif ind == 3:
                mask = working_df[cond_var] > quartiles[3]
            
            # Check if this condition results in empty dataset
            if mask.sum() == 0:
                # No observations meet this condition - reject this variable combination
                return [], []
                
            working_df = working_df[mask].copy()
    
    # Second pass: create masks on FULL dataset using determined bounds
    for bound in condition_bounds:
        if bound[0] == 'boolean':
            _, cond_var, ind = bound
            cond = data_df[cond_var] == ind
            description = f"{cond_var} == {ind}"
            conds.append(cond)
            descriptions.append(description)
        else:
            if bound[2] == 'le':
                _, cond_var, _, threshold = bound
                cond = data_df[cond_var] <= threshold
                description = f"{cond_var} <= {threshold:.1f}"
            elif bound[2] == 'between':
                _, cond_var, _, lower, upper = bound
                cond = (data_df[cond_var] > lower) & (data_df[cond_var] <= upper)
                description = f"{lower:.1f} < {cond_var} <= {upper:.1f}"
            elif bound[2] == 'gt':
                _, cond_var, _, threshold = bound
                cond = data_df[cond_var] > threshold
                description = f"{cond_var} > {threshold:.1f}"
            
            conds.append(cond)
            descriptions.append(description)
    
    return conds, descriptions


def apply_conditions_get_data_subset(data_df, var, conditions):
    """Apply conditions using the same approach as generate_variables.ipynb"""
    if not conditions:
        # No conditions - return the full dataset
        df_subset = data_df[data_df[var].notna()]
        if df_subset.shape[0] < 30:
            return None
        return df_subset
    
    # Get condition masks using the same logic as the notebook
    conds, descriptions = get_conditions(data_df, conditions)
    if not conds:  # get_conditions returned empty (insufficient data)
        return None
        
    # Combine all condition masks
    subpop_mask = combine_conditions_to_mask(data_df, conds)
    
    # Apply the combined mask and filter for non-missing target variable
    df_subset = data_df[subpop_mask & data_df[var].notna()]
    
    if df_subset.shape[0] < 30:
        return None
    
    return df_subset


def generate_nhanes(generation_config): 
    difference_threshold = generation_config['difference_threshold']
    gt = {}
    df = load_and_preprocess_nhanes_data()
    for var in target_variables_continuous + target_variables_boolean:
        results_manual = nhanes_domain_mean_and_variance(df, var, domain_var=None,
                        weight_name='WTMEC2YR',
                        strata_name='SDMVSTRA',
                        psu_name='SDMVPSU')
        if var in target_variables_continuous:
            gt[var] = {'mean': results_manual['mean'], 'std': results_manual['stdev'], 'se': results_manual['se'], 'mean_lognormal': results_manual['mean_lognormal'], 'std_lognormal': results_manual['std_lognormal']}
        else:  # boolean variables
            gt[var] = {'mean': results_manual['mean'], 'std': results_manual['stdev'], 'se': results_manual['se']}
    
    all_possible_conditions = {}
    for var in target_variables_boolean + target_variables_continuous:
        all_possible_conditions[var] = []
        unique_values = df[var].unique()
        if var in target_variables_boolean:
            for value in unique_values:
                all_possible_conditions[var].append((var, value))
        else:
            for i, q in enumerate(range(4)):
                all_possible_conditions[var].append((var, q))
    
    variables = create_variables_by_difficulty(
        df,
        gt,
        all_possible_conditions,
        generation_config['target_num_single_condition_vars'],
        generation_config['target_num_double_condition_vars'],
        generation_config['target_num_triple_condition_vars'],
        difference_threshold
    )

    for var, info in variables.items():
        base = info['base_variable']
        conditions = info['conditions']
        paraphrase = paraphrase_conditions(df, base, conditions)
        variables[var]['variable'] = paraphrase


    # 2. Constants
    SUBSAMPLE_SIZES = [5, 10, 20, 30]  # target effective n
    RESAMPLES_PER_N = 25                     # M

    # uninformative priors
    ALPHA0, BETA0 = 1.0, 1.0
    MU0, SIGMA0 = 0.0, 100_000.0            # huge σ₀ → flat prior

    baselines: Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]] = {}

    base_dir = "baseline_data_samples"
    dataset = 'nhanes'

    # 3. Iterate over variables
    for var_key, spec in variables.items():
        base_var = spec["base_variable"]
        conditions = spec.get("conditions", [])

        subset = apply_conditions_get_data_subset(df, base_var, conditions)
        if subset is None or subset.empty:
            continue

        # Keep only the variable and weights, drop missing
        subset = subset[[base_var, "WTMEC2YR"]].dropna(subset=[base_var])

        # directory to save the individual resamples for this variable
        sample_var_dir = Path("baselines") / Path(dataset) / Path(base_dir) / var_key

        sample_var_dir.mkdir(parents=True, exist_ok=True)

        baselines[var_key] = {}

        is_boolean = base_var in target_variables_boolean

        # ──────────────────────────────────────────────────────────────
        # Compute posterior using ALL available data (entry "all")
        # ──────────────────────────────────────────────────────────────
        w = subset["WTMEC2YR"]
        n_eff = kish_effn(w)
        if base_var in target_variables_boolean:
            # For boolean variables, compute weighted proportion
            p_hat = float(np.average(subset[base_var], weights=w))
            s_eff = p_hat * n_eff
            alpha, beta = complex_beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
            baseline_all = {
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": alpha / (alpha + beta),
                "effective_n": n_eff,
                "actual_n": len(subset),
                "weighted_proportion": p_hat
            }
        else:
            # For continuous variables, compute weighted mean and std
            dsw = DescrStatsW(subset[base_var], weights=w, ddof=0)
            mean_hat = float(dsw.mean)
            pop_sd = float(dsw.std)
            mu_n, sig_n = complex_gaussian_posterior(MU0, SIGMA0, n_eff, mean_hat, pop_sd)
            baseline_all = {
                "mu": mu_n,
                "sigma": sig_n,
                "posterior_mean": mu_n,
                "effective_n": n_eff,
                "actual_n": len(subset),
                "weighted_mean": mean_hat,
                "weighted_std": pop_sd
            }
        # Add the "all" entry to the output for this variable
        baselines[var_key]["all"] = [baseline_all]

        for n in SUBSAMPLE_SIZES:
            if n == "all":
                # Special case: use all available data
                trials: List[Dict[str, float]] = []
                lognorm_trials: List[Dict[str, float]] = []
                
                for trial_idx in range(RESAMPLES_PER_N):
                    # Use the full subset as the sample
                    samp = subset.copy()
                    w = samp["WTMEC2YR"]
                    n_eff = kish_effn(w)
                    
                    # Save this sample to disk for debugging / reproducibility
                    sample_path = sample_var_dir / f"nall_trial{trial_idx}.csv"
                    samp.reset_index(drop=False).to_csv(sample_path, index=False)

                    # ──────────────────────────────────────────────────────
                    # Posterior update
                    # ──────────────────────────────────────────────────────
                    if is_boolean:
                        # weighted success proportion
                        p_hat = float(np.average(samp[base_var], weights=w))
                        s_eff = p_hat * n_eff
                        alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                        trials.append({"alpha": alpha, "beta": beta})
                    else:  # continuous
                        dsw = DescrStatsW(samp[base_var], weights=w, ddof=0)
                        mean_hat = float(dsw.mean)
                        pop_sd = float(dsw.std)  # population σ
                        mu_n, sig_n = gaussian_posterior(
                            MU0, SIGMA0 * pop_sd, n_eff, mean_hat, pop_sd
                        )
                        trials.append({"mu": mu_n, "sigma": sig_n})

                        dsw_log = DescrStatsW(np.log(samp[base_var] + 1e-6), weights=w, ddof=0)
                        mean_hat_log = float(dsw_log.mean)
                        pop_sd_log = float(dsw_log.std)  # population σ in log-scale
                        # Use log-space priors (mean of log is ~log of median, use wide prior)
                        MU0_LOG = 0.0
                        SIGMA0_LOG = 10.0 + pop_sd_log
                        mu_n_log, sig_n_log = gaussian_posterior(
                            MU0_LOG, SIGMA0_LOG, n_eff, mean_hat_log, pop_sd_log
                        )
                        lognorm_trials.append({"mu": mu_n_log, "sigma": sig_n_log})
                
                baselines[var_key]["all"] = trials
                baselines[var_key]["all_lognorm"] = lognorm_trials
                continue
            
            if len(subset) < n:
                continue

            trials: List[Dict[str, float]] = []
            lognorm_trials: List[Dict[str, float]] = []

            for trial_idx in range(RESAMPLES_PER_N):
                # ──────────────────────────────────────────────────────
                # Draw until Kish effective n ≥ target n (or exhausted)
                # ──────────────────────────────────────────────────────
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
                # Save this sample to disk for debugging / reproducibility
                sample_path = sample_var_dir / f"n{n}_trial{trial_idx}.csv"
                samp.reset_index(drop=False).to_csv(sample_path, index=False)

                # ──────────────────────────────────────────────────────
                # Posterior update
                # ──────────────────────────────────────────────────────
                if is_boolean:
                    # weighted success proportion
                    p_hat = float(np.average(samp[base_var], weights=w))
                    s_eff = p_hat * n_eff
                    alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                    trials.append({"alpha": alpha, "beta": beta})
                else:  # continuous
                    dsw = DescrStatsW(samp[base_var], weights=w, ddof=0)
                    mean_hat = float(dsw.mean)
                    pop_sd = float(dsw.std)  # population σ
                    mu_n, sig_n = gaussian_posterior(
                        MU0, SIGMA0 * pop_sd_log, n_eff, mean_hat, pop_sd
                    )
                    trials.append({"mu": mu_n, "sigma": sig_n})

                    dsw_log = DescrStatsW(np.log(samp[base_var] + 1e-6), weights=w, ddof=0)
                    mean_hat_log = float(dsw_log.mean)
                    pop_sd_log = float(dsw_log.std)  # population σ in log-scale
                    # Use log-space priors
                    MU0_LOG = 0.0
                    SIGMA0_LOG = 10.0 + pop_sd_log
                    mu_n_log, sig_n_log = gaussian_posterior(
                        MU0_LOG, SIGMA0_LOG, n_eff, mean_hat_log, pop_sd_log
                    )
                    lognorm_trials.append({"mu": mu_n_log, "sigma": sig_n_log})

            baselines[var_key][str(n)] = trials
            baselines[var_key][f"{n}_lognorm"] = lognorm_trials
    return variables, baselines


if __name__ == "__main__":
    data_df = load_and_preprocess_nhanes_data()
    # Define a fixed set of three conditions
    fixed_conditions = [
        ("WaistCircumferenceCm", 1),  # Quartile 1 for Waist Circumference
        ("CoveredByHealthInsurance", True),  # Boolean condition for health insurance
        ("BloodLeadUgdl", 2)  # Quartile 2 for Blood Lead level
    ]

    # Generate a variable using the fixed conditions
    test_variable = "BloodCadmiumUgl"
    test_result = apply_conditions(data_df, test_variable, fixed_conditions)

    # Check if the result is valid and print the output
    if test_result:
        mean, stdev, se = test_result
        print(f"Generated variable with fixed conditions:")
        print(f"Mean: {mean:.2f}, Standard Deviation: {stdev:.2f}, Standard Error: {se:.2f}")
    else:
        print("Failed to generate a variable with the fixed conditions.")