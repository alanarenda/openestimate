from __future__ import annotations
import re
import os
import json
import random 
import kagglehub
import numpy as np 
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Union
from utils import (check_difference_threshold_continuous, check_difference_threshold_proportion,
compute_mean_continuous, SUBSAMPLE_SIZES, RESAMPLES_PER_N, ALPHA0, BETA0, MU0, SIGMA0, save_sample_to_csv,
gaussian_posterior, beta_posterior, compute_lognormal_mean_continuous)

random.seed(42)

target_variables_discrete = ['IsPublic', 'Sector', 'Location', 'Size', 'Revenue']
target_variables_continuous = ['Midpoint Salary']
cond_phrases = {
    'IsPublic': 'the company is {}',
    'Sector': 'the company is in the {} sector',
    'Location': 'the company is in {}',
    'Size': 'the company has {}',
    'Revenue': 'the company has {} revenue',
}


def get_quartile_df(df, cond_var, quartile):
    values = df[cond_var].dropna()
    quartiles = np.percentile(values, [1, 25, 50, 75, 99])
    q25, q50, q75 = quartiles[1], quartiles[2], quartiles[3]
    print("Cond var: {}, Quartiles: {} {} {}".format(cond_var, q25, q50, q75))

    if quartile == 0:
        return df[df[cond_var] <= q25], f"{cond_var} ≤ {q25:.2f}"
    elif quartile == 1:
        return df[(df[cond_var] > q25) & (df[cond_var] <= q50)], f"{cond_var} in ({q25:.2f}, {q50:.2f}]"
    elif quartile == 2:
        return df[(df[cond_var] > q50) & (df[cond_var] <= q75)], f"{cond_var} in ({q50:.2f}, {q75:.2f}]"
    elif quartile == 3:
        return df[df[cond_var] > q75], f"{cond_var} > {q75:.2f}"

    
def apply_conditions_get_data_subset(data_df, var, conditions):
    df_subset = data_df.copy()
    descriptions = []
    for cond_var, ind in conditions:
        if cond_var in target_variables_discrete:
            df_subset = df_subset.loc[df_subset[cond_var] == ind]
            description = cond_phrases[cond_var].format(ind)
        else:
            df_subset, description = get_quartile_df(df_subset, cond_var, ind)
        descriptions.append(description)
    # ensure at least 30 valid datapoints for the target variable
    var = var.split('_')[0]
    df_subset = df_subset[df_subset[var].notna()]
    if df_subset.shape[0] < 30:
        return None, None # Return None when insufficient data
    else:
        return df_subset, descriptions  


def extract_salary_range_from_text(salary_description: str) -> tuple:
    """
    Extracts the salary range from text.
    First, it attempts to extract numeric values using regex.
    If that fails, it calls the LLM using the new OpenAI client to extract the low and high salary.
    Returns a tuple (low, high) if successful, else None.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Skip invalid entries
    if not salary_description or salary_description.strip() in ["-1", "nan", ""]:
        return None
    
    # Check if this is an hourly rate
    is_hourly = "per hour" in salary_description.lower() or "hour" in salary_description.lower()
    
    # Enhanced regex pattern to handle K suffix and various formats
    pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)[Kk]?'
    matches = re.findall(pattern, salary_description)
    
    if matches:
        try:
            # Convert matches to actual numbers
            numbers = []
            for match in matches:
                # Remove commas and convert to float first to handle decimals
                num = float(match.replace(',', ''))
                
                if is_hourly:
                    # Convert hourly to annual (assume 40 hours/week, 52 weeks/year)
                    annual_salary = int(num * 40 * 52)
                    numbers.append(annual_salary)
                else:
                    # Check if this looks like it has K suffix in original
                    if 'K' in salary_description or 'k' in salary_description:
                        # If the number is small (likely in thousands), multiply by 1000
                        if num < 1000:
                            num *= 1000
                    numbers.append(int(num))
            
            if len(numbers) >= 2:
                return (min(numbers), max(numbers))
            elif len(numbers) == 1:
                # If only one number, use it as both low and high
                return (numbers[0], numbers[0])
                
        except (ValueError, IndexError):
            pass

    # Fallback using LLM extraction via the new OpenAI client
    prompt = (
        f"Extract the salary from the following text. If it's hourly, convert to annual salary (40 hours/week, 52 weeks/year). "
        f"Return ONLY the annual salary as a number. Convert K to thousands (e.g., 85K = 85000):\n"
        f"{salary_description}\n"
        f"Format: just the number"
    )
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30
        )
        answer = response.choices[0].message.content.strip()
        
        # Extract numbers from LLM response
        numbers = re.findall(r'\d+', answer)
        if len(numbers) >= 1:
            num = int(numbers[0])
            return (num, num)
            
    except Exception as e:
        print(f"Error during LLM extraction for '{salary_description}': {e}")
    
    return None


def midpoint_salary(salary_range: tuple) -> float:
    """
    Calculates the midpoint (average) salary given a tuple of (low, high) salaries.
    """
    if salary_range is None:
        return None
    return sum(salary_range) / 2


def categorize_job_title_detailed(title_str):
    """
    Categorize job titles with seniority levels preserved.
    """
    if pd.isna(title_str):
        return 'Other', 'Unknown'
    
    title_lower = str(title_str).lower()
    
    # Determine seniority level
    if any(term in title_lower for term in ['senior', 'sr.', 'lead', 'principal', 'staff', 'phd', 'manager']):
        seniority = 'Senior'
    elif any(term in title_lower for term in ['junior', 'jr.', 'associate', 'entry']):
        seniority = 'Junior'
    elif any(term in title_lower for term in ['intern', 'graduate']):
        seniority = 'Entry'
    else:
        seniority = 'Mid'
    
    # Determine job category
    if 'data scientist' in title_lower or 'data science' in title_lower:
        category = 'Data Scientist'
    elif 'data analyst' in title_lower or 'data analytics' in title_lower or 'analytics' in title_lower:
        category = 'Data Analyst'
    elif any(term in title_lower for term in ['data engineer', 'machine learning', 'ml engineer']):
        category = 'Data Engineer'
    elif 'business analyst' in title_lower:
        category = 'Business Analyst'
    elif any(term in title_lower for term in ['consultant', 'specialist']):
        category = 'Consultant'
    elif 'data engineer' in title_lower:
        category = 'Data Engineer'
    elif 'research' in title_lower:
        category = 'Research Scientist'
    elif 'database' in title_lower:
        category = 'Database Administrator'
    elif 'product manager' in title_lower:
        category = 'Product Manager'
    else:
        category = 'Other'
    return category, seniority


def load_glassdoor_data(): 
    if not os.path.exists('data/glassdoor_data.csv'): 
        path = kagglehub.dataset_download("rrkcoder/glassdoor-data-science-job-listings")
        data = pd.read_csv(path + "/glassdoor_jobs.csv")
        # Apply extraction to the salary estimate column in the DataFrame
        data["Salary_Range"] = data["Salary Estimate"].apply(
            lambda x: extract_salary_range_from_text(str(x)) if pd.notna(x) else None
        )
        data["Midpoint_Salary"] = data["Salary_Range"].apply(
            lambda x: midpoint_salary(x) if x else None
        )
        data[['job_category', 'seniority_level']] = data['Job Title'].apply(
        lambda x: pd.Series(categorize_job_title_detailed(x))
        )

        # Drop rows where job category is 'Other' or job title is untitled/unknown
        data = data[~((data['job_category'] == 'Other') | (data['Job Title'].str.lower().str.contains('untitled', na=False)))]

        # Filter to keep only Data Analyst, Data Scientist and Data Engineer roles
        data = data[data['job_category'].isin(['Data Analyst', 'Data Scientist', 'Data Engineer'])]

        data['Midpoint Salary'] = data['Midpoint_Salary']
        data['IsPublic'] = data['Type of ownership'].apply(lambda x: 'public' if x == 'Company - Public' else 'not public')
        data = data[data['Revenue'] != 'Unknown / Non-Applicable']
        data = data[data['Size'] != 'Unknown']
        for col in target_variables_discrete + target_variables_continuous:
            # Print value counts if the column is discrete (for checking purposes)
            if col in target_variables_discrete and col in data.columns:
                data[col] = data[col].replace("-1", np.nan)
                data[col] = data[col].replace(-1, np.nan)
            if col in target_variables_discrete and col in data.columns:
                data[col] = data[col].replace("-1", np.nan)
                data[col] = data[col].replace(-1, np.nan)

        # Drop rows with any NaN values in target variables
        data = data.dropna(subset=target_variables_discrete + target_variables_continuous)

        # Extract just the state from Location column
        data['Location'] = data['Location'].str.split(',').str[-1].str.strip()

        # Define a mapping from full state names to their abbreviations
        state_mapping = {
            "California": "CA",
            "Texas": "TX",
            "Michigan": "MI",
            "Maryland": "MD",
            "New Jersey": "NJ",
            "Illinois": "IL",
            "Arizona": "AZ",
            "Pennsylvania": "PA",
            "Delaware": "DE",
            "New York State": "NY", 
            "Virginia": "VA",
            "Manhattan": "NY", 
            "Point Loma": "CA"
        }

        # Replace full state names with abbreviations in the 'Location' column
        data["Location"] = data["Location"].replace(state_mapping)
        data.to_csv('data/glassdoor_data.csv', index=False)
    else: 
        data = pd.read_csv('data/glassdoor_data.csv')
    return data 


def compute_proportion_boolean(data_df, var, val):
    var = var.split('_')[0]
    col = data_df[var]  # Get single column instead of DataFrame
    # Compare each value to the single target value, not array of options
    boolean_indicator = [(1 if str(x) == str(val) else 0) for x in col]
    boolean_indicator = np.array(boolean_indicator)  # Convert to numpy array for mean calculation
    mean_proportion = boolean_indicator.mean()
    std_proportion = boolean_indicator.std()
    se_proportion = std_proportion / np.sqrt(len(boolean_indicator))
    return mean_proportion, std_proportion, se_proportion


def prep_base_phrases(data): 
    # Define base phrases for variables
    base_phrases = {
        'IsPublic': 'The probability that a company hiring data scientists is {}',
        'Sector': 'The probability that a company hiring data scientists is in the {} sector',
        'Location': 'The probability that a company hiring data scientists for roles in {}',
        'Midpoint Salary': 'The average midpoint of the posted salary range (in dollars) for data science and adjacent jobs',
        'Size': 'The probability that a company hiring data scientists has {}',
        'Revenue': 'The probability that a company hiring data scientists has {} revenue',
    }

    # Generate detailed phrases for discrete variables
    base_phrases2 = {
        f"{col}_{option}": base_phrases[col].format(option)
        for col in target_variables_discrete
        for option in data[col].unique()
    }

    # Add a specific phrase for 'Midpoint Salary'
    base_phrases2['Midpoint Salary'] = (
        'The average midpoint of the posted salary range (in dollars) for data science and adjacent jobs in the US'
    )

    # Update base_phrases with the new detailed phrases
    base_phrases = base_phrases2
    return base_phrases


def compute_ground_truths(data_df):
    ground_truths = {}
    for col in target_variables_discrete:
        options = data_df[col].unique()
        for option in options:
            mean_proportion, std_proportion, se_proportion = compute_proportion_boolean(data_df, col, option)
            var = "{}_{}".format(col, option)
            ground_truths[var] = { 'mean': mean_proportion, 'std': std_proportion, 'se': se_proportion, 'base_variable': var, 'ground_truth_distribution_type': 'beta'}

    # Compute ground truth proportions for base continuous variables
    for var in target_variables_continuous:
        mean_value, std_value, se_value = compute_mean_continuous(data_df, var)
        lognormal_mean_value, lognormal_std_value = compute_lognormal_mean_continuous(data_df, var)

        ground_truths[var] = { 'mean': mean_value, 'std': std_value, 'se': se_value, 'mean_lognormal': lognormal_mean_value, 'std_lognormal': lognormal_std_value, 'base_variable': var, 'ground_truth_distribution_type': 'normal'}
    return ground_truths


def apply_conditions(data_df, var, conditions):
    subset, descriptions = apply_conditions_get_data_subset(data_df, var, conditions)
    if subset is None:
        return None
    mean_value, std_value, se_value = compute_mean_continuous(subset, var)
    lognormal_mean_value, lognormal_std_value = compute_lognormal_mean_continuous(subset, var)
    res = (mean_value, std_value, se_value, lognormal_mean_value, lognormal_std_value)
    return {'res': res, 'nat_langs': descriptions}


def sample_conditions(data_df, var, num_conditions, all_conditions): 
    # Get available variables excluding the target variable
    available_vars = [k for k in all_conditions.keys() if k != var and k != 'Midpoint Salary']
    
    # Sample distinct columns for conditions
    all_conds = []
    
    # Sample num_conditions random conditions from available variables
    while len(all_conds) < num_conditions and available_vars:
        cond_var = random.choice(available_vars)
        sampled_cond = random.choice(all_conditions[cond_var])
        all_conds.append(sampled_cond)
        available_vars.remove(cond_var)
    
    res = apply_conditions(data_df, var, all_conds)
    return {"res": res, "conds": all_conds}


# Create variables with different numbers of conditionals by randomly sample conditionals on a base variable and check if 
# they shift the point estimate for that variable by more than a predefined threshold, keep if they do, resample if they don't 
def create_variables_by_difficulty(data_df, ground_truths, all_conditions, num_single, num_double, num_triple, base_phrases, difference_threshold):
    variables_by_difficulty = {}
    seen_signatures = []
    # Initialize base variables
    for var, info in ground_truths.items(): 
        if var == 'Midpoint Salary':
            paraphrase = base_phrases[var]
            info['variable'] = paraphrase
            info['base_variable'] = var
            info['conditions'] = []
            info['nat_langs'] = []
            firstpart = var.split('_')[0]
            if firstpart in target_variables_discrete: 
                ground_truth_distribution_type = 'beta'
            elif firstpart in target_variables_continuous:
                ground_truth_distribution_type = 'normal'
            info['ground_truth_distribution_type'] = ground_truth_distribution_type
            variables_by_difficulty[var] = info
            seen_signatures.append((var, []))
        
    single_vars = 0
    attempts = 0
    while single_vars < num_single and attempts < 10000:
        attempts += 1
        var = 'Midpoint Salary'
        results = sample_conditions(data_df, var, 1, all_conditions)
        res = results['res']
        conds = results['conds']
        if res is None:
            attempts += 1
            continue

        res, nat_langs = res['res'], res['nat_langs']
        mean, stdev, se, lognormal_mean, lognormal_std = res

        if (var in target_variables_discrete and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        if var in target_variables_discrete:
            ground_truth_distribution_type = 'beta'
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'

        varname = 'single_{}'.format(single_vars)
        signature = (var, nat_langs)
        dup = False
        for other_signature in seen_signatures:
            var_other, nat_langs_other = other_signature
            if var_other == var and set(nat_langs) == set(nat_langs_other):
                dup = True
        if dup:
            continue
        else:
            seen_signatures.append(signature)
        single_vars += 1
        variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs,'difficulty': 'single', 'ground_truth_distribution_type': ground_truth_distribution_type}


    double_vars = 0
    attempts = 0
    while double_vars < num_double and attempts < 10000:
        attempts += 1
        var = 'Midpoint Salary'
        results = sample_conditions(data_df, var, 2, all_conditions)
        res = results['res']
        conds = results['conds']
        if res is None:
            attempts += 1
            continue

        res, nat_langs = res['res'], res['nat_langs']
        mean, stdev, se, lognormal_mean, lognormal_std = res

        if (var in target_variables_discrete and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        if var in target_variables_discrete:
            ground_truth_distribution_type = 'beta'
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
        varname = 'double_{}'.format(double_vars)
        signature = (var, nat_langs)
        dup = False
        for other_signature in seen_signatures:
            var_other, nat_langs_other = other_signature
            if var_other == var and set(nat_langs) == set(nat_langs_other):
                dup = True
        if dup:
            continue
        else:
            seen_signatures.append(signature)
        double_vars += 1
        variables_by_difficulty[varname] = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'double', 'ground_truth_distribution_type': ground_truth_distribution_type}  

    triple_vars = 0
    attempts = 0
    while triple_vars < num_triple and attempts < 10000:
        attempts += 1
        var = 'Midpoint Salary'
        results = sample_conditions(data_df, var, 3, all_conditions)
        res = results['res']
        conds = results['conds']
        if res is None:
            attempts += 1
            continue

        res, nat_langs = res['res'], res['nat_langs']
        mean, stdev, se, lognormal_mean, lognormal_std = res

        if (var in target_variables_discrete and not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue
        elif (var in target_variables_continuous and not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold)) or (mean == 0 or stdev == 0):
            attempts += 1
            continue

        if var in target_variables_discrete:
            ground_truth_distribution_type = 'beta'
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
        varname = 'triple_{}'.format(triple_vars)

        signature = (var, nat_langs)
        dup = False
        for other_signature in seen_signatures:
            var_other, nat_langs_other = other_signature
            if var_other == var and set(nat_langs) == set(nat_langs_other):
                dup = True
        if dup:
            continue
        else:
            seen_signatures.append(signature)
        var_dict = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'triple', 'ground_truth_distribution_type': ground_truth_distribution_type}
        variables_by_difficulty[varname] = var_dict
        triple_vars += 1
    return variables_by_difficulty  


def prep_cond_phrases(data): 
    cond_phrases = {
        'IsPublic': 'the company is {}',
        'Sector': 'the company is in the {} sector',
        'Location': 'the company is in {}',
        'Size': 'the company has {}',
        'Revenue': 'the company has {} revenue',
    }

    cond_phrases2 = {}

    for col in target_variables_discrete:
        options = data[col].unique()
        for option in options:
            phrase = cond_phrases[col].format(option)
            cond_phrases2["{}_{}".format(col, option)] = phrase

    cond_phrases = cond_phrases2
    return cond_phrases


def is_discrete_variable(base_var):
    """Properly detect if a variable is discrete (has _value format) or continuous."""
    return '_' in base_var and base_var.split('_')[0] in target_variables_discrete


def compute_discrete_proportion(sample_df, base_var):
    """Properly compute proportion for discrete variables in Glassdoor format."""
    if '_' not in base_var:
        raise ValueError(f"Expected discrete variable format 'variable_value', got: {base_var}")
    
    var_name, target_value = base_var.split('_', 1)
    
    if var_name not in sample_df.columns:
        raise ValueError(f"Variable {var_name} not found in data")
    
    # Calculate proportion where variable equals target value
    matches = (sample_df[var_name] == target_value).astype(int)
    return float(matches.mean())


def generate_glassdoor(generation_config): 
    df = load_glassdoor_data()
    base_phrases = prep_base_phrases(df)
    gt = compute_ground_truths(df)
    all_possible_conditions = {}
    for var in target_variables_discrete + target_variables_continuous:
        all_possible_conditions[var] = []
        unique_values = df[var].unique()
        if var in target_variables_discrete:
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
        base_phrases,
        generation_config['difference_threshold']
    )

    cond_phrases = prep_cond_phrases(df)

    for var, info in variables.items():
        conds = info['conditions']
        synth = base_phrases[info['base_variable']]
        i = 0
        for cond in conds:
            cond_phrase = "{}_{}".format(cond[0], cond[1])
            if i == 0: 
                synth = synth + ", given " + cond_phrases[cond_phrase]
            elif i == 1:
                synth = synth + ", " + cond_phrases[cond_phrase]
            elif i == len(conds) - 1:
                synth = synth + ", and " + cond_phrases[cond_phrase]
            i += 1
            
        variables[var]['variable'] = synth

    baselines: Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]] = {}

    for var_key, spec in variables.items():
        base_var   = spec["base_variable"]
        conditions = spec.get("conditions", [])

        subset, descriptions = apply_conditions_get_data_subset(df, base_var, conditions)
        if subset is None or subset.empty:
            continue

        is_discrete = is_discrete_variable(base_var)
        
        if is_discrete:
            # For discrete variables, we need the base columns (e.g., IsPublic, Sector)
            var_name = base_var.split('_')[0]
            required_cols = [var_name]
        else:
            # For continuous variables, we need the variable itself
            required_cols = [base_var]
        
        # Keep only required columns, drop missing
        available_cols = [col for col in required_cols if col in subset.columns]
            
        subset = subset[available_cols].dropna(subset=available_cols)
                
        baselines[var_key] = {}

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
                save_sample_to_csv("glassdoor", samp, var_key, n, trial_idx)

                n_eff = len(samp)

                if is_discrete:
                    p_hat = compute_discrete_proportion(samp, base_var)
                    s_eff = p_hat * n_eff
                    alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                    trials.append({"alpha": alpha, "beta": beta})

                else:  # continuous
                    # Normal update
                    mean_hat = float(samp[base_var].mean())
                    pop_sd   = float(samp[base_var].std())
                    mu_n, sig_n = gaussian_posterior(
                        MU0, SIGMA0, n_eff, mean_hat, pop_sd
                    )

                    # Lognormal update
                    log_values = np.log(samp[base_var] + 1e-6)  # avoid log(0)
                    mean_log_hat = float(log_values.mean())
                    pop_log_sd   = float(log_values.std())
                    mu_n_log, sig_n_log = gaussian_posterior(MU0, SIGMA0, n_eff, mean_log_hat, pop_log_sd)
                    mu_n_exp = np.exp(mu_n_log + 0.5 * sig_n_log ** 2)
                    sig_n_exp = np.sqrt((np.exp(sig_n_log ** 2) - 1) * np.exp(2 * mu_n_log + sig_n_log ** 2))

                    trials.append({"mu": mu_n, "sigma": sig_n})
                    lognorm_trials.append({"mu": mu_n_exp, "sigma": sig_n_exp})

            baselines[var_key][str(n)] = trials
            if len(lognorm_trials) > 0:
                baselines[var_key][str(n) + "_lognorm"] = lognorm_trials

        all_trials: List[Dict[str, float]] = []
        all_lognorm_trials: List[Dict[str, float]] = []

        # Save the full subset as "ALL" sample
        save_sample_to_csv("glassdoor", subset, var_key, "ALL", 0)

        n_eff_all = len(subset)
        if is_discrete:
            p_hat_all = compute_discrete_proportion(subset, base_var)
            s_eff_all = p_hat_all * n_eff_all
            alpha_all, beta_all = beta_posterior(ALPHA0, BETA0, s_eff_all, n_eff_all)
            all_trials.append({"alpha": alpha_all, "beta": beta_all})
        else:
            mean_all = float(subset[base_var].mean())
            sd_all   = float(subset[base_var].std())
            mu_all, sig_all = gaussian_posterior(MU0, SIGMA0, n_eff_all, mean_all, sd_all)

            # Lognormal update for ALL
            log_values_all = np.log(subset[base_var] + 1e-6)  # avoid log(0)
            mean_log_all = float(log_values_all.mean())
            sd_log_all   = float(log_values_all.std())
            mu_all_log, sig_all_log = gaussian_posterior(MU0, SIGMA0, n_eff_all, mean_log_all, sd_log_all)
            mu_all_exp = np.exp(mu_all_log + 0.5 * sig_all_log ** 2)
            sig_all_exp = np.sqrt((np.exp(sig_all_log ** 2) - 1) * np.exp(2 * mu_all_log + sig_all_log ** 2))
            all_lognorm_trials.append({"mu": mu_all_exp, "sigma": sig_all_exp})

            all_trials.append({"mu": mu_all, "sigma": sig_all})

        baselines[var_key]["ALL"] = all_trials
        if len(all_lognorm_trials) > 0:
            baselines[var_key]["ALL_lognorm"] = all_lognorm_trials

    return variables, baselines 



