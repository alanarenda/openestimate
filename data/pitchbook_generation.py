from __future__ import annotations
import os 
import json
import random 
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI 
from typing import List, Dict, Union
from utils import (check_difference_threshold_continuous, check_difference_threshold_proportion, 
compute_mean_continuous, SUBSAMPLE_SIZES, RESAMPLES_PER_N, ALPHA0, BETA0, MU0, SIGMA0, save_sample_to_csv, 
gaussian_posterior, beta_posterior, compute_lognormal_mean_continuous)

random.seed(42)

target_variables_boolean = ['IsUSBased', 'IsTechCompany']
target_variables_continuous = ['TotalRaised', 'Employees']

base_phrases = {
    'IsUSBased': 'The probability that a venture-backed company is based in the US',
    'IsTechCompany': 'The probability that a venture-backed company is a technology company',
    'TotalRaised': 'The average total raised in millions USD for venture-backed companies',
    'Employees': 'The average number of employees for venture-backed companies'
}

cond_phrases = {
    'IsUSBased': {
        1: 'the company is based in the US',
        0: 'the company is not based in the US'
    },
    'IsTechCompany': {
        1: 'the company is a technology company',
        0: 'the company is not a technology company'
    },
    'TotalRaised':  'the total amount of money the company raised in millions USD', 
    'FirstFinancingSize': 'the amount of money the company raised in millions USD in its first financing',
    'Employees': 'the number of employees that work at the company',
}

def get_quartile_df(df, cond_var, quartile):
    unique_values = df[cond_var].dropna()
    quartiles = np.percentile(unique_values, [1, 25, 50, 75, 99])
    q25, q50, q75 = quartiles[1], quartiles[2], quartiles[3]

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
        if cond_var in target_variables_boolean:
            df_subset = df_subset.loc[df_subset[cond_var] == ind]
            description = cond_phrases[cond_var][ind]
        else:
            df_subset, description = get_quartile_df(df_subset, cond_var, ind)
        descriptions.append(description)
    
    # ensure at least 30 valid datapoints for the target variable
    df_subset = df_subset[df_subset[var].notna()]
    
    if df_subset.shape[0] < 30:
        return None, None # Return None when insufficient data
    else:
        return df_subset, descriptions  


def clean_value(value):
    """Clean and validate numeric values in millions"""
    try:
        # Convert to string first to handle any numeric formats
        str_value = str(value)
        # Remove any duplicate numbers and handle decimals
        if '.' in str_value:
            integer_part, decimal_part = str_value.split('.')
            str_value = f"{integer_part}.{decimal_part[:2]}"  # Keep 2 decimal places for funding
        return float(str_value)
    except (ValueError, TypeError):
        return None


def classify_cities_us_non_us(locations: Union[List[str], pd.Series], use_llm: bool = True, 
                             llm_model: str = "gpt-3.5-turbo", batch_size: int = 50) -> Dict[str, bool]:
    """
    Classify cities/locations as US or non-US with caching.
    """

    # Add caching
    CACHE_FILE = "data/location_classifications.json"

    # Load existing cache (use JSON consistently)
    try:
        with open(CACHE_FILE, 'r') as f:
            cached_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        cached_results = {}

    # Convert to list if pandas Series
    if isinstance(locations, pd.Series):
        locations = locations.dropna().unique().tolist()
    
    # Remove duplicates and filter out None/empty values
    unique_locations = list(set([loc for loc in locations if loc and isinstance(loc, str) and loc.strip()]))
    
    # Check what locations we already have cached
    locations_to_classify = [loc for loc in unique_locations if loc not in cached_results]
    # Start with cached results
    results = cached_results.copy()
    
    if not locations_to_classify:
        return {loc: results[loc] for loc in unique_locations}
    
    # Rule-based classification for new locations only
    us_states_abbrev = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
        'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
        'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    }
    
    us_states_full = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 
        'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 
        'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 
        'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 
        'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 
        'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 
        'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 
        'wisconsin', 'wyoming', 'district of columbia'
    }
    
    # Major US cities (expanded list)
    us_cities = {
        'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio',
        'san diego', 'dallas', 'san jose', 'austin', 'jacksonville', 'fort worth', 'columbus',
        'charlotte', 'san francisco', 'indianapolis', 'seattle', 'denver', 'washington dc',
        'boston', 'el paso', 'nashville', 'detroit', 'oklahoma city', 'portland', 'las vegas',
        'memphis', 'louisville', 'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno',
        'sacramento', 'kansas city', 'mesa', 'atlanta', 'omaha', 'colorado springs', 'raleigh',
        'miami', 'cleveland', 'tulsa', 'oakland', 'minneapolis', 'wichita', 'arlington', 'tampa',
        'new orleans', 'honolulu', 'anaheim', 'santa ana', 'st. louis', 'riverside', 'corpus christi',
        'lexington', 'pittsburgh', 'anchorage', 'stockton', 'cincinnati', 'st. paul', 'toledo',
        'greensboro', 'newark', 'plano', 'henderson', 'lincoln', 'buffalo', 'jersey city',
        'chula vista', 'fort wayne', 'orlando', 'st. petersburg', 'chandler', 'laredo', 'norfolk',
        'durham', 'madison', 'lubbock', 'irvine', 'winston-salem', 'glendale', 'garland',
        'hialeah', 'reno', 'chesapeake', 'gilbert', 'baton rouge', 'irving', 'scottsdale',
        'north las vegas', 'fremont', 'boise', 'richmond', 'san bernardino', 'birmingham'
    }
    
    # Common non-US indicators
    non_us_indicators = {
        'uk', 'united kingdom', 'england', 'london', 'canada', 'toronto', 'vancouver', 'montreal',
        'germany', 'berlin', 'munich', 'france', 'paris', 'italy', 'rome', 'milan', 'spain',
        'madrid', 'barcelona', 'netherlands', 'amsterdam', 'sweden', 'stockholm', 'denmark',
        'copenhagen', 'norway', 'oslo', 'finland', 'helsinki', 'switzerland', 'zurich', 'geneva',
        'austria', 'vienna', 'belgium', 'brussels', 'ireland', 'dublin', 'australia', 'sydney',
        'melbourne', 'japan', 'tokyo', 'china', 'beijing', 'shanghai', 'india', 'mumbai',
        'bangalore', 'singapore', 'hong kong', 'south korea', 'seoul', 'israel', 'tel aviv'
    }
    
    # Rule-based classification
    ambiguous_locations = []
    
    for location in locations_to_classify:
        location_lower = location.lower().strip()
        
        # Check for US indicators
        is_us = False
        is_non_us = False
        
        # Check for state abbreviations (e.g., "San Francisco, CA")
        if any(f", {state}" in location for state in us_states_abbrev):
            is_us = True
        elif any(f" {state}" in location for state in us_states_abbrev):
            is_us = True
            
        # Check for full state names
        elif any(state in location_lower for state in us_states_full):
            is_us = True
            
        # Check for major US cities
        elif any(city in location_lower for city in us_cities):
            is_us = True
            
        # Check for "USA" or "United States"
        elif any(indicator in location_lower for indicator in ['usa', 'united states', 'u.s.', 'us']):
            # Be careful with "us" - only if it's clearly indicating country
            if 'usa' in location_lower or 'united states' in location_lower or location_lower.endswith(' us'):
                is_us = True
                
        # Check for non-US indicators
        elif any(indicator in location_lower for indicator in non_us_indicators):
            is_non_us = True
            
        # Assign result or mark as ambiguous
        if is_us and not is_non_us:
            results[location] = True
        elif is_non_us and not is_us:
            results[location] = False
        else:
            ambiguous_locations.append(location)
        
    # Step 2: LLM classification for ambiguous cases
    if use_llm and ambiguous_locations:
        try:
            
            # Process in batches
            for i in range(0, len(ambiguous_locations), batch_size):
                batch = ambiguous_locations[i:i+batch_size]
                
                prompt = f"""
                Classify each of the following locations as either "US" (United States) or "NON-US".
                Consider cities, states, regions, and any location indicators.
                
                Return your response as a JSON object where each location is a key and the value is either "US" or "NON-US".
                
                Locations to classify:
                {json.dumps(batch, indent=2)}
                
                Response format:
                {{
                    "location1": "US",
                    "location2": "NON-US",
                    ...
                }}
                """
                
                try:
                    client = OpenAI()
                    response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "You are a geography expert that can accurately classify locations as US or non-US."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    
                    # Parse response
                    response_text = response.choices[0].message.content.strip()
                    
                    # Extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        llm_results = json.loads(response_text[json_start:json_end])
                        
                        # Add to results
                        for loc, classification in llm_results.items():
                            if loc in batch:  # Ensure it's from our batch
                                results[loc] = (classification.upper() == "US")
                    
                except Exception as e:
                    # Fallback: classify as non-US for safety
                    for loc in batch:
                        results[loc] = False
                        
        except ImportError:
            print("OpenAI library not available. Install with: pip install openai")
            # Fallback for ambiguous cases - use more conservative rules
            for location in ambiguous_locations:
                # Default to non-US for ambiguous cases
                results[location] = False
    
    else:
        # If not using LLM, default ambiguous cases to False (non-US)
        for location in ambiguous_locations:
            results[location] = False

    # IMPORTANT: Make sure ALL locations_to_classify get added to results
    # Add any remaining unprocessed locations as non-US (fallback)
    for location in locations_to_classify:
        if location not in results:
            results[location] = False
    
    # After rule-based classification, save intermediate results to cache
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")
    
    # Return only the requested locations
    return {loc: results[loc] for loc in unique_locations}


def apply_us_classification_to_dataframe(df: pd.DataFrame, location_column: str = 'HQLocation', 
                                       new_column: str = 'IsUSBased', use_llm: bool = True) -> pd.DataFrame:
    """
    Apply US/non-US classification to a DataFrame.
    
    Args:
        df: DataFrame containing location data
        location_column: Name of column containing location strings
        new_column: Name of new column to create with classification results
        use_llm: Whether to use LLM for ambiguous cases
    
    Returns:
        DataFrame with new classification column added
    """
    unique_locations = df[location_column].dropna().unique()
    classification_results = classify_cities_us_non_us(unique_locations, use_llm=use_llm)
    mapped_result = df[location_column].map(classification_results)
    filled_result = mapped_result.fillna(False)
    final_result = filled_result.astype(int)
    df[new_column] = final_result
    return df


def pitchbook_load(raw_df: pd.Dataframe) -> pd.DataFrame:
    """
    Preprocess the raw company DataFrame for variable generation pipeline.

    Steps:
    1. Filter to venture-backed companies.
    2. Select required columns.
    3. Drop rows with missing values in key fields.
    4. Clean numeric columns (TotalRaised, FirstFinancingSize).
    5. Convert Employees counts to ranges.

    Returns:
        A cleaned DataFrame ready for analysis.
    """
    # Make a copy to avoid modifying original
    df = raw_df.copy()

    # 1) Keep only venture-backed companies
    df = df[df['CompanyFinancingStatus'] == 'Venture Capital-Backed']

    # 2) Select the columns we need
    required_cols = [
        'CompanyFinancingStatus',
        'TotalRaised',
        'FirstFinancingSize',
        'HQLocation',
        'PrimaryIndustryGroup',
        'OwnershipStatus',
        'Employees'
    ]
    df = df[required_cols]

    # 3) Drop rows missing any of these
    df = df.dropna(subset=required_cols)

    # 4) Clean numeric columns
    df['TotalRaised'] = df['TotalRaised'].apply(clean_value)
    df['FirstFinancingSize'] = df['FirstFinancingSize'].apply(clean_value)
    # GrowthRate is assumed numeric; if not, cast:
    df = df.dropna(subset=['TotalRaised', 'FirstFinancingSize'])

    # 6) Create boolean variables
    df = apply_us_classification_to_dataframe(df, 'HQLocation', 'IsUSBased', use_llm=True)
    df['IsTechCompany'] = df['PrimaryIndustryGroup'].apply(lambda x: 1 if isinstance(x, str) and any(tech_term.lower() in x.lower() for tech_term in ['Software', 'Computer Hardware', 'Communications and Networking', 'IT Services', 'Healthcare Technology Systems', 'Other Information Technology', 'Semiconductors']) else 0)

    # Final drop in case converting employees introduced nulls
    df = df.dropna(subset=['Employees'])

    # Reset index and return
    return df.reset_index(drop=True)


def get_variable_name(var_name):
    # For pitchbook data, we'll use the variable name directly
    return var_name


def paraphrase_conditions(base, base_nat_lang, conditions=None):
    """Paraphrase the variable description to sound more natural"""
    
    if not conditions:
        text_to_paraphrase = f"Base variable: {base_nat_lang}"
    else:
        text_to_paraphrase = f"Base variable: {base_nat_lang}"
        for cond in conditions: 
            text_to_paraphrase += f"\nCondition: {cond}"
    
    messages = [
        {"role": "system", "content": """
            You are an expert at writing clear, concise statistical descriptions for a venture capital dataset. Your job is to turn variable and condition information into a single, natural-sounding English description. Follow these rules:

            1. For proportions, use the format: 'proportion of venture-backed companies with [characteristic]'.
            2. For averages about money, use the format: 'average [metric] in millions USD for venture-backed companies'.
            3. Don't be redundant. For example, don't say 'venture-backed companies' twice. Don't say 'private companies'. Just say 'venture-backed companies'.
            4. Never use parentheses in descriptions.
            5. Append conditions after the base description using: 'where [condition1] and [condition2] ...'. If it makes sense, you can also put it in the base description.
            6. If there are multiple conditions, join them with 'and'.
            7. Be consistent with terminology (do not mix 'proportion' and 'percentage'). 
            8. Always keep 'in millions USD' for monetary metrics. 
            9. Always include units about the conditions if applicable. For example, if a condition is growth rate of 5, say 'growth rate of 5%'. If a condition is financing size between 1 and 10 million USD, say 'financing size between 1 and 10 million USD'.
            10. Do not hallucinate or add any information not present in the prompt.
            11. Respond with only the description, no extra commentary.
            12. Don't use the variable name, i.e. if IsTechCompany is false, use 'non-technology companies' instead. Don't say IsTechCompany in the description.

            Examples:
            - Base variable only: 'The average total raised in millions USD for venture-backed companies.'
            - Base variable with one condition: 'The average total raised in millions USD for US-based venture-backed companies.'
            - Base variable with two conditions: 'The average total raised in millions USD for US-based venture-backed companies where the growth rate is between 0.1 and 0.5.'
            - Boolean variable: 'The proportion of venture-backed technology companies.'
            - Boolean variable with a condition: 'The proportion of venture-backed technology companies that are US-based.'
        """}
    ]
    messages.append({"role": "user", "content": text_to_paraphrase})
    client = OpenAI()
    response = client.chat_completion(
        model="o3-mini",
        messages=messages,
        stream=False,
    )
    res = response.choices[0].message.content
    return res


def compute_proportion_boolean(data_df, var):
    col = data_df[[var]]
    # Original boolean logic
    boolean_indicator = col[var].apply(lambda x: 1 if (x == '1' or x == 1.0) else 0)
    mean_proportion = boolean_indicator.mean()
    std_proportion = boolean_indicator.std()
    se_proportion = std_proportion / np.sqrt(len(boolean_indicator))
    return mean_proportion, std_proportion, se_proportion


def compute_ground_truths(data_df): 
    ground_truths = {}

    # Compute ground truth proportions for base boolean variables 
    for var in target_variables_boolean: 
        mean_proportion, std_proportion, se_proportion = compute_proportion_boolean(data_df, var)
        ground_truths[var] = { 'mean': mean_proportion, 'std': std_proportion, 'se': se_proportion, 'base_variable': var, 'ground_truth_distribution_type': 'beta'}

    # Compute ground truth proportions for base continuous variables 
    for var in target_variables_continuous:
        mean_value, std_value, se_value = compute_mean_continuous(data_df, var)
        lognormal_mean_value, lognormal_std_value = compute_lognormal_mean_continuous(data_df, var)
        ground_truths[var] = { 'mean': mean_value, 'std': std_value, 'se': se_value, 'mean_lognormal': lognormal_mean_value, 'std_lognormal': lognormal_std_value, 'base_variable': var, 'ground_truth_distribution_type': 'normal'}

    return ground_truths


def apply_conditions(data_df, var, conditions):
    res = apply_conditions_get_data_subset(data_df, var, conditions)
    subset, descriptions = res
    if subset is None:
        return None, None
    if var in target_variables_boolean:
        return {"res": compute_proportion_boolean(subset, var), "nat_langs": descriptions}
    else:
        mean_value, std_value, se_value = compute_mean_continuous(subset, var)
        lognormal_mean_value, lognormal_std_value = compute_lognormal_mean_continuous(subset, var)
        res = (mean_value, std_value, se_value, lognormal_mean_value, lognormal_std_value)
        return {"res": res, "nat_langs": descriptions} 


def sample_conditions(data_df, var, num_conditions, all_conditions): 
    # Get available variables excluding the target variable
    available_vars = [k for k in all_conditions.keys() if k != var]
    
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


def create_variables_by_difficulty(data_df, ground_truths, all_conditions, num_easy, num_medium, num_hard, difference_threshold):
    variables_by_difficulty = {}
    seen_signatures = []
    # Initialize base variables
    for var, info in ground_truths.items(): 
        paraphrase = base_phrases[var]
        info['variable'] = paraphrase
        info['base_variable'] = var
        info['conditions'] = []
        info['nat_langs'] = []
        if var in target_variables_boolean: 
            ground_truth_distribution_type = 'beta'
        elif var in target_variables_continuous:
            ground_truth_distribution_type = 'normal'
        info['ground_truth_distribution_type'] = ground_truth_distribution_type
        variables_by_difficulty[var] = info
        seen_signatures.append((var, []))
        
    easy_vars = 0
    attempts = 0
    while easy_vars < num_easy:
        attempts += 1

        if attempts > 1000:
            break

        var = random.choice(list(ground_truths.keys()))
        results = sample_conditions(data_df, var, 1, all_conditions)
        res_data = results['res']
        conds = results['conds']
        if res_data is None:
            continue
        res, nat_langs = res_data['res'], res_data['nat_langs']

        if var in target_variables_boolean:
            mean, stdev, se = res
            if not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'beta'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs,'difficulty': 'easy', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            mean, stdev, se, lognormal_mean, lognormal_std = res
            if not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'normal'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs,'difficulty': 'easy', 'ground_truth_distribution_type': ground_truth_distribution_type}

        varname = 'easy_{}'.format(easy_vars)
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
        easy_vars += 1
        variables_by_difficulty[varname] = var_dict
    medium_vars = 0
    attempts = 0
    while medium_vars < num_medium:
        attempts += 1
        if attempts > 1000:
            break

        var = random.choice(list(ground_truths.keys()))
        results = sample_conditions(data_df, var, 2, all_conditions)
        res_data = results['res']
        conds = results['conds']
        if res_data is None:
            continue
        res, nat_langs = res_data['res'], res_data['nat_langs']

        if var in target_variables_boolean:
            mean, stdev, se = res
            if not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'beta'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'medium', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            mean, stdev, se, lognormal_mean, lognormal_std = res
            if not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'normal'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'medium', 'ground_truth_distribution_type': ground_truth_distribution_type}

        varname = 'medium_{}'.format(medium_vars)
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
        medium_vars += 1
        variables_by_difficulty[varname] = var_dict
    hard_vars = 0
    attempts = 0
    while hard_vars < num_hard:
        attempts += 1
        if attempts > 1000:
            break

        var = random.choice(list(ground_truths.keys()))
        results = sample_conditions(data_df, var, 3, all_conditions)
        res_data = results['res']
        conds = results['conds']
        if res_data is None:
            continue
        res, nat_langs = res_data['res'], res_data['nat_langs']

        if var in target_variables_boolean:
            mean, stdev, se = res
            if not check_difference_threshold_proportion(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'beta'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'hard', 'ground_truth_distribution_type': ground_truth_distribution_type}
        elif var in target_variables_continuous:
            mean, stdev, se, lognormal_mean, lognormal_std = res
            if not check_difference_threshold_continuous(mean, ground_truths[var]['mean'], se, difference_threshold) or (mean == 0 or stdev == 0):
                attempts += 1
                continue
            ground_truth_distribution_type = 'normal'
            var_dict = {'mean': mean, 'std': stdev, 'se': se, 'mean_lognormal': lognormal_mean, 'std_lognormal': lognormal_std, 'base_variable': var, 'conditions': conds, 'nat_langs': nat_langs, 'difficulty': 'hard', 'ground_truth_distribution_type': ground_truth_distribution_type}

        varname = 'hard_{}'.format(hard_vars)

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
        variables_by_difficulty[varname] = var_dict
        hard_vars += 1
    return variables_by_difficulty  


def get_quartile_df_nat_langs(data_df, cond_var, quartile, conditions_so_far=None):
    """
    Compute quartiles dynamically on the current filtered subset
    """
    # Apply all conditions that have been processed so far to get the relevant subset
    if conditions_so_far:
        subset = data_df.copy()
        for prev_cond_var, prev_ind in conditions_so_far:
            if prev_cond_var in target_variables_boolean:
                subset = subset.loc[subset[prev_cond_var] == prev_ind]
            elif prev_cond_var in target_variables_continuous and prev_cond_var != cond_var:
                # Apply previous quartile conditions, but skip the current one
                subset, _ = get_quartile_df(subset, prev_cond_var, prev_ind, data_df)
    else:
        subset = data_df
    
    # Compute quartiles on this filtered subset
    full_vector = subset[cond_var].dropna()
    quartiles = np.percentile(full_vector, [25, 50, 75])
    q25, q50, q75 = quartiles
    
    var_name = cond_phrases[cond_var]
    if quartile == 0:
        nat_lang = "given that {} is less than or equal to {:.2f}".format(var_name, q25)
    elif quartile == 1:
        nat_lang = "given that {} is greater than {:.2f} and less than or equal to {:.2f}".format(var_name, q25, q50)
    elif quartile == 2:
        nat_lang = "given that {} is greater than {:.2f} and less than or equal to {:.2f}".format(var_name, q50, q75)
    elif quartile == 3:
        nat_lang = "given that {} is greater than {:.2f}".format(var_name, q75)
    
    return nat_lang


def paraphrase_conditions(base, nat_langs):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if base in target_variables_boolean and len(nat_langs) == 0:
        txt = base_phrases[base]
        base_txt = txt
    elif base in target_variables_continuous and len(nat_langs) == 0:
        txt = base_phrases[base]
        base_txt = txt
    else:
        base_txt = base_phrases[base]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert paraphraser. Rewrite part of the following sentence so it "
                    "sounds natural but does not lose any information. Respond ONLY with "
                    "the paraphrase. Specifically only paraphrase the conditions, not the base text. Return one full new sentence with the conditions paraphrased. Do not change the base text at all and do not change the meaning of the conditionals. The precise meaning of the sentence should be the same. Only include one grammatically correct sentence in your response. Round UP to the nearest integer when there's floating point numbers. When discussing the amount raised, clarify it's in millions USD."
                ),
            },
            {"role": "user", "content": "Base text: {}\nConditions: {}".format(base_txt, nat_langs)},
        ]
        response = client.chat.completions.create(
            model="o4-mini", messages=messages
        ).choices[0].message.content
        base_txt = response
    return base_txt


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


def generate_pitchbook(generation_config): 
    company_raw = pd.read_csv('Company.csv')
    df = pitchbook_load(company_raw)
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
    gt = compute_ground_truths(df)
    variables = create_variables_by_difficulty(
        df,
        gt,
        all_possible_conditions,
        generation_config['target_num_single_condition_vars'],
        generation_config['target_num_double_condition_vars'],
        generation_config['target_num_triple_condition_vars'],
        generation_config['difference_threshold']
    )
    for var, info in variables.items():
        base = info['base_variable']
        nat_langs = info['nat_langs']
        paraphrase = paraphrase_conditions(base, nat_langs)
        variables[var]['variable'] = paraphrase

    baselines: Dict[str, Dict[str, List[Dict[str, Union[float, int]]]]] = {}

    # 3. Iterate over variables
    for var_key, spec in variables.items():
        base_var   = spec["base_variable"]
        conditions = spec.get("conditions", [])

        subset, descriptions = apply_conditions_get_data_subset(df, base_var, conditions)
        if subset is None or subset.empty:
            continue

        # Keep only the variable, drop missing
        subset = subset[[base_var]].dropna(subset=[base_var])
        baselines[var_key] = {}

        is_boolean = base_var in target_variables_boolean

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
                save_sample_to_csv('pitchbook', samp, var_key, n, trial_idx)

                # For Pitchbook data, we don't have survey weights, so treat as unweighted
                n_eff = len(samp)

                if is_boolean:
                    # unweighted success proportion
                    p_hat = float(samp[base_var].mean())
                    s_eff = p_hat * n_eff
                    alpha, beta = beta_posterior(ALPHA0, BETA0, s_eff, n_eff)
                    trials.append({"alpha": alpha, "beta": beta})

                else:  # continuous
                    # Normal update
                    mean_hat = float(samp[base_var].mean())
                    pop_sd   = float(samp[base_var].std())          # sample σ
                    mu_n, sig_n = gaussian_posterior(
                        MU0, SIGMA0 * pop_sd, n_eff, mean_hat, pop_sd
                    )

                    # Lognormal update
                    log_values = np.log(samp[base_var] + 1e-6)  # avoid log(0)
                    mean_log_hat = float(log_values.mean())
                    pop_log_sd   = float(log_values.std())
                    # Use log-space priors (mean of log is ~log of median, use wide prior)
                    MU0_LOG = 0.0
                    SIGMA0_LOG = 10.0 * pop_log_sd
                    mu_n_log, sig_n_log = gaussian_posterior(MU0_LOG, SIGMA0_LOG, n_eff, mean_log_hat, pop_log_sd)

                    trials.append({"mu": mu_n, "sigma": sig_n})
                    lognorm_trials.append({"mu": mu_n_log, "sigma": sig_n_log})

            baselines[var_key][str(n)] = trials
            if len(lognorm_trials) > 0:
                baselines[var_key][str(n) + "_lognorm"] = lognorm_trials

        # ───────────────────────────────────────────────────
        # ALL-examples baseline (single trial using all rows)
        # ───────────────────────────────────────────────────
        all_trials: List[Dict[str, float]] = []
        all_lognorm_trials: List[Dict[str, float]] = []
        
        # Save the full subset as "ALL" sample
        save_sample_to_csv('pitchbook', subset, var_key, "ALL", 0)
        
        n_eff_all = len(subset)
        if is_boolean:
            p_hat_all = float(subset[base_var].mean())
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
            # Use log-space priors
            MU0_LOG = 0.0
            SIGMA0_LOG = 100.0
            mu_all_log, sig_all_log = gaussian_posterior(MU0_LOG, SIGMA0_LOG, n_eff_all, mean_log_all, sd_log_all)
            all_lognorm_trials.append({"mu": mu_all_log, "sigma": sig_all_log})
            all_trials.append({"mu": mu_all, "sigma": sig_all})

        baselines[var_key]["ALL"] = all_trials
        baselines[var_key]["ALL_lognorm"] = all_lognorm_trials
    
    return variables, baselines 

if __name__ == "__main__":
    # Test case
    company_raw = pd.read_csv('Company.csv')
    data_df = pitchbook_load(company_raw)
    conditions = [['IsTechCompany', 0], ['Employees', 3], ['TotalRaised', 3]]
    subset, descriptions = apply_conditions_get_data_subset(data_df, 'IsUSBased', conditions)# Ensure the subset is valid
    if subset is not None and not subset.empty:
        # Compute stats for the boolean variable 'IsUSBased'
        mean, std, se = compute_proportion_boolean(subset, 'IsUSBased')
        print(f"Statistics for 'IsUSBased':")
        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std}")
        print(f"Standard Error: {se}")
    else:
        print("No valid subset found for the given conditions.")