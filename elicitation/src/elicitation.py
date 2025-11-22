import os
from pathlib import Path
from utils import (
    replace_placeholders,
    extract_variable_from_response,
    uncapitalize,
)
from clients import get_model_client


# Project root configuration - automatically detect based on script location
# This script is in openestimate/elicitation/src/, so go up two levels
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())


def get_protocol_file_path(distribution_type):
    """Get the file path for the appropriate protocol based on distribution type."""
    protocol_mapping = {
        'normal': f'{PROJECT_ROOT}/elicitation/prompts/normal-direct.txt',
        'binomial': f'{PROJECT_ROOT}/elicitation/prompts/beta-direct.txt',
        'beta': f'{PROJECT_ROOT}/elicitation/prompts/beta-direct.txt',
        'unified': f'{PROJECT_ROOT}/elicitation/prompts/unified-lognormal-direct.txt',
        'unified-lognormal-direct': f'{PROJECT_ROOT}/elicitation/prompts/unified-lognormal-direct.txt',
        'unified-no-lognormal-direct': f'{PROJECT_ROOT}/elicitation/prompts/unified-no-lognormal-direct.txt',
        'lognormal': f'{PROJECT_ROOT}/elicitation/prompts/unified-direct.txt'
    }

    if distribution_type not in protocol_mapping:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

    return protocol_mapping[distribution_type]


def load_protocol_file(file_path):
    """Load protocol content from file with error handling."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Protocol file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading protocol file {file_path}: {e}")


def prepare_variable_context(variable):
    """Prepare variable context dictionary for protocol template replacement."""
    return {
        'variable': uncapitalize(variable['variable']),
        'variable_description': variable.get('description', ""),
        'units_description': variable.get('units_description', ""), 
        'ground_truth_distribution_type': variable['ground_truth_distribution_type']
    }


def get_default_protocol(distribution_type):
    """Get default protocol text for a given distribution type."""
    protocol_file = get_protocol_file_path(distribution_type)
    protocol_content = load_protocol_file(protocol_file)
    return protocol_content


def elicit_priors(protocol, variables, experts, protocol_name=None):
    """
    Elicit priors for a list of variables.

    Args:
        protocol: Optional protocol text. If None, uses protocol_name or default for distribution type.
        variables: List of variable dictionaries or dict of variables
        experts: Expert configuration dictionary
        protocol_name: Optional protocol name (e.g., 'unified-lognormal', 'direct') to use instead of distribution type

    Returns:
        Dictionary of elicited priors keyed by variable name
    """
    sys_prompt = experts["system_prompt"]
    elicited_priors = {}
    model_client = get_model_client(
        experts["model_type"],
        experts["model_kwargs"]["temperature"],
        experts["model_kwargs"]["max_tokens"]
    )

    # Handle both dict and list inputs
    if isinstance(variables, dict):
        variables = list(variables.values())

    for var in variables:
        var_name = var['variable']
        print(f"Eliciting prior for: {var_name}")

        # Prepare variable context
        all_vars = prepare_variable_context(var)

        # Get protocol text
        distribution_type = var['ground_truth_distribution_type']
        if protocol is None:
            # Use protocol_name if provided, otherwise fall back to distribution_type
            protocol_type = protocol_name if protocol_name else distribution_type
            use_protocol = get_default_protocol(protocol_type)
        else:
            use_protocol = protocol
        
        # Replace placeholders in protocol
        init_protocol = replace_placeholders(use_protocol, all_vars)
  
        # Generate completion
        messages = [{"role": "system", "content": sys_prompt}]
        print(f"Init protocol: {init_protocol}")
        messages.append({"role": "user", "content": init_protocol})
        
        try:
            response = model_client.generate_completion(messages)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error generating completion for {var_name}: {e}")
            raise
        
        # Extract variables from response and store
        all_vars = extract_variable_from_response(response, all_vars)
        messages.append({"role": "assistant", "content": response})
        elicited_priors[var_name] = {
            'var_output': all_vars, 
            'conversation': messages
        }
    
    return elicited_priors