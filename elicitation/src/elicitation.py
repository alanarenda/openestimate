import os
from utils import (
    replace_placeholders,
    extract_variable_from_response,
    uncapitalize,
)
from clients import get_model_client


def get_protocol_file_path(distribution_type):
    """Get the file path for the appropriate protocol based on distribution type."""
    protocol_mapping = {
        'normal': '~/openestimate/elicitation/prompts/normal-direct.txt',
        'binomial': '~/openestimate/elicitation/prompts/beta-direct.txt',
        'beta': '~/openestimate/elicitation/prompts/beta-direct.txt'
    }
    
    if distribution_type not in protocol_mapping:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return os.path.expanduser(protocol_mapping[distribution_type])


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


def elicit_priors(protocol, variables, experts):
    """
    Elicit priors for a list of variables.
    
    Args:
        protocol: Optional protocol text. If None, uses default for distribution type.
        variables: List of variable dictionaries or dict of variables
        experts: Expert configuration dictionary
        
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
            use_protocol = get_default_protocol(distribution_type)
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