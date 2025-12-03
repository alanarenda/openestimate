import os
import json
import argparse
from itertools import product
from pathlib import Path


# Project root configuration - automatically detect based on script location
# This script is in openestimate/experiments/, so go up one level
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())


def get_protocol_spec(protocol):
    """Generate protocol spec for a given protocol name."""
    if protocol == "unified":
        return {"individual_elicitation_protocol": "unified"}
    elif protocol == "unified-lognormal-direct":
        return {"individual_elicitation_protocol": "unified-lognormal-direct"}
    elif protocol == "unified-no-lognormal":
        return {"individual_elicitation_protocol": "unified-no-lognormal-direct"}
    elif protocol == "direct":
        return {"individual_elicitation_protocol": "direct"}
    else:
        return {
            "individual_elicitation_protocol": f"{PROJECT_ROOT}/elicitation/prompts/{protocol}.txt"
        }


# Dataset-specific configurations
DATASET_CONFIGS = {
    'glassdoor': {
        'system_prompts': {
            'base': "You are a helpful assistant that can answer questions about the labor market.",
            'superforecaster': "You are a superforecaster that follows the principles outlined by Philip Tetlock to answer questions about the labor market.",
            'conservative': "You are a helpful assistant that can answer questions about the labor market. You tend to be conservative in your estimates."
        }
    },
    'nhanes': {
        'system_prompts': {
            'base': "You are a helpful assistant that can answer questions about human health.",
            'superforecaster': "You are a superforecaster that follows the principles outlined by Philip Tetlock to answer questions about human health.",
            'conservative': "You are a helpful assistant that can answer questions about human health. You tend to be conservative in your estimates."
        }
    },
    'pitchbook': {
        'system_prompts': {
            'base': "You are a helpful assistant.",
            'superforecaster': "You are a superforecaster that follows the principles outlined by Philip Tetlock to answer questions about venture capital.",
            'conservative': "You are a helpful assistant that can answer questions about venture capital. You tend to be conservative in your estimates."
        }
    }
}


def generate_experiment_specs(dataset):
    """Generate experiment specifications with different prompts and temperatures."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset]
    max_tokens = 4096

    # Define different model configurations
    models_part1 = [
        {
            "name": "openai/gpt-4o",
            "temperatures": [0.5]
        },
        {
            "name": "openai/o4-mini",
            "temperatures": ['medium'] 
        },
        {
            "name": "openai/o3-mini",
            "temperatures": ['medium']
        },
        {
            "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "temperatures": [0.5]
        }, 
        {
            "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "temperatures": [0.5]
        }, 
        { 
            "name": "Qwen/Qwen3-235B-A22B-fp8-tput", 
            "temperatures": [0.6], 
        }
    ]
    
    # Use dataset-specific system prompts
    system_prompts = config['system_prompts']
    
    experiment_names = ['model_family_comparison', 'ablations']
    for experiment_name in experiment_names:
        # Create experiment directory if it doesn't exist
        if not os.path.exists(f"{dataset}/{experiment_name}"):
            os.makedirs(f"{dataset}/{experiment_name}")

    # Part 1 Evaluation
    for model, protocol in product(
        models_part1,
        ["unified-lognormal-direct"]
    ):
        prompt_name = "base"
        experiment_name = "model_family_comparison"
        # Determine the shortened model name based on model naming conventions
        if "llama" in model['name'].lower():
            if "8b" in model['name'].lower():
                model_short_name = "meta-llama-3-8b"
            elif "70b" in model['name'].lower():
                model_short_name = "meta-llama-3-70b"
            else: 
                model_short_name = "meta-llama-3"
        else:
            model_short_name = model['name'].split('/')[-1].lower()
        
        # Generate specs for each temperature
        for temp in model['temperatures']:
            # Create experts specification
            if model['name'] == "Qwen/Qwen3-235B-A22B-fp8-tput":
                max_tokens = 8192
            else:
                max_tokens = 4096
            experts_spec = {
                "model_type": model['name'],
                "model_kwargs": {
                    "temperature": temp,
                    "max_tokens": max_tokens
                },
                "system_prompt": system_prompts[prompt_name]
            }
            
            # Construct the filename for the experts spec
            experts_spec_name = f"{dataset}_{model_short_name}_{prompt_name}_temp{temp}_experts.json"
            experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
            with open(experts_spec_path, 'w') as f:
                json.dump(experts_spec, f, indent=2)
            
            # Create experiment specification
            exp_name = f"{model_short_name}_{prompt_name}_{protocol}_temp{temp}"
            protocol_spec = get_protocol_spec(protocol)
            exp_spec = {
                "experiment_name": exp_name,
                "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
                "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
                "protocol_spec": protocol_spec
            }
            exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
            with open(exp_spec_path, 'w') as f:
                json.dump(exp_spec, f, indent=2)
            print(f"Generated specs for {experiment_name}")

    # Ablations
    eval_model = "openai/o4-mini"
    eval_model_short_name = "o4-mini"
    eval_default_temp = 'medium'
    eval_default_protocol = "unified-lognormal-direct"
    eval_default_prompt = "base"
    eval_system_prompts = ["base", "superforecaster", "conservative"]
    eval_protocols = ["unified-lognormal-direct", "quantile", "mean-variance"]
    
    experiment_name = "ablations"
    
    # Temperature ablations
    for temp in ['low', 'medium', 'high']:
        # Create experts specification
        experts_spec = {
            "model_type": eval_model,
            "model_kwargs": {
                "temperature": temp,
                "max_tokens": max_tokens
            },
            "system_prompt": system_prompts[eval_default_prompt]
        }
        experts_spec_name = f"{dataset}_{eval_model_short_name}_{eval_default_prompt}_temp{temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)

        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{eval_model_short_name}_{eval_default_prompt}_{eval_default_protocol}_temp{temp}"
        protocol_spec = get_protocol_spec(eval_default_protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")

    # System prompt ablations
    for prompt_name in eval_system_prompts:
        experts_spec = {
            "model_type": eval_model,
            "model_kwargs": {
                "temperature": eval_default_temp,
                "max_tokens": max_tokens
            },
            "system_prompt": system_prompts[prompt_name]
        }

        experts_spec_name = f"{dataset}_{eval_model_short_name}_{prompt_name}_temp{eval_default_temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{eval_model_short_name}_{prompt_name}_{eval_default_protocol}_temp{eval_default_temp}"
        protocol_spec = get_protocol_spec(eval_default_protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")

    # Protocol ablations
    for protocol in eval_protocols:
        experts_spec = {
            "model_type": eval_model,
            "model_kwargs": {
                "temperature": eval_default_temp,
                "max_tokens": max_tokens
            },
            "system_prompt": system_prompts[eval_default_prompt]
        }
        experts_spec_name = f"{dataset}_{eval_model_short_name}_{eval_default_prompt}_temp{eval_default_temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{eval_model_short_name}_{eval_default_prompt}_{protocol}_temp{eval_default_temp}"
        protocol_spec = get_protocol_spec(protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")

    regular_model = "openai/gpt-4o"
    regular_model_short_name = "gpt-4o"
    regular_default_temp = 0.5
    regular_default_protocol = "unified-lognormal-direct"
    regular_default_prompt = "base"
    regular_temperatures = [0.2, 0.5, 1.0]
    regular_system_prompts = ["base", "superforecaster", "conservative"]
    regular_protocols = ["unified-lognormal-direct", "quantile", "mean-variance"]
    
    experiment_name = "ablations"
    if not os.path.exists(f"{dataset}/{experiment_name}"):
        os.makedirs(f"{dataset}/{experiment_name}")
    
    for temp in regular_temperatures:
        # Create experts specification
        experts_spec = {
            "model_type": regular_model,
            "model_kwargs": {
                "temperature": temp,
                "max_tokens": max_tokens
            },
            "system_prompt": system_prompts[regular_default_prompt]
        }
        experts_spec_name = f"{dataset}_{regular_model_short_name}_{regular_default_prompt}_temp{temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{regular_model_short_name}_{regular_default_prompt}_{regular_default_protocol}_temp{temp}"
        protocol_spec = get_protocol_spec(regular_default_protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")

    for prompt_name in regular_system_prompts:
        experts_spec = {
            "model_type": regular_model,
            "model_kwargs": {
                "temperature": regular_default_temp,
                "max_tokens": max_tokens
            },
            "system_prompt": system_prompts[prompt_name]
        }

        experts_spec_name = f"{dataset}_{regular_model_short_name}_{prompt_name}_temp{regular_default_temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{regular_model_short_name}_{prompt_name}_{regular_default_protocol}_temp{regular_default_temp}"
        protocol_spec = get_protocol_spec(regular_default_protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")

    for protocol in regular_protocols:
        experts_spec = {
            "model_type": regular_model,
            "model_kwargs": {
                "temperature": regular_default_temp,
                "max_tokens": max_tokens
            },  
            "system_prompt": system_prompts[regular_default_prompt]
        }
        experts_spec_name = f"{dataset}_{regular_model_short_name}_{regular_default_prompt}_temp{regular_default_temp}_experts.json"
        experts_spec_path = os.path.join(f"{dataset}/{experiment_name}", experts_spec_name)
        with open(experts_spec_path, 'w') as f:
            json.dump(experts_spec, f, indent=2)
        exp_name = f"{regular_model_short_name}_{regular_default_prompt}_{protocol}_temp{regular_default_temp}"
        protocol_spec = get_protocol_spec(protocol)
        exp_spec = {
            "experiment_name": exp_name,
            "experts_spec": f"{PROJECT_ROOT}/experiments/{dataset}/{experiment_name}/{experts_spec_name}",
            "variables": f"{PROJECT_ROOT}/data/variables/{dataset}_variables.json",
            "protocol_spec": protocol_spec
        }
        exp_spec_path = os.path.join(f"{dataset}/{experiment_name}", f"{dataset}-{exp_name}-exp-spec.json")
        with open(exp_spec_path, 'w') as f:
            json.dump(exp_spec, f, indent=2)
        print(f"Generated specs for {experiment_name}")


if __name__ == "__main__":
    generate_experiment_specs("glassdoor")
    generate_experiment_specs("nhanes")
    generate_experiment_specs("pitchbook")
   
