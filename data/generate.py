import json 
import click
import numpy as np
from pathlib import Path 
from nhanes_generation import generate_nhanes
from glassdoor import generate_glassdoor 
from pitchbook import generate_pitchbook


GENERATION_CONFIG = {
    "target_num_single_condition_vars": 20, 
    "target_num_double_condition_vars": 20,
    "target_num_triple_condition_vars": 20,
    "difference_threshold": 0.05, 
}

generators = {'nhanes': generate_nhanes, 'glassdoor': generate_glassdoor, 'pitchbook': generate_pitchbook}

def default_serializer(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def generate(dataset: str):
    if dataset == 'all': 
        datasets = ['glassdoor', 'pitchbook', 'nhanes']
    else:
        datasets = [dataset]
    
    for ds in datasets:
        generator = generators.get(ds)
        variables, baselines = generator(GENERATION_CONFIG)
        
        var_output_dir_path = Path(f"variables/{ds}_variables.json")
        baseline_output_dir_path = f"baselines/{ds}_baselines.json"

        # Extract the directory portion of the path
        var_output_dir = var_output_dir_path.parent
        var_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the variables to a JSON file
        with open(var_output_dir_path, 'w') as json_file:
            json.dump(variables, json_file, indent=4, default=default_serializer)

        # Save the baselines to a JSON file
        with open(baseline_output_dir_path, 'w') as json_file:
            json.dump(baselines, json_file, indent=4, default=default_serializer)


if __name__ == '__main__':
    dataset = click.prompt("Please enter the dataset to generate (glassdoor, pitchbook, nhanes, all)", type=click.Choice(['glassdoor', 'pitchbook', 'nhanes', 'all']))
    generate(dataset)
    click.echo("Dataset generation complete!")
    