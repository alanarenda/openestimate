import os
import glob
import json
import argparse
from datetime import datetime


def generate_run_script(dataset, output_dir='results'):
    """Generate a shell script to run all experiment specs in parallel."""
    num_trials = 5

    exp_names = ['model_family_comparison', 'ablations']
    
    for exp_name in exp_names:
        # Create the script content
        script_lines = []
        for trial in range(num_trials):
            results_dir = f"{dataset}/{exp_name}/trial_{trial}_{output_dir}"
            
            # Find all experiment spec files matching models with their correct temperatures
            exp_specs = glob.glob(f"{dataset}/{exp_name}/*exp-spec.json")
            total_exps = len(exp_specs)
            print(f"Found {total_exps} experiment specs")
            print(exp_specs)
            
            # Create results directory
            script_lines.append(f'mkdir -p {results_dir}')
            
            # Create results and logs directories
            script_lines.append(f'mkdir -p {results_dir}/logs')
            
            # Run experiments in parallel with error logging
            script_lines.append('# Run experiments in parallel')
            for i, spec in enumerate(exp_specs, 1):
                # Create a log file name based on the spec name
                log_base = os.path.basename(spec).replace('exp-spec.json', 'error.log')
                log_file = f'{results_dir}/logs/{log_base}'
                wait_command = " && wait" if ("llama" in spec.lower() or "deepseek" in spec.lower()) else ""
                script_lines.append(f'''(
                    echo "[{i}/{total_exps}] Running {spec}..." && 
                    python3 ~/openestimate/elicitation/src/main.py --experiment_config ~/openestimate/experiments/{spec} --output_dir {results_dir} 2> {log_file} && 
                    echo "[{i}/{total_exps}] Completed {spec}" || 
                    echo "[{i}/{total_exps}] Failed {spec} - see {log_file} for details"{wait_command}
                ) &'''.replace('\n', ' '))
                
                # Add a wait after every 20 commands to limit parallelism
                if i % 20 == 0:
                    script_lines.append('wait')

            # Add final wait to ensure all processes complete
            script_lines.append('wait')

        # Write the script file
        with open(f"{dataset}/{exp_name}/run_experiments_generated.sh", 'w') as f:
            f.write('\n'.join(script_lines))
        
        # Make the script executable
        os.chmod(f"{dataset}/{exp_name}/run_experiments_generated.sh", 0o755)
        
        print(f"Generated run script with {total_exps} experiments")
        print(f"Results will be saved to: {results_dir}")


if __name__ == "__main__":
    generate_run_script("glassdoor")
    generate_run_script("nhanes")
    generate_run_script("pitchbook")
