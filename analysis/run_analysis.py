import os
import argparse
from compare_models import compare_models
from ablations import run_ablations


def parse_args():
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument(
        '--datasets', 
        type=str, 
        nargs='+',  # Accept one or more strings as a list
        required=True, 
        help='Datasets to analyze (e.g., glassdoor, pitchbook, nhanes)',
    )   
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory', default='analysis_results')
    return parser.parse_args()


def main(datasets, output_dir): 
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    # compare_models(datasets, output_dir)
    run_ablations(datasets, output_dir)
    print("Analysis completed for datasets:", datasets)


if __name__ == "__main__":
    args = parse_args()
    datasets = args.datasets[0].split(',')
    datasets = [ds.strip() for ds in datasets]
    main(datasets, args.output_dir)