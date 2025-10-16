from utils import load_experiment_results, print_completion_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dimension_impact_comprehensive(
    results_sets,
    baseline_config_reasoning,
    baseline_config_regular,
    dimension='protocol',
    dimension_values=None,
    dataset_names=['NHANES', 'Pitchbook', 'Glassdoor'],
    save_path=None,
    figsize=(15, 3.5)
):
    """
    Create comprehensive dimension impact visualization with error ratio, ECE, and uncertainty ratio.

    This is a generic function that generates a 6-panel heatmap showing:
    - Error Ratio (reasoning & non-reasoning)
    - ECE (reasoning & non-reasoning)
    - Uncertainty Ratio (reasoning & non-reasoning)

    For any dimension variation (protocol, temperature, or sysprompt).

    Args:
        results_sets: List of 3 DataFrames [nhanes_results, pitchbook_results, glassdoor_results]
        baseline_config_reasoning: Dict with baseline for reasoning models
                                   e.g., {'model': 'o4-mini', 'sysprompt': 'base', 'protocol': 'direct', 'temperature': 'medium'}
        baseline_config_regular: Dict with baseline for regular models
                                 e.g., {'model': 'gpt-4o', 'sysprompt': 'base', 'protocol': 'direct', 'temperature': 0.5}
        dimension: String indicating which dimension to analyze ('protocol', 'temperature', or 'sysprompt')
        dimension_values: List of values to display for this dimension (if None, auto-detected)
        dataset_names: List of dataset display names
        save_path: Optional path to save figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """

    # Map dimension names to column names
    dimension_to_column = {
        'protocol': 'elicitation_protocol',
        'temperature': 'temperature',
        'sysprompt': 'sysprompt_type'
    }

    # Map dimension names to config keys
    dimension_to_config_key = {
        'protocol': 'protocol',
        'temperature': 'temperature',
        'sysprompt': 'sysprompt'
    }

    dimension_col = dimension_to_column[dimension]
    dimension_config_key = dimension_to_config_key[dimension]

    # Mapping for temperature values to labels
    temp_to_label = {
        0.2: 'low',
        0.5: 'medium',
        1.0: 'high'
    }

    # Auto-detect dimension values if not provided
    if dimension_values is None:
        if dimension == 'protocol':
            dimension_values = ['direct', 'quantile', 'mean-variance']
        elif dimension == 'temperature':
            # For temperature, we want to show: low, medium, high
            dimension_values = ['low', 'medium', 'high']
        elif dimension == 'sysprompt':
            # Get unique sysprompts from the data
            all_prompts = set()
            for df in results_sets:
                all_prompts.update(df[dimension_col].dropna().unique())
            dimension_values = sorted(list(all_prompts))

    # ============= Helper Functions =============

    def compute_error_ratios_by_dimension(results, baseline_config):
        """Compute error ratios for dimension variations."""
        df = results.copy()

        baseline_mask = (
            (df['model'] == baseline_config['model']) &
            (df['sysprompt_type'] == baseline_config['sysprompt']) &
            (df['temperature'] == baseline_config['temperature']) &
            (df['elicitation_protocol'] == baseline_config['protocol'])
        )

        baseline_data = df[baseline_mask]
        if baseline_data.empty:
            print(f"Warning: No baseline data found for {baseline_config}")
            print(f"Available models: {df['model'].unique()}")
            print(f"Available temperatures: {df['temperature'].unique()}")
            return pd.DataFrame(columns=[dimension, 'ratio_mean'])

        baseline_mae_by_var = baseline_data.groupby('variable')['abs_error'].mean()

        # Create mask for all other dimensions (excluding the one we're varying)
        other_dims = ['model', 'sysprompt_type', 'temperature', 'elicitation_protocol']
        other_dims.remove(dimension_col)

        dim_mask = (df['model'] == baseline_config['model'])
        if 'sysprompt_type' in other_dims:
            dim_mask &= (df['sysprompt_type'] == baseline_config['sysprompt'])
        if 'temperature' in other_dims:
            dim_mask &= (df['temperature'] == baseline_config['temperature'])
        if 'elicitation_protocol' in other_dims:
            dim_mask &= (df['elicitation_protocol'] == baseline_config['protocol'])

        subset = df[dim_mask]
        if subset.empty:
            return pd.DataFrame(columns=[dimension, 'ratio_mean'])

        dim_results = []
        for value in subset[dimension_col].unique():
            value_data = subset[subset[dimension_col] == value]
            value_mae_by_var = value_data.groupby('variable')['abs_error'].mean()
            ratio_by_var = value_mae_by_var / baseline_mae_by_var.reindex(value_mae_by_var.index)
            valid_ratios = ratio_by_var.dropna().values

            if len(valid_ratios) > 0:
                # Map temperature values to labels if we're analyzing temperature
                display_value = temp_to_label.get(value, value) if dimension == 'temperature' else value
                dim_results.append({
                    dimension: display_value,
                    'ratio_mean': np.nanmean(valid_ratios)
                })

        return pd.DataFrame(dim_results)

    def compute_ece_for_dimension(results, baseline_config):
        """Compute ECE for dimension variations."""
        df = results.copy()

        # Create mask for all other dimensions
        other_dims = ['model', 'sysprompt_type', 'temperature', 'elicitation_protocol']
        other_dims.remove(dimension_col)

        dim_mask = (df['model'] == baseline_config['model'])
        if 'sysprompt_type' in other_dims:
            dim_mask &= (df['sysprompt_type'] == baseline_config['sysprompt'])
        if 'temperature' in other_dims:
            dim_mask &= (df['temperature'] == baseline_config['temperature'])
        if 'elicitation_protocol' in other_dims:
            dim_mask &= (df['elicitation_protocol'] == baseline_config['protocol'])

        subset = df[dim_mask]
        if subset.empty:
            return pd.DataFrame(columns=[dimension, 'ece_normalized'])

        # Get baseline ECE
        baseline_data = subset[subset[dimension_col] == baseline_config[dimension_config_key]]
        if baseline_data.empty:
            return pd.DataFrame(columns=[dimension, 'ece_normalized'])

        quartile_counts = baseline_data['quartile_of_gt'].value_counts(normalize=True) * 100
        baseline_ece = sum(abs(quartile_counts.get(q, 0) - 25.0) for q in [1, 2, 3, 4]) / 4.0

        dim_results = []
        for value in subset[dimension_col].unique():
            value_data = subset[subset[dimension_col] == value]
            quartile_counts = value_data['quartile_of_gt'].value_counts(normalize=True) * 100
            ece = sum(abs(quartile_counts.get(q, 0) - 25.0) for q in [1, 2, 3, 4]) / 4.0

            # Map temperature values to labels if we're analyzing temperature
            display_value = temp_to_label.get(value, value) if dimension == 'temperature' else value
            dim_results.append({
                dimension: display_value,
                'ece_normalized': ece / baseline_ece if baseline_ece > 0 else 1.0
            })

        return pd.DataFrame(dim_results)

    def compute_std_ratios_by_dimension(results, baseline_config):
        """Compute std ratios for dimension variations."""
        df = results.copy()

        baseline_mask = (
            (df['model'] == baseline_config['model']) &
            (df['sysprompt_type'] == baseline_config['sysprompt']) &
            (df['temperature'] == baseline_config['temperature']) &
            (df['elicitation_protocol'] == baseline_config['protocol'])
        )

        baseline_data = df[baseline_mask]
        if baseline_data.empty:
            print(f"Warning: No baseline data found for {baseline_config}")
            print(f"Available models: {df['model'].unique()}")
            print(f"Available temperatures: {df['temperature'].unique()}")
            return pd.DataFrame(columns=[dimension, 'ratio_std'])

        baseline_std_by_var = baseline_data.groupby('variable')['std'].mean()

        # Create mask for all other dimensions
        other_dims = ['model', 'sysprompt_type', 'temperature', 'elicitation_protocol']
        other_dims.remove(dimension_col)

        dim_mask = (df['model'] == baseline_config['model'])
        if 'sysprompt_type' in other_dims:
            dim_mask &= (df['sysprompt_type'] == baseline_config['sysprompt'])
        if 'temperature' in other_dims:
            dim_mask &= (df['temperature'] == baseline_config['temperature'])
        if 'elicitation_protocol' in other_dims:
            dim_mask &= (df['elicitation_protocol'] == baseline_config['protocol'])

        subset = df[dim_mask]
        if subset.empty:
            return pd.DataFrame(columns=[dimension, 'ratio_std'])

        dim_results = []
        for value in subset[dimension_col].unique():
            value_data = subset[subset[dimension_col] == value]
            value_std_by_var = value_data.groupby('variable')['std'].mean()
            ratio_by_var = value_std_by_var / baseline_std_by_var.reindex(value_std_by_var.index)
            valid_ratios = ratio_by_var.dropna().values

            if len(valid_ratios) > 0:
                # Map temperature values to labels if we're analyzing temperature
                display_value = temp_to_label.get(value, value) if dimension == 'temperature' else value
                dim_results.append({
                    dimension: display_value,
                    'ratio_std': np.nanmean(valid_ratios)
                })

        return pd.DataFrame(dim_results)


    def create_heatmap_data(compute_func, results_sets, baseline_config, value_col):
        """Create heatmap DataFrame from results."""
        heatmap_data = []

        for results in results_sets:
            dim_results = compute_func(results, baseline_config)
            row_data = []
            for value in dimension_values:
                # Check if dim_results has data and the required column
                if not dim_results.empty and dimension in dim_results.columns:
                    matching = dim_results[dim_results[dimension] == value]
                    if not matching.empty and value_col in matching.columns:
                        row_data.append(matching[value_col].iloc[0])
                    else:
                        row_data.append(np.nan)
                else:
                    row_data.append(np.nan)
            heatmap_data.append(row_data)

        return pd.DataFrame(heatmap_data, index=dataset_names, columns=dimension_values)

    # ============= Compute All Metrics =============

    # Error ratios
    error_r = create_heatmap_data(compute_error_ratios_by_dimension, results_sets,
                                   baseline_config_reasoning, 'ratio_mean')
    error_nr = create_heatmap_data(compute_error_ratios_by_dimension, results_sets,
                                    baseline_config_regular, 'ratio_mean')

    # Std ratios
    std_r = create_heatmap_data(compute_std_ratios_by_dimension, results_sets,
                                   baseline_config_reasoning, 'ratio_std')
    std_nr = create_heatmap_data(compute_std_ratios_by_dimension, results_sets,
                                    baseline_config_regular, 'ratio_std')

    # ECE
    ece_r = create_heatmap_data(compute_ece_for_dimension, results_sets,
                                 baseline_config_reasoning, 'ece_normalized')
    ece_nr = create_heatmap_data(compute_ece_for_dimension, results_sets,
                                  baseline_config_regular, 'ece_normalized')

    print(f"Error Ratios (Reasoning) for {dimension}:")
    print(error_r)
    print(f"\nError Ratios (Non-Reasoning) for {dimension}:")
    print(error_nr)
    print(f"\nECE Ratios (Reasoning) for {dimension}:")
    print(ece_r)
    print(f"\nECE Ratios (Non-Reasoning) for {dimension}:")
    print(ece_nr)
    print(f"\nStd Ratios (Reasoning) for {dimension}:")
    print(std_r)
    print(f"\nStd Ratios (Non-Reasoning) for {dimension}:")
    print(std_nr)

    # ============= Compute Color Scales =============

    # Error ratio and std use same scale
    all_vals = pd.concat([error_r.stack(), error_nr.stack(),
                          std_r.stack(), std_nr.stack()]).dropna()
    vmin, vmax = (all_vals.min(), all_vals.max()) if len(all_vals) > 0 else (0, 1)

    # ECE uses separate scale
    all_ece_vals = pd.concat([ece_r.stack(), ece_nr.stack()]).dropna()
    vmin_ece, vmax_ece = (all_ece_vals.min(), all_ece_vals.max()) if len(all_ece_vals) > 0 else (0, 1)

    monochrome = plt.cm.Blues

    # ============= Drawing Functions =============

    # Determine if we should highlight the baseline (first column)
    highlight_baseline = dimension == 'protocol'  # Only highlight for protocol

    def draw(ax, df, title, xlabel, show_y=False, show_xlabel=True, highlight_first=False):
        """Draw heatmap with formatting."""
        sns.heatmap(df, ax=ax, cmap=monochrome, vmin=vmin, vmax=vmax,
                    annot=True, fmt='.3f', cbar=False, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel if show_xlabel else '')

        if show_y:
            ax.set_ylabel('Dataset')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        if highlight_first:
            ax.plot([0, 0], [0, len(df)], color='black', linewidth=3)
            ax.plot([1, 1], [0, len(df)], color='black', linewidth=2)
            ax.plot([0, 1], [0, 0], color='black', linewidth=3)
            ax.plot([0, 1], [len(df), len(df)], color='black', linewidth=3)

    def draw_ece(ax, df, title, xlabel, show_y=False, show_xlabel=True, highlight_first=False):
        """Draw ECE heatmap."""
        sns.heatmap(df, ax=ax, cmap=monochrome, vmin=vmin_ece, vmax=vmax_ece,
                    annot=True, fmt='.1f', cbar=False, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel if show_xlabel else '')

        if show_y:
            ax.set_ylabel('Dataset')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        if highlight_first:
            ax.plot([0, 0], [0, len(df)], color='black', linewidth=3)
            ax.plot([1, 1], [0, len(df)], color='black', linewidth=2)
            ax.plot([0, 1], [0, 0], color='black', linewidth=3)
            ax.plot([0, 1], [len(df), len(df)], color='black', linewidth=3)

    # ============= Create Plot =============

    fig, axes = plt.subplots(1, 6, figsize=figsize)

    draw(axes[0], error_r, 'Reasoning', '', show_y=True, show_xlabel=False, highlight_first=highlight_baseline)
    draw(axes[1], error_nr, 'Non-Reasoning', '', show_xlabel=False, highlight_first=highlight_baseline)
    draw_ece(axes[2], ece_r, 'Reasoning', '', show_xlabel=False, highlight_first=highlight_baseline)
    draw_ece(axes[3], ece_nr, 'Non-Reasoning', '', show_xlabel=False, highlight_first=highlight_baseline)
    draw(axes[4], std_r, 'Reasoning', '', show_xlabel=False, highlight_first=highlight_baseline)
    draw(axes[5], std_nr, 'Non-Reasoning', '', show_xlabel=False, highlight_first=highlight_baseline)

    # Add main title and section labels
    dimension_title = dimension.replace('_', ' ').title()
    fig.suptitle(f'Impact of {dimension_title} on Performance',
                 fontsize=14, y=0.98)

    fig.text(0.23, 0.85, 'Error Ratio', ha='center', fontsize=12)
    fig.text(0.53, 0.85, 'ECE', ha='center', fontsize=12)
    fig.text(0.85, 0.85, 'Uncertainty Ratio', ha='center', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.70)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{dimension_title} impact plot saved to: {save_path}")
    plt.show()
    return fig


def run_ablations(datasets, output_dir):
    experiment_name = 'ablations'
    results_sets = [load_experiment_results(dataset, experiment_name) for dataset in datasets]
    for results in results_sets:
        print_completion_stats(results)

    # Define baselines (use numeric temperature to match data)
    baseline_reasoning = {
        'model': 'o4-mini',
        'sysprompt': 'base',
        'protocol': 'direct',
        'temperature': 'medium'
    }

    baseline_regular = {
        'model': 'gpt-4o',
        'sysprompt': 'base',
        'protocol': 'direct',
        'temperature': 0.5
    }

    dataset_names = ['NHANES', 'Pitchbook', 'Glassdoor']

    # Generate protocol impact plot
    print("\n" + "="*80)
    print("PROTOCOL ABLATION")
    print("="*80 + "\n")
    fig_protocol = plot_dimension_impact_comprehensive(
        results_sets,
        baseline_reasoning,
        baseline_regular,
        dimension='protocol',
        dataset_names=dataset_names,
        save_path=f'{output_dir}/protocol_impact.png'
    )
    plt.close()

    # Generate temperature impact plot
    print("\n" + "="*80)
    print("TEMPERATURE ABLATION")
    print("="*80 + "\n")
    fig_temp = plot_dimension_impact_comprehensive(
        results_sets,
        baseline_reasoning,
        baseline_regular,
        dimension='temperature',
        dataset_names=dataset_names,
        save_path=f'{output_dir}/temperature_impact.png'
    )
    print("Wrote temperature impact plot.")
    plt.close()

    # Generate sysprompt impact plot
    print("\n" + "="*80)
    print("SYSPROMPT ABLATION")
    print("="*80 + "\n")
    fig_sysprompt = plot_dimension_impact_comprehensive(
        results_sets,
        baseline_reasoning,
        baseline_regular,
        dimension='sysprompt',
        dataset_names=dataset_names,
        save_path=f'{output_dir}/sysprompt_impact.png'
    )
    plt.close()

    print("\n" + "="*80)
    print("All ablation plots generated successfully!")
    print("="*80)
