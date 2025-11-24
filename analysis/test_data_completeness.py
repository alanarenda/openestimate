"""
Test suite to ensure data completeness in analysis results.

This module tests that:
1. No missing (NaN) z-scores for LLM models
2. No missing (NaN) error ratios for LLM models
3. No missing (NaN) std_ratios for LLM models
4. No missing (NaN) std values for LLM models
5. All required fields are present
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_combined_results(dataset):
    """Load the combined processed results for a dataset."""
    openestimate_root = os.environ.get('OPENESTIMATE_ROOT')
    if not openestimate_root:
        raise ValueError("OPENESTIMATE_ROOT environment variable must be set")

    results_path = Path(openestimate_root) / 'experiments' / dataset / f'{dataset}_combined_processed_results.csv'

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    return pd.read_csv(results_path)


def test_no_missing_z_scores(results, dataset):
    """Test that all LLM models have valid z-scores (no NaN values)."""
    # Filter to LLM models only (exclude statistical baselines)
    llm_results = results[~results['approach'].str.contains('statistical', case=False, na=False)]

    # Check if z-score column exists, if not compute it
    if 'z' not in llm_results.columns:
        llm_results = llm_results.copy()
        # Check for missing prerequisites
        missing_abs_error = llm_results['abs_error_from_mean'].isna().sum()
        missing_std = llm_results['std'].isna().sum()

        if missing_abs_error > 0 or missing_std > 0:
            print(f"\n❌ FAILED: Cannot compute z-scores for {dataset}")
            print(f"   Missing abs_error_from_mean: {missing_abs_error}")
            print(f"   Missing std: {missing_std}")
            return False

        llm_results['z'] = llm_results['abs_error_from_mean'] / llm_results['std']

    # Find rows with missing z-scores
    missing_z = llm_results[llm_results['z'].isna()]

    if len(missing_z) > 0:
        print(f"\n❌ FAILED: {dataset} has {len(missing_z)} missing z-scores")
        print(f"   Total LLM rows: {len(llm_results)}")
        print(f"   Missing z-scores: {len(missing_z)} ({len(missing_z)/len(llm_results)*100:.1f}%)")
        print("\n   Missing z-scores by approach:")
        print(missing_z.groupby('approach').size())

        # Check why z-scores are missing
        print("\n   Rows with missing z-scores (first 10):")
        print(missing_z[['approach', 'variable', 'abs_error_from_mean', 'std', 'z']].head(10))

        # Check for zero or NaN std values
        zero_std = llm_results[llm_results['std'] == 0]
        nan_std = llm_results[llm_results['std'].isna()]
        inf_z = llm_results[np.isinf(llm_results['z'])]

        if len(zero_std) > 0:
            print(f"\n   Found {len(zero_std)} rows with std=0 (division by zero)")
        if len(nan_std) > 0:
            print(f"   Found {len(nan_std)} rows with std=NaN")
        if len(inf_z) > 0:
            print(f"   Found {len(inf_z)} rows with infinite z-scores")

        return False
    else:
        print(f"✅ PASSED: {dataset} has no missing z-scores ({len(llm_results)} rows checked)")
        return True


def test_no_missing_error_ratios(results, dataset):
    """Test that all LLM models have valid error ratios (no NaN values)."""
    # Filter to LLM models only (exclude statistical baselines)
    llm_results = results[~results['approach'].str.contains('statistical', case=False, na=False)]

    # Check all three error ratio columns
    error_ratio_cols = ['error_ratio_mean', 'error_ratio_median', 'error_ratio_mode']

    failed = False
    for col in error_ratio_cols:
        if col not in llm_results.columns:
            print(f"\n❌ FAILED: {dataset} is missing column '{col}'")
            failed = True
            continue

        missing = llm_results[llm_results[col].isna()]

        if len(missing) > 0:
            print(f"\n❌ FAILED: {dataset} has {len(missing)} missing {col}")
            print(f"   Total LLM rows: {len(llm_results)}")
            print(f"   Missing {col}: {len(missing)} ({len(missing)/len(llm_results)*100:.1f}%)")
            print(f"\n   Missing {col} by approach:")
            print(missing.groupby('approach').size())

            # Show sample of missing data
            print(f"\n   Rows with missing {col} (first 10):")
            display_cols = ['approach', 'variable', 'abs_error_from_mean', col]
            if f'associated_baseline_error_{col.split("_")[-1]}' in missing.columns:
                display_cols.append(f'associated_baseline_error_{col.split("_")[-1]}')
            print(missing[display_cols].head(10))

            failed = True

    if not failed:
        print(f"✅ PASSED: {dataset} has no missing error ratios ({len(llm_results)} rows checked)")

    return not failed


def test_no_missing_std_ratios(results, dataset):
    """Test that all LLM models have valid std ratios (no NaN values)."""
    # Filter to LLM models only (exclude statistical baselines)
    llm_results = results[~results['approach'].str.contains('statistical', case=False, na=False)]

    if 'std_ratio' not in llm_results.columns:
        print(f"\n❌ FAILED: {dataset} is missing column 'std_ratio'")
        return False

    missing = llm_results[llm_results['std_ratio'].isna()]

    if len(missing) > 0:
        print(f"\n❌ FAILED: {dataset} has {len(missing)} missing std_ratios")
        print(f"   Total LLM rows: {len(llm_results)}")
        print(f"   Missing std_ratios: {len(missing)} ({len(missing)/len(llm_results)*100:.1f}%)")
        print(f"\n   Missing std_ratios by approach:")
        print(missing.groupby('approach').size())

        # Show sample of missing data
        print("\n   Rows with missing std_ratios (first 10):")
        print(missing[['approach', 'variable', 'std', 'std_ratio', 'associated_baseline_std']].head(10))

        return False
    else:
        print(f"✅ PASSED: {dataset} has no missing std_ratios ({len(llm_results)} rows checked)")
        return True


def test_no_missing_std_values(results, dataset):
    """Test that all LLM models have valid std values (no NaN values)."""
    # Filter to LLM models only (exclude statistical baselines)
    llm_results = results[~results['approach'].str.contains('statistical', case=False, na=False)]

    if 'std' not in llm_results.columns:
        print(f"\n❌ FAILED: {dataset} is missing column 'std'")
        return False

    missing = llm_results[llm_results['std'].isna()]

    if len(missing) > 0:
        print(f"\n❌ FAILED: {dataset} has {len(missing)} missing std values")
        print(f"   Total LLM rows: {len(llm_results)}")
        print(f"   Missing std values: {len(missing)} ({len(missing)/len(llm_results)*100:.1f}%)")
        print(f"\n   Missing std values by approach:")
        print(missing.groupby('approach').size())

        # Show sample of missing data
        print("\n   Rows with missing std values (first 10):")
        display_cols = ['approach', 'variable', 'fitted_distribution_type', 'std']
        if 'mu' in missing.columns:
            display_cols.append('mu')
        if 'sigma' in missing.columns:
            display_cols.append('sigma')
        if 'a' in missing.columns:
            display_cols.append('a')
        if 'b' in missing.columns:
            display_cols.append('b')
        print(missing[display_cols].head(10))

        return False
    else:
        print(f"✅ PASSED: {dataset} has no missing std values ({len(llm_results)} rows checked)")
        return True


def test_required_columns_exist(results, dataset):
    """Test that all required columns are present in the results."""
    required_cols = [
        'approach',
        'variable',
        'ground_truth',
        'abs_error_from_mean',
        'abs_error_from_median',
        'abs_error_from_mode',
        'std',
        'error_ratio_mean',
        'error_ratio_median',
        'error_ratio_mode',
        'std_ratio',
        'quartile_of_gt',
        'fitted_distribution_type',
    ]

    missing_cols = [col for col in required_cols if col not in results.columns]

    if len(missing_cols) > 0:
        print(f"\n❌ FAILED: {dataset} is missing required columns: {missing_cols}")
        return False
    else:
        print(f"✅ PASSED: {dataset} has all required columns")
        return True


def test_no_infinite_values(results, dataset):
    """Test that there are no infinite values in key numeric columns."""
    numeric_cols = [
        'abs_error_from_mean',
        'abs_error_from_median',
        'abs_error_from_mode',
        'std',
        'error_ratio_mean',
        'error_ratio_median',
        'error_ratio_mode',
        'std_ratio',
    ]

    llm_results = results[~results['approach'].str.contains('statistical', case=False, na=False)]

    failed = False
    for col in numeric_cols:
        if col not in llm_results.columns:
            continue

        inf_count = np.isinf(llm_results[col]).sum()

        if inf_count > 0:
            print(f"\n❌ FAILED: {dataset} has {inf_count} infinite values in column '{col}'")
            print(f"   Rows with inf values:")
            inf_rows = llm_results[np.isinf(llm_results[col])]
            print(inf_rows[['approach', 'variable', col]].head(10))
            failed = True

    if not failed:
        print(f"✅ PASSED: {dataset} has no infinite values in key columns")

    return not failed


def run_all_tests(datasets):
    """Run all tests for the specified datasets."""
    print("="*80)
    print("RUNNING DATA COMPLETENESS TESTS")
    print("="*80)

    all_passed = True
    results_by_dataset = {}

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Testing dataset: {dataset.upper()}")
        print(f"{'='*80}")

        try:
            results = load_combined_results(dataset)
            results_by_dataset[dataset] = results
            print(f"Loaded {len(results)} rows for {dataset}")

            # Get LLM row count
            llm_count = len(results[~results['approach'].str.contains('statistical', case=False, na=False)])
            baseline_count = len(results[results['approach'].str.contains('statistical', case=False, na=False)])
            print(f"  - LLM rows: {llm_count}")
            print(f"  - Baseline rows: {baseline_count}")

            # Run tests
            test_results = []
            test_results.append(test_required_columns_exist(results, dataset))
            test_results.append(test_no_missing_std_values(results, dataset))
            test_results.append(test_no_missing_z_scores(results, dataset))
            test_results.append(test_no_missing_error_ratios(results, dataset))
            test_results.append(test_no_missing_std_ratios(results, dataset))
            test_results.append(test_no_infinite_values(results, dataset))

            if not all(test_results):
                all_passed = False

        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            all_passed = False
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    if all_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


def main():
    """Main entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Test data completeness in analysis results')
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['glassdoor', 'pitchbook', 'nhanes'],
        help='Datasets to test (default: glassdoor pitchbook nhanes)'
    )

    args = parser.parse_args()

    # Check that OPENESTIMATE_ROOT is set
    if not os.environ.get('OPENESTIMATE_ROOT'):
        print("ERROR: OPENESTIMATE_ROOT environment variable must be set")
        print("Example: export OPENESTIMATE_ROOT=/path/to/openestimate")
        return 1

    return run_all_tests(args.datasets)


if __name__ == '__main__':
    sys.exit(main())
