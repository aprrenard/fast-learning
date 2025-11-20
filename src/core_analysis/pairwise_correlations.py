import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
# from src.utils.utils_behavior import *  # Not needed for this script


# #############################################################################
# Pairwise correlations between cells during 2 sec quiet windows (all trials)
# #############################################################################
#
# This script computes pairwise correlations between neurons during correct
# rejection trials and performs statistical analysis.
#
# Key features:
# - Per-trial correlation computation (avoids drift artifacts)
# - Parallelized computation across mice
# - Separate computation and analysis modes (set ANALYSIS_MODE below)
# - Two-level analysis: cell-pair level and mouse-average level
# - Day-by-day analysis (days -2, -1, 0, +1, +2)
# - Pre-post comparison (pre: days -2/-1, post: days +1/+2)
# - Comprehensive statistical testing with Mann-Whitney U tests
# - Automatic annotation of ALL significant comparisons on plots (sorted by span)
#
# Correlation method:
# - For each cell pair, correlations are computed within each trial
# - These per-trial correlations are then averaged across trials
# - This approach prevents drift between trials from inflating correlations
#
# Outputs (per reward group R+/R-):
# Data files:
# - pairwise_correlations.csv: Raw correlation data
# - pairwise_correlations_mouse_averages.csv: Mouse-level averages
# - statistical_tests.csv: Pair-level day-by-day statistical tests
# - statistical_tests_prepost.csv: Pair-level pre-post statistical tests
# - statistical_tests_mouse_level.csv: Mouse-level day-by-day statistical test
# - statistical_tests_mouse_level_prepost.csv: Mouse-level pre-post statistical tests
#
# Plots:
# - Plot 1: Pair-level correlations by day (5 panels, one per day)
# - Plot 2: Pair-level correlations across days (4 panels, one per pair type)
# - Plot 3: Mouse-level correlations by day (5 panels, one per day)
# - Plot 4: Mouse-level correlations across days (4 panels, one per pair type)
# - Plot 5: Pair-level pre vs post comparison (4 panels, one per pair type)
# - Plot 5b: Pair-level comparison of pair types within pre and post (2 panels)
# - Plot 6: Mouse-level pre vs post comparison (4 panels, one per pair type)
# - Plot 6b: Mouse-level comparison of pair types within pre and post (2 panels)
#
# #############################################################################

# Parameters
# ----------
sampling_rate = 30
win_sec = (2, 0)  # Use quiet window: 2s before to 0s before every stimulus
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']
N_CORES = 50  # Number of cores for parallel processing

# Analysis mode: 'compute' to compute correlations, 'analyze' to load and analyze existing data
# Set to 'analyze' after first run to skip time-consuming correlation computation
ANALYSIS_MODE = 'analyze'  # Options: 'compute' or 'analyze'

# Debug mode: Set to True to run diagnostic checks for coordinate-data mismatches
DEBUG_MODE = False  # Set to False to skip diagnostics

# Select sessions from database
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

# Remove problematic mice
# mice = [m for m in mice if m != 'GF305']


# Separate mice by reward group
mice_by_group = {}
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    if reward_group not in mice_by_group:
        mice_by_group[reward_group] = []
    mice_by_group[reward_group].append(mouse_id)


# Define function to process a single mouse (for parallel processing)
def process_mouse(mouse_id):
    """
    Process correlations for a single mouse across all days.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier

    Returns
    -------
    list of dict
        List of correlation results for this mouse
    """
    print(f"\nProcessing mouse: {mouse_id}")

    mouse_results = []

    # Get reward group for this mouse
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Load xarray data for this mouse
    file_name = 'tensor_xarray_learning_quietwindow.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr.name = 'dff'

    xarr = xarr.sel(trial=xarr['day'].isin(days))

    # Process each day separately
    for day in days:
        print(f"  Mouse {mouse_id}: Processing day {day}...")

        # Select trials for this day
        xarr_day = xarr.sel(trial=xarr['day'] == day)

        # Pre-extract all cell data at once to avoid repeated xarray indexing (optimization)
        # Shape: (n_cells, n_trials, n_timepoints)
        all_cells_data = xarr_day.values

        # Get actual dimensions from the extracted data shape
        n_cells = all_cells_data.shape[0]
        n_trials = all_cells_data.shape[1]
        n_timepoints = all_cells_data.shape[2]

        if n_trials == 0:
            print(f"    Mouse {mouse_id}: No trials for day {day}, skipping")
            continue

        print(f"    Mouse {mouse_id}: Found {n_cells} cells, {n_trials} trials (shape: {all_cells_data.shape})")

        # Get cell type information for this day's data
        cell_types = xarr_day.coords['cell_type'].values
        rois = xarr_day.coords['roi'].values

        # For each pair of cells, compute correlation per trial and average
        # This avoids artificial correlations due to drift between trials
        for i, j in combinations(range(n_cells), 2):
            # Get activity traces for both cells using fast numpy indexing
            # Shape: (n_trials, n_timepoints)
            cell_i_data = all_cells_data[i, :, :]
            cell_j_data = all_cells_data[j, :, :]

            # Compute correlation for each trial separately
            trial_correlations = []
            for trial_idx in range(n_trials):
                cell_i_trial = cell_i_data[trial_idx, :]
                cell_j_trial = cell_j_data[trial_idx, :]

                # Remove any NaN values (if present)
                valid_idx = ~(np.isnan(cell_i_trial) | np.isnan(cell_j_trial))
                cell_i_clean = cell_i_trial[valid_idx]
                cell_j_clean = cell_j_trial[valid_idx]

                # Compute Pearson correlation for this trial
                # Check for sufficient data and non-constant arrays
                if len(cell_i_clean) > 1:  # Need at least 2 points for correlation
                    # Check if both arrays have variance (not constant)
                    if np.std(cell_i_clean) > 0 and np.std(cell_j_clean) > 0:
                        corr, _ = pearsonr(cell_i_clean, cell_j_clean)
                        trial_correlations.append(corr)
                    # If either array is constant, correlation is undefined - skip this trial

            # Average correlation across trials
            if len(trial_correlations) > 0:
                final_corr = np.mean(trial_correlations)

                # Store result
                mouse_results.append({
                    'mouse_id': mouse_id,
                    'reward_group': reward_group,
                    'day': day,
                    'cell_i': i,
                    'cell_j': j,
                    'roi_i': rois[i],
                    'roi_j': rois[j],
                    'cell_type_i': cell_types[i],
                    'cell_type_j': cell_types[j],
                    'correlation': final_corr,
                    'n_trials': len(trial_correlations)  # Track how many trials contributed
                })

    print(f"  Mouse {mouse_id}: Completed! Computed {len(mouse_results)} correlations")
    return mouse_results


# #############################################################################
# DEBUGGING FUNCTIONS
# #############################################################################

def check_coordinate_data_alignment(mice_list, days_to_check):
    """
    Diagnostic function to check for mismatches between xarray coordinates
    and actual numpy data shapes.

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to check
    days_to_check : list
        List of days to check

    Returns
    -------
    pandas.DataFrame
        DataFrame with mismatch information
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking coordinate-data alignment")
    print("="*80)

    mismatches = []

    for mouse_id in mice_list:
        print(f"\nChecking mouse: {mouse_id}")

        try:
            # Load data
            file_name = 'tensor_xarray_learning_quietwindow.nc'
            folder = os.path.join(io.processed_dir, 'mice')
            xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
            xarr = xarr.sel(trial=xarr['day'].isin(days_to_check))

            reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

            # Check each day
            for day in days_to_check:
                xarr_day = xarr.sel(trial=xarr['day'] == day)

                # Get coordinate lengths
                coord_n_trials = len(xarr_day.trial)
                coord_n_cells = len(xarr_day.cell)
                coord_n_rois = len(xarr_day.coords['roi'].values)

                # Get actual data shape
                data_shape = xarr_day.values.shape
                data_n_trials = data_shape[1] if len(data_shape) > 0 else 0
                data_n_timepoints = data_shape[2] if len(data_shape) > 1 else 0
                data_n_cells = data_shape[0] if len(data_shape) > 2 else 0

                # Check for mismatches
                trial_mismatch = coord_n_trials != data_n_trials
                cell_mismatch = coord_n_cells != data_n_cells
                roi_mismatch = coord_n_rois != data_n_cells

                if trial_mismatch or cell_mismatch or roi_mismatch:
                    mismatch_info = {
                        'mouse_id': mouse_id,
                        'reward_group': reward_group,
                        'day': day,
                        'coord_n_trials': coord_n_trials,
                        'data_n_trials': data_n_trials,
                        'trial_mismatch': trial_mismatch,
                        'coord_n_cells': coord_n_cells,
                        'coord_n_rois': coord_n_rois,
                        'data_n_cells': data_n_cells,
                        'cell_mismatch': cell_mismatch,
                        'roi_mismatch': roi_mismatch,
                        'data_shape': str(data_shape)
                    }
                    mismatches.append(mismatch_info)

                    print(f"  ⚠️  MISMATCH on day {day}:")
                    if trial_mismatch:
                        print(f"      Trial dimension: coord={coord_n_trials}, data={data_n_trials}")
                    if cell_mismatch:
                        print(f"      Cell dimension: coord={coord_n_cells}, data={data_n_cells}")
                    if roi_mismatch:
                        print(f"      ROI coordinate: len={coord_n_rois}, data_cells={data_n_cells}")
                    print(f"      Data shape: {data_shape}")
                else:
                    print(f"  ✓ Day {day}: OK (trials={data_n_trials}, cells={data_n_cells})")

        except Exception as e:
            print(f"  ❌ Error checking mouse {mouse_id}: {e}")
            mismatches.append({
                'mouse_id': mouse_id,
                'reward_group': 'unknown',
                'day': 'all',
                'error': str(e)
            })

    # Create summary
    if mismatches:
        df_mismatches = pd.DataFrame(mismatches)
        print("\n" + "="*80)
        print(f"SUMMARY: Found {len(mismatches)} mismatch(es)")
        print("="*80)
        print(df_mismatches.to_string())
        return df_mismatches
    else:
        print("\n" + "="*80)
        print("SUMMARY: No mismatches found! ✓")
        print("="*80)
        return pd.DataFrame()


# Helper function to classify cell pair types
def classify_pair_type(row):
    """Classify the type of cell pair"""
    ct_i = row['cell_type_i']
    ct_j = row['cell_type_j']

    # Sort to make wS2-wM1 and wM1-wS2 the same
    pair = tuple(sorted([ct_i, ct_j]))

    if pair == ('wS2', 'wS2'):
        return 'wS2-wS2'
    elif pair == ('wM1', 'wM1'):
        return 'wM1-wM1'
    elif pair == ('wM1', 'wS2'):
        return 'wS2-wM1'
    else:
        return 'other'


# Main execution: Process mice in parallel
if __name__ == '__main__':
    # Run diagnostic checks if DEBUG_MODE is enabled
    if DEBUG_MODE:
        print("\n" + "="*80)
        print("DEBUG MODE ENABLED - Running diagnostics")
        print("="*80)

        # Get all mice across both reward groups
        all_mice = []
        for rg in ['R-', 'R+']:
            if rg in mice_by_group:
                all_mice.extend(mice_by_group[rg])

        # Run diagnostic check
        mismatch_df = check_coordinate_data_alignment(all_mice, days)

        # Save diagnostic results if any mismatches found
        if not mismatch_df.empty:
            debug_output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/pairwise_correlations/debug'
            debug_output_dir = io.adjust_path_to_host(debug_output_dir)
            os.makedirs(debug_output_dir, exist_ok=True)

            debug_csv_path = os.path.join(debug_output_dir, 'coordinate_data_mismatches.csv')
            mismatch_df.to_csv(debug_csv_path, index=False)
            print(f"\n💾 Saved mismatch report to: {debug_csv_path}")

        print("\n" + "="*80)
        print("DEBUG MODE - Diagnostics complete")
        print("="*80)

        # Ask user if they want to continue
        user_input = input("\nContinue with analysis? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            print("Exiting...")
            sys.exit(0)

    # Loop through each reward group
    for reward_group in ['R-', 'R+']:
        if reward_group not in mice_by_group:
            print(f"\nNo mice found for reward group {reward_group}, skipping...")
            continue

        group_mice = mice_by_group[reward_group]
        print(f"\n{'='*80}")
        print(f"PROCESSING REWARD GROUP: {reward_group}")
        print(f"{'='*80}")

        # Set up output directory
        output_dir = f'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/pairwise_correlations/{reward_group}'
        output_dir = io.adjust_path_to_host(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Check if we should compute or load existing data
        corr_csv_path = os.path.join(output_dir, 'pairwise_correlations.csv')

        if ANALYSIS_MODE == 'compute' or not os.path.exists(corr_csv_path):
            print(f"\nMode: COMPUTE - Computing pairwise correlations for {len(group_mice)} mice using {N_CORES} cores...")
            print(f"Mice in {reward_group}: {group_mice}")

            # Use multiprocessing to parallelize across mice
            try:
                with Pool(processes=N_CORES) as pool:
                    results_list = pool.map(process_mouse, group_mice)

                # Flatten the list of lists into a single list
                correlation_results = [item for sublist in results_list for item in sublist]

                print(f"\nCorrelation computation complete for {reward_group}!")

            except Exception as e:
                print(f"\n❌ ERROR during correlation computation: {e}")
                print(f"Attempting to save any partial results before exiting...")

                # Check if we have any partial results
                if 'results_list' in locals() and results_list:
                    correlation_results = [item for sublist in results_list for item in sublist if sublist]
                    print(f"Found {len(correlation_results)} partial correlation results from successful mice")
                else:
                    print("No partial results available to save")
                    raise  # Re-raise the exception to exit

            # Convert results to DataFrame
            corr_df = pd.DataFrame(correlation_results)

            # Check if we got any results
            if len(corr_df) == 0:
                print("⚠️  WARNING: No correlation results computed! Check for errors above.")
            else:
                print(f"✓ Successfully computed correlations for {corr_df['mouse_id'].nunique()} mice")

            # IMPORTANT: Save raw results immediately as backup before any further processing
            raw_backup_path = os.path.join(output_dir, 'pairwise_correlations_raw_backup.csv')
            corr_df.to_csv(raw_backup_path, index=False)
            print(f"💾 Saved raw backup to: {raw_backup_path}")

            # Add a column for the pair type (all-all, wS2-wS2, wM1-wM1, wS2-wM1)
            corr_df['pair_type'] = corr_df.apply(classify_pair_type, axis=1)

            # Also keep track of all-all pairs
            corr_df_all = corr_df.copy()
            corr_df_all['pair_type'] = 'all-all'

            # Combine
            corr_df_combined = pd.concat([corr_df_all, corr_df], ignore_index=True)

            # Filter to only keep the pair types we're interested in
            pair_types_of_interest = ['all-all', 'wS2-wS2', 'wM1-wM1', 'wS2-wM1']
            corr_df_filtered = corr_df_combined[corr_df_combined['pair_type'].isin(pair_types_of_interest)]

            print(f"\nTotal correlation pairs computed: {len(corr_df)}")
            print(f"Breakdown by pair type:")
            print(corr_df['pair_type'].value_counts())

            # Save correlation data
            corr_df_filtered.to_csv(corr_csv_path, index=False)
            print(f"\nSaved correlation data to: {corr_csv_path}")

        else:
            print(f"\nMode: ANALYZE - Loading existing correlation data from {corr_csv_path}")

            # Try to load main file, fall back to backup if needed
            try:
                corr_df_filtered = pd.read_csv(corr_csv_path)
                pair_types_of_interest = ['all-all', 'wS2-wS2', 'wM1-wM1', 'wS2-wM1']
                print(f"✓ Loaded {len(corr_df_filtered)} correlation pairs")
                print(f"Breakdown by pair type:")
                print(corr_df_filtered['pair_type'].value_counts())
            except Exception as e:
                print(f"⚠️  Could not load main file: {e}")
                raw_backup_path = os.path.join(output_dir, 'pairwise_correlations_raw_backup.csv')
                if os.path.exists(raw_backup_path):
                    print(f"Attempting to load from backup: {raw_backup_path}")
                    corr_df = pd.read_csv(raw_backup_path)

                    # Recreate the filtered dataframe from backup
                    corr_df_all = corr_df.copy()
                    corr_df_all['pair_type'] = 'all-all'
                    corr_df_combined = pd.concat([corr_df_all, corr_df], ignore_index=True)
                    pair_types_of_interest = ['all-all', 'wS2-wS2', 'wM1-wM1', 'wS2-wM1']
                    corr_df_filtered = corr_df_combined[corr_df_combined['pair_type'].isin(pair_types_of_interest)]

                    print(f"✓ Loaded {len(corr_df_filtered)} correlation pairs from backup")
                else:
                    print(f"❌ No backup file found. Please recompute.")
                    raise


        # #############################################################################
        # Pre-Post data preparation
        # #############################################################################

        print("\n" + "="*80)
        print("PREPARING PRE-POST COMPARISON DATA")
        print("="*80)

        # Add period column: "pre" for days -2/-1, "post" for days +1/+2
        # Exclude day 0 from pre-post comparison
        def assign_period(day):
            if day in [-2, -1]:
                return 'pre'
            elif day in [1, 2]:
                return 'post'
            else:
                return None  # day 0 excluded

        corr_df_filtered['period'] = corr_df_filtered['day'].apply(assign_period)

        # Create pre-post filtered dataframe (excluding day 0)
        corr_df_prepost = corr_df_filtered[corr_df_filtered['period'].notna()].copy()

        print(f"\nPre-post data summary:")
        print(f"  Pre (days -2, -1): {len(corr_df_prepost[corr_df_prepost['period'] == 'pre'])} correlation pairs")
        print(f"  Post (days +1, +2): {len(corr_df_prepost[corr_df_prepost['period'] == 'post'])} correlation pairs")
        print(f"  Day 0 excluded: {len(corr_df_filtered[corr_df_filtered['day'] == 0])} correlation pairs")


        # #############################################################################
        # Statistical tests (compute before plotting)
        # #############################################################################

        from scipy.stats import ttest_ind, mannwhitneyu
        
        print("\n" + "="*80)
        print("STATISTICAL TESTS")
        print("="*80)
        
        # Test 1: Compare correlation between different pair types for each day
        print("\n1. Comparing different pair types within each day (Mann-Whitney U test)")
        print("-" * 80)
        
        stat_results = []
        
        for day in days:
            print(f"\nDay {day}:")
            day_data = corr_df_filtered[corr_df_filtered['day'] == day]
        
            # Compare each pair of pair types
            for i, pair_type_1 in enumerate(pair_types_of_interest):
                for pair_type_2 in pair_types_of_interest[i+1:]:
                    data_1 = day_data[day_data['pair_type'] == pair_type_1]['correlation'].values
                    data_2 = day_data[day_data['pair_type'] == pair_type_2]['correlation'].values
        
                    if len(data_1) > 0 and len(data_2) > 0:
                        stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')
        
                        mean_1 = np.mean(data_1)
                        mean_2 = np.mean(data_2)
        
                        print(f"  {pair_type_1} (μ={mean_1:.3f}) vs {pair_type_2} (μ={mean_2:.3f}): "
                              f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
                        stat_results.append({
                            'test_type': 'pair_type_comparison',
                            'day': day,
                            'pair_type_1': pair_type_1,
                            'pair_type_2': pair_type_2,
                            'mean_1': mean_1,
                            'mean_2': mean_2,
                            'stat': stat,
                            'p_value': p_value
                        })
        
        # Test 2: Compare correlation across days for each pair type
        print("\n\n2. Comparing across days for each pair type (Mann-Whitney U test)")
        print("-" * 80)

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            # Compare each pair of days
            for i, day_1 in enumerate(days):
                for day_2 in days[i+1:]:
                    data_1 = pair_data[pair_data['day'] == day_1]['correlation'].values
                    data_2 = pair_data[pair_data['day'] == day_2]['correlation'].values

                    if len(data_1) > 1 and len(data_2) > 1:  # Need at least 2 samples
                        try:
                            # Check if data is not identical
                            if np.array_equal(data_1, data_2):
                                print(f"  Day {day_1} vs Day {day_2}: Identical data, skipping")
                                continue

                            stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')

                            # Check if p_value is NaN
                            if np.isnan(p_value):
                                print(f"  Day {day_1} vs Day {day_2}: NaN p-value, likely identical distributions")
                                continue

                            mean_1 = np.mean(data_1)
                            mean_2 = np.mean(data_2)

                            print(f"  Day {day_1} (μ={mean_1:.3f}, n={len(data_1)}) vs Day {day_2} (μ={mean_2:.3f}, n={len(data_2)}): "
                                  f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                            stat_results.append({
                                'test_type': 'day_comparison',
                                'pair_type': pair_type,
                                'day_1': day_1,
                                'day_2': day_2,
                                'mean_1': mean_1,
                                'mean_2': mean_2,
                                'stat': stat,
                                'p_value': p_value
                            })
                        except Exception as e:
                            print(f"  Day {day_1} vs Day {day_2}: Error - {str(e)}")
                    else:
                        print(f"  Day {day_1} vs Day {day_2}: Insufficient data (n1={len(data_1)}, n2={len(data_2)})")
        
        # Save statistical results
        stat_df = pd.DataFrame(stat_results)
        stat_df.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
        print(f"\n\nSaved statistical test results to: {os.path.join(output_dir, 'statistical_tests.csv')}")


        # #############################################################################
        # Pre-Post Statistical Tests (pair-level)
        # #############################################################################

        print("\n" + "="*80)
        print("PRE-POST STATISTICAL TESTS (PAIR-LEVEL)")
        print("="*80)

        prepost_stat_results = []

        # Test: Compare pre vs post for each pair type
        print("\nComparing pre vs post for each pair type (Mann-Whitney U test, n=cell pairs)")
        print("-" * 80)

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_prepost_data = corr_df_prepost[corr_df_prepost['pair_type'] == pair_type]

            data_pre = pair_prepost_data[pair_prepost_data['period'] == 'pre']['correlation'].values
            data_post = pair_prepost_data[pair_prepost_data['period'] == 'post']['correlation'].values

            if len(data_pre) > 0 and len(data_post) > 0:
                stat, p_value = mannwhitneyu(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)

                print(f"  Pre (μ={mean_pre:.3f}, n={len(data_pre)}) vs Post (μ={mean_post:.3f}, n={len(data_post)}): "
                      f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_stat_results.append({
                    'test_type': 'prepost_comparison',
                    'pair_type': pair_type,
                    'period_1': 'pre',
                    'period_2': 'post',
                    'mean_pre': mean_pre,
                    'mean_post': mean_post,
                    'n_pre': len(data_pre),
                    'n_post': len(data_post),
                    'stat': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient data (n_pre={len(data_pre)}, n_post={len(data_post)})")

        # Save pre-post statistical results
        prepost_stat_df = pd.DataFrame(prepost_stat_results)
        prepost_stat_df.to_csv(os.path.join(output_dir, 'statistical_tests_prepost.csv'), index=False)
        print(f"\n\nSaved pre-post statistical test results to: {os.path.join(output_dir, 'statistical_tests_prepost.csv')}")


        # #############################################################################
        # Plots with statistical annotations
        # #############################################################################

        print("\nGenerating plots with statistical annotations...")

        # Define colors for each pair type
        pair_palette = {
            'all-all': '#7f7f7f',
            'wS2-wS2': '#d62728',
            'wM1-wM1': '#1f77b4',
            'wS2-wM1': '#ff7f0e'
        }

        # Helper function to add significance stars
        def add_significance_stars(ax, x1, x2, y, p_value):
            """Add significance stars between two bars"""
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            else:
                return  # Don't add anything if not significant

            # Draw line
            h = y
            line_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.plot([x1, x1, x2, x2], [h, h + line_height, h + line_height, h], lw=1.5, c='black')
            # Add stars
            ax.text((x1 + x2) / 2, h + line_height, stars, ha='center', va='bottom', fontsize=12, fontweight='bold')

        def get_all_significant_comparisons_sorted(stats_df, pair_types, data_df, value_col='correlation'):
            """
            Get all significant comparisons sorted by the span between indices.
            This ensures we plot shorter comparisons first (to avoid overlaps).

            Returns: list of tuples (pair_type_1, pair_type_2, idx1, idx2, p_value)
            """
            comparisons = []
            for i, pt1 in enumerate(pair_types):
                for j, pt2 in enumerate(pair_types[i+1:], start=i+1):
                    comp_stat = stats_df[((stats_df['pair_type_1'] == pt1) & (stats_df['pair_type_2'] == pt2)) |
                                         ((stats_df['pair_type_1'] == pt2) & (stats_df['pair_type_2'] == pt1))]

                    if not comp_stat.empty:
                        p_val = comp_stat.iloc[0]['p_value']
                        if p_val < 0.05:
                            comparisons.append((pt1, pt2, i, j, p_val, abs(j - i)))

            # Sort by span (shorter spans first), then by p-value
            comparisons.sort(key=lambda x: (x[5], x[4]))
            return [(c[0], c[1], c[2], c[3], c[4]) for c in comparisons]

        def get_all_significant_day_comparisons_sorted(stats_df, days_list):
            """
            Get all significant day comparisons sorted by the span between days.

            Returns: list of tuples (day_1, day_2, idx1, idx2, p_value)
            """
            comparisons = []
            for i, d1 in enumerate(days_list):
                for j, d2 in enumerate(days_list[i+1:], start=i+1):
                    comp_stat = stats_df[((stats_df['day_1'] == d1) & (stats_df['day_2'] == d2)) |
                                         ((stats_df['day_1'] == d2) & (stats_df['day_2'] == d1))]

                    if not comp_stat.empty:
                        p_val = comp_stat.iloc[0]['p_value']
                        if p_val < 0.05:
                            comparisons.append((d1, d2, i, j, p_val, abs(j - i)))

            # Sort by span (shorter spans first), then by p-value
            comparisons.sort(key=lambda x: (x[5], x[4]))
            return [(c[0], c[1], c[2], c[3], c[4]) for c in comparisons]

        # Plot 1: Average correlation for each day with significance
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

        for idx, day in enumerate(days):
            ax = axes[idx]

            # Get data for this day
            day_data = corr_df_filtered[corr_df_filtered['day'] == day]

            # Plot bar plot
            sns.barplot(
                data=day_data,
                x='pair_type',
                y='correlation',
                order=pair_types_of_interest,
                palette=pair_palette,
                ax=ax,
                errorbar='ci',
                capsize=0.1,
                n_boot=100
            )

            ax.set_title(f'Day {day}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cell Pair Type', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations after all subplots are created (so ylim is set)
        for idx, day in enumerate(days):
            ax = axes[idx]
            day_data = corr_df_filtered[corr_df_filtered['day'] == day]

            # Get all significant comparisons for this day
            day_stats = stat_df[(stat_df['test_type'] == 'pair_type_comparison') &
                                (stat_df['day'] == day) &
                                (stat_df['p_value'] < 0.05)]

            if not day_stats.empty:
                # Get all significant comparisons sorted by span
                comparisons = get_all_significant_comparisons_sorted(day_stats, pair_types_of_interest, day_data)

                # Add all significant comparisons with proper offset
                for offset_level, (pair1, pair2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get the maximum y value across both bars
                    y_vals = []
                    for pair_type in [pair1, pair2]:
                        pair_data = day_data[day_data['pair_type'] == pair_type]['correlation']
                        if len(pair_data) > 0:
                            y_vals.append(pair_data.mean() + 1.96 * pair_data.sem())  # mean + CI
                    if y_vals:
                        y_max = max(y_vals)
                        # Add offset for multiple comparisons
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Average Pairwise Correlation During 2 sec quiet windows (all trials) ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_by_day.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()



        # Plot 2: Correlation across days
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_data,
                x='day',
                y='correlation',
                order=days,
                color=pair_palette[pair_type],
                ax=ax,
                errorbar='ci',
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations for day comparisons
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            # Get significant day comparisons for this pair type
            pair_stats = stat_df[(stat_df['test_type'] == 'day_comparison') &
                                 (stat_df['pair_type'] == pair_type) &
                                 (stat_df['p_value'] < 0.05)]

            if not pair_stats.empty:
                # Get all significant day comparisons sorted by span
                comparisons = get_all_significant_day_comparisons_sorted(pair_stats, days)

                # Add all significant comparisons with proper offset
                for offset_level, (day_1, day_2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get max y values for these days
                    y_vals = []
                    for day in [day_1, day_2]:
                        day_corr = pair_data[pair_data['day'] == day]['correlation']
                        if len(day_corr) > 0:
                            y_vals.append(day_corr.mean() + 1.96 * day_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Pairwise Correlation Across Days During 2 sec quiet windows (all trials) ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_across_days.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # Plot 5: Pre-Post comparison (pair-level)
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_prepost_data = corr_df_prepost[corr_df_prepost['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_prepost_data,
                x='period',
                y='correlation',
                order=['pre', 'post'],
                color=pair_palette[pair_type],
                ax=ax,
                errorbar='ci',
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_prepost_data = corr_df_prepost[corr_df_prepost['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_stats = prepost_stat_df[prepost_stat_df['pair_type'] == pair_type]

            if not pair_prepost_stats.empty:
                p_val = pair_prepost_stats.iloc[0]['p_value']
                if p_val < 0.05:
                    # Get max y values for pre and post
                    y_vals = []
                    for period in ['pre', 'post']:
                        period_corr = pair_prepost_data[pair_prepost_data['period'] == period]['correlation']
                        if len(period_corr) > 0:
                            y_vals.append(period_corr.mean() + 1.96 * period_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06
                        add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)

        plt.suptitle(f'Pre vs Post Pairwise Correlation During 2 sec quiet windows (all trials) ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # #############################################################################
        # Plot 5b: Pair-level comparison of pair types within pre and post separately
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING PLOT 5B: PAIR TYPE COMPARISON WITHIN PRE AND POST PERIODS")
        print("="*80)

        # Statistical tests for pair type comparisons within pre and post
        period_pairtype_stat_results = []

        for period in ['pre', 'post']:
            print(f"\n{period.upper()} period:")
            period_data = corr_df_prepost[corr_df_prepost['period'] == period]

            # Compare each pair of pair types
            for i, pair_type_1 in enumerate(pair_types_of_interest):
                for pair_type_2 in pair_types_of_interest[i+1:]:
                    data_1 = period_data[period_data['pair_type'] == pair_type_1]['correlation'].values
                    data_2 = period_data[period_data['pair_type'] == pair_type_2]['correlation'].values

                    if len(data_1) > 0 and len(data_2) > 0:
                        stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')

                        mean_1 = np.mean(data_1)
                        mean_2 = np.mean(data_2)

                        print(f"  {pair_type_1} (μ={mean_1:.3f}, n={len(data_1)}) vs {pair_type_2} (μ={mean_2:.3f}, n={len(data_2)}): "
                              f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                        period_pairtype_stat_results.append({
                            'test_type': 'pair_type_comparison_within_period',
                            'period': period,
                            'pair_type_1': pair_type_1,
                            'pair_type_2': pair_type_2,
                            'mean_1': mean_1,
                            'mean_2': mean_2,
                            'n_1': len(data_1),
                            'n_2': len(data_2),
                            'stat': stat,
                            'p_value': p_value
                        })

        # Save statistical results
        period_pairtype_stat_df = pd.DataFrame(period_pairtype_stat_results)
        period_pairtype_stat_df.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_pair_type_comparisons.csv'), index=False)
        print(f"\nSaved pair type comparison stats to: {os.path.join(output_dir, 'statistical_tests_prepost_pair_type_comparisons.csv')}")

        # Create plot 5b
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]

            # Get data for this period
            period_data = corr_df_prepost[corr_df_prepost['period'] == period]

            # Plot bar plot
            sns.barplot(
                data=period_data,
                x='pair_type',
                y='correlation',
                order=pair_types_of_interest,
                palette=pair_palette,
                ax=ax,
                errorbar='ci',
                capsize=0.1,
                n_boot=100
            )

            ax.set_title(f'{period.upper()} Period', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cell Pair Type', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]
            period_data = corr_df_prepost[corr_df_prepost['period'] == period]

            # Get significant comparisons for this period
            period_stats = period_pairtype_stat_df[(period_pairtype_stat_df['period'] == period) &
                                                    (period_pairtype_stat_df['p_value'] < 0.05)]

            if not period_stats.empty:
                # Get all significant comparisons sorted by span
                comparisons = get_all_significant_comparisons_sorted(period_stats, pair_types_of_interest, period_data)

                # Add all significant comparisons with proper offset
                for offset_level, (pair1, pair2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get the maximum y value across both bars
                    y_vals = []
                    for pair_type in [pair1, pair2]:
                        pair_data = period_data[period_data['pair_type'] == pair_type]['correlation']
                        if len(pair_data) > 0:
                            y_vals.append(pair_data.mean() + 1.96 * pair_data.sem())
                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Pair Type Comparison Within Pre and Post Periods ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost_pair_type_comparison.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # #############################################################################
        # Mouse-level analysis (average correlation per mouse)
        # #############################################################################

        print("\n" + "="*80)
        print("MOUSE-LEVEL STATISTICAL ANALYSIS")
        print("="*80)

        # Compute average correlation per mouse for each day and pair type
        mouse_avg_corr = corr_df_filtered.groupby(['mouse_id', 'reward_group', 'day', 'pair_type'])['correlation'].mean().reset_index()
        mouse_avg_corr.rename(columns={'correlation': 'mean_correlation'}, inplace=True)

        print(f"\nComputed average correlations for {len(mouse_avg_corr['mouse_id'].unique())} mice")
        print(f"Sample size per group:")
        for pt in pair_types_of_interest:
            for d in days:
                n_mice = len(mouse_avg_corr[(mouse_avg_corr['pair_type'] == pt) & (mouse_avg_corr['day'] == d)])
                print(f"  {pt}, Day {d}: {n_mice} mice")

        # Save mouse-level averages
        mouse_avg_corr.to_csv(os.path.join(output_dir, 'pairwise_correlations_mouse_averages.csv'), index=False)
        print(f"\nSaved mouse-level averages to: {os.path.join(output_dir, 'pairwise_correlations_mouse_averages.csv')}")

        # Prepare mouse-level pre-post data
        mouse_avg_corr['period'] = mouse_avg_corr['day'].apply(assign_period)
        mouse_avg_corr_prepost = mouse_avg_corr[mouse_avg_corr['period'].notna()].copy()

        print(f"\nMouse-level pre-post data summary:")
        print(f"  Pre (days -2, -1): {len(mouse_avg_corr_prepost[mouse_avg_corr_prepost['period'] == 'pre'])} mouse-day pairs")
        print(f"  Post (days +1, +2): {len(mouse_avg_corr_prepost[mouse_avg_corr_prepost['period'] == 'post'])} mouse-day pairs")

        # Statistical tests on mouse-level data
        print("\n" + "="*80)
        print("STATISTICAL TESTS (MOUSE-LEVEL)")
        print("="*80)

        mouse_stat_results = []

        # Test 1: Compare correlation between different pair types for each day (across mice)
        print("\n1. Comparing different pair types within each day (Mann-Whitney U test, n=mice)")
        print("-" * 80)

        for day in days:
            print(f"\nDay {day}:")
            day_mouse_data = mouse_avg_corr[mouse_avg_corr['day'] == day]

            # Compare each pair of pair types
            for i, pair_type_1 in enumerate(pair_types_of_interest):
                for pair_type_2 in pair_types_of_interest[i+1:]:
                    data_1 = day_mouse_data[day_mouse_data['pair_type'] == pair_type_1]['mean_correlation'].values
                    data_2 = day_mouse_data[day_mouse_data['pair_type'] == pair_type_2]['mean_correlation'].values

                    if len(data_1) > 0 and len(data_2) > 0:
                        stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')

                        mean_1 = np.mean(data_1)
                        mean_2 = np.mean(data_2)

                        print(f"  {pair_type_1} (μ={mean_1:.3f}, n={len(data_1)} mice) vs {pair_type_2} (μ={mean_2:.3f}, n={len(data_2)} mice): "
                              f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                        mouse_stat_results.append({
                            'test_type': 'pair_type_comparison',
                            'day': day,
                            'pair_type_1': pair_type_1,
                            'pair_type_2': pair_type_2,
                            'mean_1': mean_1,
                            'mean_2': mean_2,
                            'n_mice_1': len(data_1),
                            'n_mice_2': len(data_2),
                            'stat': stat,
                            'p_value': p_value
                        })

        # Test 2: Compare correlation across days for each pair type (across mice)
        print("\n\n2. Comparing across days for each pair type (Mann-Whitney U test, n=mice)")
        print("-" * 80)

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Compare each pair of days
            for i, day_1 in enumerate(days):
                for day_2 in days[i+1:]:
                    data_1 = pair_mouse_data[pair_mouse_data['day'] == day_1]['mean_correlation'].values
                    data_2 = pair_mouse_data[pair_mouse_data['day'] == day_2]['mean_correlation'].values

                    if len(data_1) > 1 and len(data_2) > 1:  # Need at least 2 samples
                        try:
                            # Check if data is not identical
                            if np.array_equal(data_1, data_2):
                                print(f"  Day {day_1} vs Day {day_2}: Identical data, skipping")
                                continue

                            stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')

                            # Check if p_value is NaN
                            if np.isnan(p_value):
                                print(f"  Day {day_1} vs Day {day_2}: NaN p-value, likely identical distributions")
                                continue

                            mean_1 = np.mean(data_1)
                            mean_2 = np.mean(data_2)

                            print(f"  Day {day_1} (μ={mean_1:.3f}, n={len(data_1)} mice) vs Day {day_2} (μ={mean_2:.3f}, n={len(data_2)} mice): "
                                  f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                            mouse_stat_results.append({
                                'test_type': 'day_comparison',
                                'pair_type': pair_type,
                                'day_1': day_1,
                                'day_2': day_2,
                                'mean_1': mean_1,
                                'mean_2': mean_2,
                                'n_mice_1': len(data_1),
                                'n_mice_2': len(data_2),
                                'stat': stat,
                                'p_value': p_value
                            })
                        except Exception as e:
                            print(f"  Day {day_1} vs Day {day_2}: Error - {str(e)}")
                    else:
                        print(f"  Day {day_1} vs Day {day_2}: Insufficient data (n1={len(data_1)}, n2={len(data_2)} mice)")

        # Save mouse-level statistical results
        mouse_stat_df = pd.DataFrame(mouse_stat_results)
        mouse_stat_df.to_csv(os.path.join(output_dir, 'statistical_tests_mouse_level.csv'), index=False)
        print(f"\n\nSaved mouse-level statistical test results to: {os.path.join(output_dir, 'statistical_tests_mouse_level.csv')}")


        # #############################################################################
        # Pre-Post Statistical Tests (mouse-level)
        # #############################################################################

        print("\n" + "="*80)
        print("PRE-POST STATISTICAL TESTS (MOUSE-LEVEL)")
        print("="*80)

        prepost_mouse_stat_results = []

        # Test: Compare pre vs post for each pair type (across mice)
        print("\nComparing pre vs post for each pair type (Mann-Whitney U test, n=mice)")
        print("-" * 80)

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_prepost_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['pair_type'] == pair_type]

            data_pre = pair_prepost_mouse_data[pair_prepost_mouse_data['period'] == 'pre']['mean_correlation'].values
            data_post = pair_prepost_mouse_data[pair_prepost_mouse_data['period'] == 'post']['mean_correlation'].values

            if len(data_pre) > 0 and len(data_post) > 0:
                stat, p_value = mannwhitneyu(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)

                print(f"  Pre (μ={mean_pre:.3f}, n={len(data_pre)} mice) vs Post (μ={mean_post:.3f}, n={len(data_post)} mice): "
                      f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_mouse_stat_results.append({
                    'test_type': 'prepost_comparison',
                    'pair_type': pair_type,
                    'period_1': 'pre',
                    'period_2': 'post',
                    'mean_pre': mean_pre,
                    'mean_post': mean_post,
                    'n_mice_pre': len(data_pre),
                    'n_mice_post': len(data_post),
                    'stat': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient data (n_pre={len(data_pre)}, n_post={len(data_post)} mice)")

        # Save mouse-level pre-post statistical results
        prepost_mouse_stat_df = pd.DataFrame(prepost_mouse_stat_results)
        prepost_mouse_stat_df.to_csv(os.path.join(output_dir, 'statistical_tests_mouse_level_prepost.csv'), index=False)
        print(f"\n\nSaved mouse-level pre-post statistical test results to: {os.path.join(output_dir, 'statistical_tests_mouse_level_prepost.csv')}")


        # #############################################################################
        # Mouse-level plots with statistical annotations
        # #############################################################################

        print("\nGenerating mouse-level plots with statistical annotations...")

        # Plot 3: Mouse-level average correlation for each day with significance
        fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

        for idx, day in enumerate(days):
            ax = axes[idx]

            # Get data for this day
            day_mouse_data = mouse_avg_corr[mouse_avg_corr['day'] == day]

            # Plot bar plot
            sns.barplot(
                data=day_mouse_data,
                x='pair_type',
                y='mean_correlation',
                order=pair_types_of_interest,
                palette=pair_palette,
                ax=ax,
                errorbar='se',  # Standard error across mice
                capsize=0.1
            )

            ax.set_title(f'Day {day}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cell Pair Type', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations after all subplots are created
        for idx, day in enumerate(days):
            ax = axes[idx]
            day_mouse_data = mouse_avg_corr[mouse_avg_corr['day'] == day]

            # Get all significant comparisons for this day
            day_mouse_stats = mouse_stat_df[(mouse_stat_df['test_type'] == 'pair_type_comparison') &
                                            (mouse_stat_df['day'] == day) &
                                            (mouse_stat_df['p_value'] < 0.05)]

            if not day_mouse_stats.empty:
                # Get all significant comparisons sorted by span
                comparisons = get_all_significant_comparisons_sorted(day_mouse_stats, pair_types_of_interest, day_mouse_data, value_col='mean_correlation')

                # Add all significant comparisons with proper offset
                for offset_level, (pair1, pair2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get the maximum y value across both bars
                    y_vals = []
                    for pair_type in [pair1, pair2]:
                        pair_data = day_mouse_data[day_mouse_data['pair_type'] == pair_type]['mean_correlation']
                        if len(pair_data) > 0:
                            y_vals.append(pair_data.mean() + pair_data.sem())  # mean + SE
                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Average Pairwise Correlation During 2 sec quiet windows (all trials) - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_by_day_mouse_level.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # Plot 4: Mouse-level correlation across days
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_mouse_data,
                x='day',
                y='mean_correlation',
                order=days,
                color=pair_palette[pair_type],
                ax=ax,
                errorbar='se',  # Standard error across mice
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations for day comparisons (mouse-level)
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Get significant day comparisons for this pair type
            pair_mouse_stats = mouse_stat_df[(mouse_stat_df['test_type'] == 'day_comparison') &
                                             (mouse_stat_df['pair_type'] == pair_type) &
                                             (mouse_stat_df['p_value'] < 0.05)]

            if not pair_mouse_stats.empty:
                # Get all significant day comparisons sorted by span
                comparisons = get_all_significant_day_comparisons_sorted(pair_mouse_stats, days)

                # Add all significant comparisons with proper offset
                for offset_level, (day_1, day_2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get max y values for these days
                    y_vals = []
                    for day in [day_1, day_2]:
                        day_corr = pair_mouse_data[pair_mouse_data['day'] == day]['mean_correlation']
                        if len(day_corr) > 0:
                            y_vals.append(day_corr.mean() + day_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Pairwise Correlation Across Days During 2 sec quiet windows (all trials) - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_across_days_mouse_level.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # Plot 6: Pre-Post comparison (mouse-level)
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_prepost_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['pair_type'] == pair_type]

            # Plot bar plot
            sns.barplot(
                data=pair_prepost_mouse_data,
                x='period',
                y='mean_correlation',
                order=['pre', 'post'],
                color=pair_palette[pair_type],
                ax=ax,
                errorbar='se',  # Standard error across mice
                capsize=0.1
            )

            ax.set_title(f'{pair_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]
            pair_prepost_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_mouse_stats = prepost_mouse_stat_df[prepost_mouse_stat_df['pair_type'] == pair_type]

            if not pair_prepost_mouse_stats.empty:
                p_val = pair_prepost_mouse_stats.iloc[0]['p_value']
                if p_val < 0.05:
                    # Get max y values for pre and post
                    y_vals = []
                    for period in ['pre', 'post']:
                        period_corr = pair_prepost_mouse_data[pair_prepost_mouse_data['period'] == period]['mean_correlation']
                        if len(period_corr) > 0:
                            y_vals.append(period_corr.mean() + period_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06
                        add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)

        plt.suptitle(f'Pre vs Post Pairwise Correlation During 2 sec quiet windows (all trials) - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost_mouse_level.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()


        # #############################################################################
        # Plot 6b: Mouse-level comparison of pair types within pre and post separately
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING PLOT 6B: MOUSE-LEVEL PAIR TYPE COMPARISON WITHIN PRE AND POST PERIODS")
        print("="*80)

        # Statistical tests for pair type comparisons within pre and post (mouse-level)
        period_pairtype_mouse_stat_results = []

        for period in ['pre', 'post']:
            print(f"\n{period.upper()} period (mouse-level):")
            period_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['period'] == period]

            # Compare each pair of pair types
            for i, pair_type_1 in enumerate(pair_types_of_interest):
                for pair_type_2 in pair_types_of_interest[i+1:]:
                    data_1 = period_mouse_data[period_mouse_data['pair_type'] == pair_type_1]['mean_correlation'].values
                    data_2 = period_mouse_data[period_mouse_data['pair_type'] == pair_type_2]['mean_correlation'].values

                    if len(data_1) > 0 and len(data_2) > 0:
                        stat, p_value = mannwhitneyu(data_1, data_2, alternative='two-sided')

                        mean_1 = np.mean(data_1)
                        mean_2 = np.mean(data_2)

                        print(f"  {pair_type_1} (μ={mean_1:.3f}, n={len(data_1)} mice) vs {pair_type_2} (μ={mean_2:.3f}, n={len(data_2)} mice): "
                              f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                        period_pairtype_mouse_stat_results.append({
                            'test_type': 'pair_type_comparison_within_period',
                            'period': period,
                            'pair_type_1': pair_type_1,
                            'pair_type_2': pair_type_2,
                            'mean_1': mean_1,
                            'mean_2': mean_2,
                            'n_mice_1': len(data_1),
                            'n_mice_2': len(data_2),
                            'stat': stat,
                            'p_value': p_value
                        })

        # Save statistical results
        period_pairtype_mouse_stat_df = pd.DataFrame(period_pairtype_mouse_stat_results)
        period_pairtype_mouse_stat_df.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_pair_type_comparisons_mouse_level.csv'), index=False)
        print(f"\nSaved mouse-level pair type comparison stats to: {os.path.join(output_dir, 'statistical_tests_prepost_pair_type_comparisons_mouse_level.csv')}")

        # Create plot 6b
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]

            # Get data for this period
            period_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['period'] == period]

            # Plot bar plot
            sns.barplot(
                data=period_mouse_data,
                x='pair_type',
                y='mean_correlation',
                order=pair_types_of_interest,
                palette=pair_palette,
                ax=ax,
                errorbar='se',
                capsize=0.1
            )

            ax.set_title(f'{period.upper()} Period', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cell Pair Type', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
            else:
                ax.set_ylabel('')

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add horizontal line at 0
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

        # Add significance annotations
        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]
            period_mouse_data = mouse_avg_corr_prepost[mouse_avg_corr_prepost['period'] == period]

            # Get significant comparisons for this period
            period_mouse_stats = period_pairtype_mouse_stat_df[(period_pairtype_mouse_stat_df['period'] == period) &
                                                                (period_pairtype_mouse_stat_df['p_value'] < 0.05)]

            if not period_mouse_stats.empty:
                # Get all significant comparisons sorted by span
                comparisons = get_all_significant_comparisons_sorted(period_mouse_stats, pair_types_of_interest, period_mouse_data, value_col='mean_correlation')

                # Add all significant comparisons with proper offset
                for offset_level, (pair1, pair2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get the maximum y value across both bars
                    y_vals = []
                    for pair_type in [pair1, pair2]:
                        pair_data = period_mouse_data[period_mouse_data['pair_type'] == pair_type]['mean_correlation']
                        if len(pair_data) > 0:
                            y_vals.append(pair_data.mean() + pair_data.sem())
                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Pair Type Comparison Within Pre and Post Periods - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_prepost_pair_type_comparison_mouse_level.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        # plt.close()

        print(f"\n{'='*80}")
        print(f"Analysis complete for {reward_group}!")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print("Files generated:")
        print("  - pairwise_correlations.csv")
        print("  - pairwise_correlations_by_day.svg")
        print("  - pairwise_correlations_across_days.svg")
        print("  - pairwise_correlations_prepost.svg")
        print("  - pairwise_correlations_prepost_pair_type_comparison.svg (NEW: Plot 5b)")
        print("  - statistical_tests.csv")
        print("  - statistical_tests_prepost.csv")
        print("  - statistical_tests_prepost_pair_type_comparisons.csv")
        print("  - pairwise_correlations_mouse_averages.csv")
        print("  - pairwise_correlations_by_day_mouse_level.svg")
        print("  - pairwise_correlations_across_days_mouse_level.svg")
        print("  - pairwise_correlations_prepost_mouse_level.svg")
        print("  - pairwise_correlations_prepost_pair_type_comparison_mouse_level.svg (NEW: Plot 6b)")
        print("  - statistical_tests_mouse_level.csv")
        print("  - statistical_tests_mouse_level_prepost.csv")
        print("  - statistical_tests_prepost_pair_type_comparisons_mouse_level.csv")

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")






