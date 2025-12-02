import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
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
# Pairwise correlations between cells during 2 sec quiet windows
# Pre vs Post comparison
# #############################################################################
#
# This script computes pairwise correlations between neurons during correct
# rejection trials and performs pre vs post statistical analysis.
#
# Key features:
# - Per-trial correlation computation (avoids drift artifacts)
# - Parallelized computation across mice
# - Separate computation and analysis modes (set ANALYSIS_MODE below)
# - Two-level analysis: cell-pair level and mouse-average level
# - Pre-post comparison (pre: days -2/-1 COMBINED, post: days +1/+2 COMBINED)
# - Comprehensive statistical testing with Wilcoxon paired tests and Mann-Whitney U tests
# - Automatic annotation of ALL significant comparisons on plots (sorted by span)
#
# Correlation method:
# - For each cell pair:
#   - PRE period: All trials from days -2 AND -1 are pooled together
#   - POST period: All trials from days +1 AND +2 are pooled together
#   - Correlations are computed within each trial
#   - These per-trial correlations are then averaged to get one value per period
#   - This approach prevents drift between trials from inflating correlations
#
# Outputs (per reward group R+/R-):
# Data files:
# - pairwise_correlations_prepost.csv: Raw correlation data (pre and post periods)
# - pairwise_correlations_mouse_averages_prepost.csv: Mouse-level averages
#
# Statistical test CSV files (pair-level):
# - statistical_tests_prepost_by_pairtype_pairlevel.csv: Pre-post comparison for each pair type (Mann-Whitney U)
# - statistical_tests_pairtypes_prepost_pairlevel.csv: Pair type comparisons within pre and post (Mann-Whitney U)
#
# Statistical test CSV files (mouse-level):
# - statistical_tests_prepost_by_pairtype_mouselevel.csv: Pre-post comparison for each pair type (Wilcoxon paired)
# - statistical_tests_pairtypes_prepost_mouselevel.csv: Pair type comparisons within pre and post (Mann-Whitney U)
#
# Plot files (pair-level):
# - pairwise_correlations_compare_prepost_by_pairtype_pairlevel.svg: Compare pre vs post for each pair type
# - pairwise_correlations_compare_pairtypes_prepost_pairlevel.svg: Compare pair types within pre and post
#
# Plot files (mouse-level):
# - pairwise_correlations_compare_prepost_by_pairtype_mouselevel.svg: Compare pre vs post for each pair type
# - pairwise_correlations_compare_pairtypes_prepost_mouselevel.svg: Compare pair types within pre and post
#
# #############################################################################

# Parameters
# ----------

sampling_rate = 30
win_sec = (-2, 0)  # Use quiet window: 2s before stim onset
pre_days = [-2, -1]  # Pre period: pool trials from these days
post_days = [1, 2]   # Post period: pool trials from these days
N_CORES = 35  # Number of cores for parallel processing

# Analysis mode: 'compute' to compute correlations, 'analyze' to load and analyze existing data
# Set to 'analyze' after first run to skip time-consuming correlation computation
ANALYSIS_MODE = 'compute'  # Options: 'compute' or 'analyze'

# Select sessions from database
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

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
    Process correlations for a single mouse for pre and post periods.

    For each cell pair:
    - Pools all trials from days -2 and -1 for the pre period
    - Pools all trials from days +1 and +2 for the post period
    - Computes correlation for each trial
    - Averages across trials to get one correlation value per period

    Parameters
    ----------
    mouse_id : str
        Mouse identifier

    Returns
    -------
    list of dict
        List of correlation results for this mouse (one per pair per period)
    """
    print(f"\nProcessing mouse: {mouse_id}")

    mouse_results = []

    # Get reward group for this mouse
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Load xarray data for this mouse
    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    xarr.name = 'dff'

    # Filter to only include pre and post days
    all_days = pre_days + post_days
    xarr = xarr.sel(trial=xarr['day'].isin(all_days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

    # Process pre and post periods separately
    for period, period_days in [('pre', pre_days), ('post', post_days)]:
        print(f"  Mouse {mouse_id}: Processing {period} period (days {period_days})...")

        # Select trials for this period (pooling days together)
        xarr_period = xarr.sel(trial=xarr['day'].isin(period_days))

        # Pre-extract all cell data at once to avoid repeated xarray indexing (optimization)
        # Shape: (n_cells, n_trials, n_timepoints)
        all_cells_data = xarr_period.values

        # Get actual dimensions from the extracted data shape
        n_cells = all_cells_data.shape[0]
        n_trials = all_cells_data.shape[1]
        n_timepoints = all_cells_data.shape[2]

        if n_trials == 0:
            print(f"    Mouse {mouse_id}: No trials for {period} period, skipping")
            continue

        print(f"    Mouse {mouse_id}: Found {n_cells} cells, {n_trials} trials (shape: {all_cells_data.shape})")

        # Get cell type information for this period's data
        cell_types = xarr_period.coords['cell_type'].values
        rois = xarr_period.coords['roi'].values

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

            # Average correlation across trials for this period
            if len(trial_correlations) > 0:
                final_corr = np.mean(trial_correlations)

                # Store result
                mouse_results.append({
                    'mouse_id': mouse_id,
                    'reward_group': reward_group,
                    'period': period,
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


# Main execution: Process mice in parallel
if __name__ == '__main__':
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
        corr_csv_path = os.path.join(output_dir, 'pairwise_correlations_prepost.csv')

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
            raw_backup_path = os.path.join(output_dir, 'pairwise_correlations_prepost_raw_backup.csv')
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
                raw_backup_path = os.path.join(output_dir, 'pairwise_correlations_prepost_raw_backup.csv')
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
        # Statistical tests - Pair level
        # #############################################################################

        print("\n" + "="*80)
        print("STATISTICAL TESTS (PAIR-LEVEL)")
        print("="*80)

        # Test 1: Compare pre vs post for each pair type (Mann-Whitney U test at pair level)
        print("\n1. Comparing pre vs post for each pair type (Mann-Whitney U test, n=cell pairs)")
        print("-" * 80)

        prepost_stat_results_pair = []

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            data_pre = pair_data[pair_data['period'] == 'pre']['correlation'].values
            data_post = pair_data[pair_data['period'] == 'post']['correlation'].values

            if len(data_pre) > 0 and len(data_post) > 0:
                stat, p_value = mannwhitneyu(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)

                print(f"  Pre (μ={mean_pre:.3f}, n={len(data_pre)}) vs Post (μ={mean_post:.3f}, n={len(data_post)}): "
                      f"U={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_stat_results_pair.append({
                    'pair_type': pair_type,
                    'mean_pre': mean_pre,
                    'mean_post': mean_post,
                    'n_pre': len(data_pre),
                    'n_post': len(data_post),
                    'stat': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient data (n_pre={len(data_pre)}, n_post={len(data_post)})")

        # Save pair-level pre-post statistical results
        prepost_stat_df_pair = pd.DataFrame(prepost_stat_results_pair)
        prepost_stat_df_pair.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_by_pairtype_pairlevel.csv'), index=False)
        print(f"\n\nSaved pair-level pre-post statistical test results to: {os.path.join(output_dir, 'statistical_tests_prepost_by_pairtype_pairlevel.csv')}")

        # Test 2: Compare pair types within pre and post periods (Mann-Whitney U test at pair level)
        print("\n\n2. Comparing different pair types within pre and post periods (Mann-Whitney U test, n=cell pairs)")
        print("-" * 80)

        period_pairtype_stat_results_pair = []

        for period in ['pre', 'post']:
            print(f"\n{period.upper()} period:")
            period_data = corr_df_filtered[corr_df_filtered['period'] == period]

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

                        period_pairtype_stat_results_pair.append({
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
        period_pairtype_stat_df_pair = pd.DataFrame(period_pairtype_stat_results_pair)
        period_pairtype_stat_df_pair.to_csv(os.path.join(output_dir, 'statistical_tests_pairtypes_prepost_pairlevel.csv'), index=False)
        print(f"\nSaved pair-level pair type comparison stats to: {os.path.join(output_dir, 'statistical_tests_pairtypes_prepost_pairlevel.csv')}")

        # #############################################################################
        # Pair-level plots
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING PAIR-LEVEL PLOTS")
        print("="*80)

        # Plot 1: Pre-Post comparison (pair-level) - 4 panels, one per pair type
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            # Plot bar plot with default seaborn styling
            sns.barplot(
                data=pair_data,
                x='period',
                y='correlation',
                order=['pre', 'post'],
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
            pair_data = corr_df_filtered[corr_df_filtered['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_stats = prepost_stat_df_pair[prepost_stat_df_pair['pair_type'] == pair_type]

            if not pair_prepost_stats.empty:
                p_val = pair_prepost_stats.iloc[0]['p_value']
                if p_val < 0.05:
                    # Get max y values for pre and post
                    y_vals = []
                    for period in ['pre', 'post']:
                        period_corr = pair_data[pair_data['period'] == period]['correlation']
                        if len(period_corr) > 0:
                            y_vals.append(period_corr.mean() + 1.96 * period_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06
                        add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)

        plt.suptitle(f'Pre vs Post Pairwise Correlation During 2 sec quiet windows ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_compare_prepost_by_pairtype_pairlevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # Plot 2: Pair type comparison within pre and post (pair-level) - 2 panels
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]

            # Get data for this period
            period_data = corr_df_filtered[corr_df_filtered['period'] == period]

            # Plot bar plot with default seaborn styling
            sns.barplot(
                data=period_data,
                x='pair_type',
                y='correlation',
                order=pair_types_of_interest,
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
            period_data = corr_df_filtered[corr_df_filtered['period'] == period]

            # Get significant comparisons for this period
            period_stats = period_pairtype_stat_df_pair[(period_pairtype_stat_df_pair['period'] == period) &
                                                         (period_pairtype_stat_df_pair['p_value'] < 0.05)]

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
        svg_file = 'pairwise_correlations_compare_pairtypes_prepost_pairlevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # #############################################################################
        # Mouse-level analysis (average correlation per mouse)
        # #############################################################################

        print("\n" + "="*80)
        print("MOUSE-LEVEL ANALYSIS")
        print("="*80)

        # Compute average correlation per mouse for each period and pair type
        mouse_avg_corr = corr_df_filtered.groupby(['mouse_id', 'reward_group', 'period', 'pair_type'])['correlation'].mean().reset_index()
        mouse_avg_corr.rename(columns={'correlation': 'mean_correlation'}, inplace=True)

        print(f"\nComputed average correlations for {len(mouse_avg_corr['mouse_id'].unique())} mice")

        # Save mouse-level averages
        mouse_avg_corr.to_csv(os.path.join(output_dir, 'pairwise_correlations_mouse_averages_prepost.csv'), index=False)
        print(f"\nSaved mouse-level averages to: {os.path.join(output_dir, 'pairwise_correlations_mouse_averages_prepost.csv')}")

        # #############################################################################
        # Statistical tests - Mouse level
        # #############################################################################

        print("\n" + "="*80)
        print("STATISTICAL TESTS (MOUSE-LEVEL)")
        print("="*80)

        # Test 1: Compare pre vs post for each pair type (Wilcoxon paired test at mouse level)
        print("\n1. Comparing pre vs post for each pair type (Wilcoxon paired test, n=mice)")
        print("-" * 80)

        prepost_stat_results_mouse = []

        for pair_type in pair_types_of_interest:
            print(f"\n{pair_type}:")

            # Get data for this pair type - need to pivot to have pre and post as columns for pairing
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Pivot to get pre and post values for each mouse
            pivot_data = pair_mouse_data.pivot(index='mouse_id', columns='period', values='mean_correlation')

            # Only keep mice that have both pre and post data
            paired_data = pivot_data.dropna()

            if len(paired_data) > 0:
                data_pre = paired_data['pre'].values
                data_post = paired_data['post'].values

                # Wilcoxon signed-rank test (paired)
                stat, p_value = wilcoxon(data_pre, data_post, alternative='two-sided')

                mean_pre = np.mean(data_pre)
                mean_post = np.mean(data_post)

                print(f"  Pre (μ={mean_pre:.3f}) vs Post (μ={mean_post:.3f}), n={len(data_pre)} mice: "
                      f"W={stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                prepost_stat_results_mouse.append({
                    'pair_type': pair_type,
                    'mean_pre': mean_pre,
                    'mean_post': mean_post,
                    'n_mice': len(data_pre),
                    'stat': stat,
                    'p_value': p_value
                })
            else:
                print(f"  Insufficient paired data")

        # Save mouse-level pre-post statistical results
        prepost_stat_df_mouse = pd.DataFrame(prepost_stat_results_mouse)
        prepost_stat_df_mouse.to_csv(os.path.join(output_dir, 'statistical_tests_prepost_by_pairtype_mouselevel.csv'), index=False)
        print(f"\n\nSaved mouse-level pre-post statistical test results to: {os.path.join(output_dir, 'statistical_tests_prepost_by_pairtype_mouselevel.csv')}")

        # Test 2: Compare pair types within pre and post periods (Mann-Whitney U test at mouse level)
        print("\n\n2. Comparing different pair types within pre and post periods (Mann-Whitney U test, n=mice)")
        print("-" * 80)

        period_pairtype_stat_results_mouse = []

        for period in ['pre', 'post']:
            print(f"\n{period.upper()} period:")
            period_mouse_data = mouse_avg_corr[mouse_avg_corr['period'] == period]

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

                        period_pairtype_stat_results_mouse.append({
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
        period_pairtype_stat_df_mouse = pd.DataFrame(period_pairtype_stat_results_mouse)
        period_pairtype_stat_df_mouse.to_csv(os.path.join(output_dir, 'statistical_tests_pairtypes_prepost_mouselevel.csv'), index=False)
        print(f"\nSaved mouse-level pair type comparison stats to: {os.path.join(output_dir, 'statistical_tests_pairtypes_prepost_mouselevel.csv')}")

        # #############################################################################
        # Mouse-level plots
        # #############################################################################

        print("\n" + "="*80)
        print("GENERATING MOUSE-LEVEL PLOTS")
        print("="*80)

        # Plot 3: Pre-Post comparison (mouse-level) - 4 panels, one per pair type
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

        for idx, pair_type in enumerate(pair_types_of_interest):
            ax = axes[idx]

            # Get data for this pair type
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Plot bar plot with default seaborn styling
            sns.barplot(
                data=pair_mouse_data,
                x='period',
                y='mean_correlation',
                order=['pre', 'post'],
                ax=ax,
                errorbar='ci',
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
            pair_mouse_data = mouse_avg_corr[mouse_avg_corr['pair_type'] == pair_type]

            # Get pre-post comparison stats
            pair_prepost_mouse_stats = prepost_stat_df_mouse[prepost_stat_df_mouse['pair_type'] == pair_type]

            if not pair_prepost_mouse_stats.empty:
                p_val = pair_prepost_mouse_stats.iloc[0]['p_value']
                if p_val < 0.05:
                    # Get max y values for pre and post
                    y_vals = []
                    for period in ['pre', 'post']:
                        period_corr = pair_mouse_data[pair_mouse_data['period'] == period]['mean_correlation']
                        if len(period_corr) > 0:
                            y_vals.append(period_corr.mean() + 1.96 * period_corr.sem())

                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06
                        add_significance_stars(ax, 0, 1, y_max + y_offset, p_val)

        plt.suptitle(f'Pre vs Post Pairwise Correlation - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_compare_prepost_by_pairtype_mouselevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # Plot 4: Pair type comparison within pre and post (mouse-level) - 2 panels
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]

            # Get data for this period
            period_mouse_data = mouse_avg_corr[mouse_avg_corr['period'] == period]

            # Plot bar plot with default seaborn styling
            sns.barplot(
                data=period_mouse_data,
                x='pair_type',
                y='mean_correlation',
                order=pair_types_of_interest,
                ax=ax,
                errorbar='ci',
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
            period_mouse_data = mouse_avg_corr[mouse_avg_corr['period'] == period]

            # Get significant comparisons for this period
            period_mouse_stats = period_pairtype_stat_df_mouse[(period_pairtype_stat_df_mouse['period'] == period) &
                                                                (period_pairtype_stat_df_mouse['p_value'] < 0.05)]

            if not period_mouse_stats.empty:
                # Get all significant comparisons sorted by span
                comparisons = get_all_significant_comparisons_sorted(period_mouse_stats, pair_types_of_interest,
                                                                     period_mouse_data, value_col='mean_correlation')

                # Add all significant comparisons with proper offset
                for offset_level, (pair1, pair2, idx1, idx2, p_val) in enumerate(comparisons):
                    # Get the maximum y value across both bars
                    y_vals = []
                    for pair_type in [pair1, pair2]:
                        pair_data = period_mouse_data[period_mouse_data['pair_type'] == pair_type]['mean_correlation']
                        if len(pair_data) > 0:
                            y_vals.append(pair_data.mean() + 1.96 * pair_data.sem())
                    if y_vals:
                        y_max = max(y_vals)
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.06 * offset_level
                        add_significance_stars(ax, idx1, idx2, y_max + y_offset, p_val)

        plt.suptitle(f'Pair Type Comparison Within Pre and Post Periods - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_compare_pairtypes_prepost_mouselevel.svg'
        plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: {svg_file}")
        plt.close()

        # #############################################################################
        # Summary
        # #############################################################################

        print(f"\n{'='*80}")
        print(f"Analysis complete for {reward_group}!")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print("Files generated:")
        print("  Data:")
        print("    - pairwise_correlations_prepost.csv")
        print("    - pairwise_correlations_mouse_averages_prepost.csv")
        print("  Pair-level:")
        print("    - pairwise_correlations_compare_prepost_by_pairtype_pairlevel.svg")
        print("    - pairwise_correlations_compare_pairtypes_prepost_pairlevel.svg")
        print("    - statistical_tests_prepost_by_pairtype_pairlevel.csv")
        print("    - statistical_tests_pairtypes_prepost_pairlevel.csv")
        print("  Mouse-level:")
        print("    - pairwise_correlations_compare_prepost_by_pairtype_mouselevel.svg")
        print("    - pairwise_correlations_compare_pairtypes_prepost_mouselevel.svg")
        print("    - statistical_tests_prepost_by_pairtype_mouselevel.csv")
        print("    - statistical_tests_pairtypes_prepost_mouselevel.csv")

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")
