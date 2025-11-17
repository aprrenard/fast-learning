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
# Pairwise correlations between cells during correct rejection trials
# #############################################################################
#
# This script computes pairwise correlations between neurons during correct
# rejection trials (no_stim=1, lick_flag=0) and performs statistical analysis.
#
# Key features:
# - Parallelized computation across mice
# - Separate computation and analysis modes (set ANALYSIS_MODE below)
# - Two-level analysis: cell-pair level and mouse-average level
# - Comprehensive statistical testing with Mann-Whitney U tests
# - Automatic annotation of ALL significant comparisons on plots (sorted by span)
#
# Outputs (per reward group R+/R-):
# - pairwise_correlations.csv: Raw correlation data
# - pairwise_correlations_mouse_averages.csv: Mouse-level averages
# - statistical_tests.csv: Pair-level statistical tests
# - statistical_tests_mouse_level.csv: Mouse-level statistical tests
# - 4 plots with statistical annotations (2 pair-level, 2 mouse-level)
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

    # Get cell type information
    cell_types = xarr.coords['cell_type'].values
    rois = xarr.coords['roi'].values
    n_cells = len(rois)

    print(f"  Mouse {mouse_id}: Found {n_cells} cells")

    # Process each day separately
    for day in days:
        print(f"  Mouse {mouse_id}: Processing day {day}...")

        # Select trials for this day
        xarr_day = xarr.sel(trial=xarr['day'] == day)
        n_trials = len(xarr_day.trial)

        if n_trials == 0:
            print(f"    Mouse {mouse_id}: No trials for day {day}, skipping")
            continue

        print(f"    Mouse {mouse_id}: Found {n_trials} trials")

        # For each pair of cells, compute correlation
        # We concatenate all time points across all trials for each cell
        for i, j in combinations(range(n_cells), 2):
            # Get activity traces for both cells
            # Shape: (n_trials, n_timepoints)
            cell_i_data = xarr_day.isel(cell=i).values
            cell_j_data = xarr_day.isel(cell=j).values

            # Flatten to concatenate all time points across trials
            # This gives us a 1D array for each cell
            cell_i_flat = cell_i_data.flatten()
            cell_j_flat = cell_j_data.flatten()

            # Remove any NaN values (if present)
            valid_idx = ~(np.isnan(cell_i_flat) | np.isnan(cell_j_flat))
            cell_i_clean = cell_i_flat[valid_idx]
            cell_j_clean = cell_j_flat[valid_idx]

            # Compute Pearson correlation
            if len(cell_i_clean) > 1:  # Need at least 2 points for correlation
                corr, _ = pearsonr(cell_i_clean, cell_j_clean)

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
                    'correlation': corr
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
        corr_csv_path = os.path.join(output_dir, 'pairwise_correlations.csv')

        if ANALYSIS_MODE == 'compute' or not os.path.exists(corr_csv_path):
            print(f"\nMode: COMPUTE - Computing pairwise correlations for {len(group_mice)} mice using {N_CORES} cores...")
            print(f"Mice in {reward_group}: {group_mice}")

            # Use multiprocessing to parallelize across mice
            with Pool(processes=N_CORES) as pool:
                results_list = pool.map(process_mouse, group_mice)

            # Flatten the list of lists into a single list
            correlation_results = [item for sublist in results_list for item in sublist]

            print(f"\nCorrelation computation complete for {reward_group}!")

            # Convert results to DataFrame
            corr_df = pd.DataFrame(correlation_results)

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
            corr_df_filtered = pd.read_csv(corr_csv_path)
            pair_types_of_interest = ['all-all', 'wS2-wS2', 'wM1-wM1', 'wS2-wM1']
            print(f"Loaded {len(corr_df_filtered)} correlation pairs")
            print(f"Breakdown by pair type:")
            print(corr_df_filtered['pair_type'].value_counts())


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

        plt.suptitle(f'Average Pairwise Correlation During Correct Rejection Trials ({reward_group})', fontsize=16, y=1.02)
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

        plt.suptitle(f'Pairwise Correlation Across Days During Correct Rejection Trials ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_across_days.svg'
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

        plt.suptitle(f'Average Pairwise Correlation During Correct Rejection Trials - Mouse Level ({reward_group})', fontsize=16, y=1.02)
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

        plt.suptitle(f'Pairwise Correlation Across Days During Correct Rejection Trials - Mouse Level ({reward_group})', fontsize=16, y=1.02)
        plt.tight_layout()
        sns.despine()

        # Save figure
        svg_file = 'pairwise_correlations_across_days_mouse_level.svg'
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
        print("  - statistical_tests.csv")
        print("  - pairwise_correlations_mouse_averages.csv")
        print("  - pairwise_correlations_by_day_mouse_level.svg")
        print("  - pairwise_correlations_across_days_mouse_level.svg")
        print("  - statistical_tests_mouse_level.csv")

    print(f"\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}")
