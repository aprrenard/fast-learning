import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon, chi2_contingency, fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils 
import src.utils.utils_io as io
from src.utils.utils_plot import *
from scipy.stats import ks_2samp
import colorsys
from matplotlib import colors as mcolors


# #############################################################################
# Contribution of projection neurons to LMI and classifier weights.
# Simplified: analyze LMI and weights independently (no merge, no statistics).
# #############################################################################


print("\n" + "="*80)
print("PROJECTION NEURON CONTRIBUTIONS TO LMI AND CLASSIFIER WEIGHTS")
print("="*80 + "\n")

# Output directory
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/projectors_contributions'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Define cell types for analysis
# For within-type analysis: only wS2 and wM1
# For composition analysis: all three types (non_projector, wS2, wM1)
cell_types = ['wS2', 'wM1']  # For within-type proportions
all_cell_types_for_viz = ['non_projector', 'wS2', 'wM1']  # For composition plots

cell_type_colors = {
    'wS2': s2_m1_palette[0],
    'wM1': s2_m1_palette[1],
}

# =============================================================================
# PART 1: LMI ANALYSIS
# =============================================================================

print("\nPART 1: LMI ANALYSIS")
print("-" * 80)

# Load LMI data
processed_folder = io.solve_common_paths('processed_data')
lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

# Add reward group
for mouse in lmi_df.mouse_id.unique():
    lmi_df.loc[lmi_df.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

print(f"LMI data: {len(lmi_df)} cells from {lmi_df['mouse_id'].nunique()} mice")

# Define cell type groups (non-projectors are labeled 'na' in cell_type column)
lmi_df['cell_type_group'] = lmi_df['cell_type'].copy()
lmi_df.loc[lmi_df['cell_type'] == 'na', 'cell_type_group'] = 'non_projector'
lmi_df.loc[lmi_df['cell_type'] == 'wS2', 'cell_type_group'] = 'wS2'
lmi_df.loc[lmi_df['cell_type'] == 'wM1', 'cell_type_group'] = 'wM1'

print(f"Cell types (all): {lmi_df['cell_type_group'].value_counts().to_dict()}")

# Define LMI criteria ON ALL CELLS (including non-projectors)
# This is important for composition analysis: "Of top X% of ALL cells, what % are projectors?"
# Significant LMI (positive and negative)
lmi_df['significant_pos_lmi'] = lmi_df['lmi_p'] >= 0.975
lmi_df['significant_neg_lmi'] = lmi_df['lmi_p'] <= 0.025

# Top percentiles for positive and negative LMI separately (calculated across ALL cells)
# Simplified to only keep top 20%
pos_lmi = lmi_df[lmi_df['lmi'] > 0]['lmi']
neg_lmi = lmi_df[lmi_df['lmi'] < 0]['lmi']

lmi_df['top_20pct_pos_lmi'] = False
lmi_df['top_20pct_neg_lmi'] = False

if len(pos_lmi) > 0:
    lmi_df.loc[lmi_df['lmi'] > 0, 'top_20pct_pos_lmi'] = lmi_df.loc[lmi_df['lmi'] > 0, 'lmi'] >= np.percentile(pos_lmi, 80)

if len(neg_lmi) > 0:
    lmi_df.loc[lmi_df['lmi'] < 0, 'top_20pct_neg_lmi'] = lmi_df.loc[lmi_df['lmi'] < 0, 'lmi'] <= np.percentile(neg_lmi, 20)

# Compute within-type proportions (only for wS2 and wM1)
# This answers: "Of all wS2 cells, what % are in top X%?"
lmi_criteria = [
    'significant_pos_lmi', 'significant_neg_lmi',
    'top_20pct_pos_lmi', 'top_20pct_neg_lmi'
]
lmi_proportions = []

for reward_group in ['R+', 'R-']:
    for cell_type in cell_types:  # Only wS2 and wM1
        subset = lmi_df[(lmi_df['reward_group'] == reward_group) & (lmi_df['cell_type_group'] == cell_type)]
        n_total = len(subset)

        for criterion in lmi_criteria:
            n_met = subset[criterion].sum()
            lmi_proportions.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_total': n_total,
                'n_met': n_met,
                'proportion': n_met / n_total if n_total > 0 else 0
            })

df_lmi_props = pd.DataFrame(lmi_proportions)
df_lmi_props.to_csv(os.path.join(output_dir, 'lmi_proportions_within.csv'), index=False)
print("Saved: lmi_proportions_within.csv")

# Compute composition perspective: "Of top X% of ALL cells, what % are non-projector, wS2, or wM1?"
# This answers the key question: among the most important cells (top percentiles),
# how many are projection neurons vs non-projectors?
lmi_composition = []
all_cell_types = ['non_projector', 'wS2', 'wM1']  # Include all three types

for reward_group in ['R+', 'R-']:
    data_rg = lmi_df[lmi_df['reward_group'] == reward_group]

    for criterion in lmi_criteria:
        # Get cells that meet this criterion (from ALL cells, not just projectors)
        top_cells = data_rg[data_rg[criterion] == True]
        n_top_total = len(top_cells)

        for cell_type in all_cell_types:  # Check all three types
            n_this_type = len(top_cells[top_cells['cell_type_group'] == cell_type])
            lmi_composition.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_top_total': n_top_total,
                'n_this_type': n_this_type,
                'proportion': n_this_type / n_top_total if n_top_total > 0 else 0
            })

df_lmi_comp = pd.DataFrame(lmi_composition)
df_lmi_comp.to_csv(os.path.join(output_dir, 'lmi_proportions_composition.csv'), index=False)
print("Saved: lmi_proportions_composition.csv")

# =============================================================================
# PART 2: CLASSIFIER WEIGHTS ANALYSIS
# =============================================================================

print("\nPART 2: CLASSIFIER WEIGHTS ANALYSIS")
print("-" * 80)

# Load weights
weights_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
weights_dir = io.adjust_path_to_host(weights_dir)
weights_df = pd.read_csv(os.path.join(weights_dir, 'classifier_weights.csv'))

print(f"Weights data: {len(weights_df)} cells from {weights_df['mouse_id'].nunique()} mice")

# Load cell types from xarray
cell_type_info = []
for mouse_id in weights_df['mouse_id'].unique():
    try:
        data_xr = imaging_utils.load_mouse_xarray(
            mouse_id,
            os.path.join(io.processed_dir, 'mice'),
            'tensor_xarray_mapping_data.nc'
        )
        rois = data_xr.coords['roi'].values
        cts = data_xr.coords['cell_type'].values if 'cell_type' in data_xr.coords else [None] * len(rois)

        for roi, ct in zip(rois, cts):
            cell_type_info.append({'mouse_id': mouse_id, 'roi': roi, 'cell_type_xr': ct})
    except Exception as e:
        print(f"Warning: Could not load cell types for {mouse_id}: {e}")
        pass

# Merge weights with cell types
if len(cell_type_info) > 0:
    weights_df = weights_df.merge(pd.DataFrame(cell_type_info), on=['mouse_id', 'roi'], how='left')
else:
    print("Warning: No cell type info loaded, all cells will be classified as non-projectors")
    weights_df['cell_type_xr'] = None

# Define cell type groups (non-projectors are labeled 'na' in xarray or NaN)
weights_df['cell_type_group'] = weights_df['cell_type_xr'].copy()
weights_df.loc[(weights_df['cell_type_xr'].isna()) | (weights_df['cell_type_xr'] == 'na'), 'cell_type_group'] = 'non_projector'
weights_df.loc[weights_df['cell_type_xr'] == 'wS2', 'cell_type_group'] = 'wS2'
weights_df.loc[weights_df['cell_type_xr'] == 'wM1', 'cell_type_group'] = 'wM1'

print(f"Cell types (all): {weights_df['cell_type_group'].value_counts().to_dict()}")

# Define weight criteria ON ALL CELLS (including non-projectors)
# This is important for composition analysis: "Of top X% of ALL cells, what % are projectors?"
# Simplified to only keep top 20%
pos_weights = weights_df[weights_df['classifier_weight'] > 0]['classifier_weight']
neg_weights = weights_df[weights_df['classifier_weight'] < 0]['classifier_weight']

weights_df['top_20pct_pos_weight'] = False
weights_df['top_20pct_neg_weight'] = False

if len(pos_weights) > 0:
    weights_df.loc[weights_df['classifier_weight'] > 0, 'top_20pct_pos_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] > 0, 'classifier_weight'] >= np.percentile(pos_weights, 80)

if len(neg_weights) > 0:
    weights_df.loc[weights_df['classifier_weight'] < 0, 'top_20pct_neg_weight'] = \
        weights_df.loc[weights_df['classifier_weight'] < 0, 'classifier_weight'] <= np.percentile(neg_weights, 20)

# Compute within-type proportions (only for wS2 and wM1)
# This answers: "Of all wS2 cells, what % are in top X% weights?"
weight_criteria = [
    'top_20pct_pos_weight', 'top_20pct_neg_weight'
]
weight_proportions = []

for reward_group in ['R+', 'R-']:
    for cell_type in cell_types:  # Only wS2 and wM1
        subset = weights_df[(weights_df['reward_group'] == reward_group) & (weights_df['cell_type_group'] == cell_type)]
        n_total = len(subset)

        for criterion in weight_criteria:
            n_met = subset[criterion].sum()
            weight_proportions.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_total': n_total,
                'n_met': n_met,
                'proportion': n_met / n_total if n_total > 0 else 0
            })

df_weight_props = pd.DataFrame(weight_proportions)
df_weight_props.to_csv(os.path.join(output_dir, 'weight_proportions_within.csv'), index=False)
print("Saved: weight_proportions_within.csv")

# Compute composition perspective: "Of top X% of ALL cells, what % are non-projector, wS2, or wM1?"
# This answers the key question: among the cells with highest decoder weights,
# how many are projection neurons vs non-projectors?
weight_composition = []
all_cell_types = ['non_projector', 'wS2', 'wM1']  # Include all three types

for reward_group in ['R+', 'R-']:
    data_rg = weights_df[weights_df['reward_group'] == reward_group]

    for criterion in weight_criteria:
        # Get cells that meet this criterion (from ALL cells, not just projectors)
        top_cells = data_rg[data_rg[criterion] == True]
        n_top_total = len(top_cells)

        for cell_type in all_cell_types:  # Check all three types
            n_this_type = len(top_cells[top_cells['cell_type_group'] == cell_type])
            weight_composition.append({
                'reward_group': reward_group,
                'cell_type': cell_type,
                'criterion': criterion,
                'n_top_total': n_top_total,
                'n_this_type': n_this_type,
                'proportion': n_this_type / n_top_total if n_top_total > 0 else 0
            })

df_weight_comp = pd.DataFrame(weight_composition)
df_weight_comp.to_csv(os.path.join(output_dir, 'weight_proportions_composition.csv'), index=False)
print("Saved: weight_proportions_composition.csv")


# =============================================================================
# PART 3: STATISTICAL TESTING (POOLED DATA)
# =============================================================================

print("\nPART 3: STATISTICAL TESTING (POOLED DATA)")
print("-" * 80)

# Parameter: minimum number of cells per cell type per mouse (for average values analysis)
MIN_CELLS_THRESHOLD = 5

def pvalue_to_stars(p):
    """Convert p-value to significance notation."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def compute_fisher_exact_pvalue(raw_df, reward_group, criterion, cell_type_col='cell_type_group'):
    """
    Compute Fisher's exact test for pooled data comparing wS2 vs wM1.

    Parameters:
    -----------
    raw_df : DataFrame with raw cell-level data
    reward_group : 'R+' or 'R-'
    criterion : criterion column name (e.g., 'significant_pos_lmi')
    cell_type_col : column name for cell type

    Returns:
    --------
    dict with statistic, pvalue, stars
    """
    # Filter data
    data_rg = raw_df[raw_df['reward_group'] == reward_group]

    # Build 2x2 contingency table
    wS2_data = data_rg[data_rg[cell_type_col] == 'wS2']
    wM1_data = data_rg[data_rg[cell_type_col] == 'wM1']

    wS2_met = wS2_data[criterion].sum()
    wS2_not_met = len(wS2_data) - wS2_met
    wM1_met = wM1_data[criterion].sum()
    wM1_not_met = len(wM1_data) - wM1_met

    # Contingency table: [[wS2_met, wS2_not_met], [wM1_met, wM1_not_met]]
    table = [[wS2_met, wS2_not_met], [wM1_met, wM1_not_met]]

    # Fisher's exact test (two-sided)
    odds_ratio, pvalue = fisher_exact(table, alternative='two-sided')

    return {
        'statistic': odds_ratio,
        'pvalue': pvalue,
        'stars': pvalue_to_stars(pvalue)
    }

# Compute statistics for LMI (pooled - Fisher's exact test)
print("Computing LMI statistics (pooled - Fisher's exact test)...")
lmi_stats_pooled = []
for reward_group in ['R+', 'R-']:
    for criterion in lmi_criteria:
        result = compute_fisher_exact_pvalue(lmi_df, reward_group, criterion)
        lmi_stats_pooled.append({
            'reward_group': reward_group,
            'criterion': criterion,
            'test_type': 'fisher_exact',
            'statistic': result['statistic'],
            'pvalue': result['pvalue'],
            'stars': result['stars']
        })

df_lmi_stats_pooled = pd.DataFrame(lmi_stats_pooled)
df_lmi_stats_pooled.to_csv(os.path.join(output_dir, 'lmi_statistics_pooled.csv'), index=False)
print(f"  Saved: lmi_statistics_pooled.csv")

# Compute statistics for Weights (pooled - Fisher's exact test)
print("Computing weight statistics (pooled - Fisher's exact test)...")
weight_stats_pooled = []
for reward_group in ['R+', 'R-']:
    for criterion in weight_criteria:
        result = compute_fisher_exact_pvalue(weights_df, reward_group, criterion)
        weight_stats_pooled.append({
            'reward_group': reward_group,
            'criterion': criterion,
            'test_type': 'fisher_exact',
            'statistic': result['statistic'],
            'pvalue': result['pvalue'],
            'stars': result['stars']
        })

df_weight_stats_pooled = pd.DataFrame(weight_stats_pooled)
df_weight_stats_pooled.to_csv(os.path.join(output_dir, 'weight_statistics_pooled.csv'), index=False)
print(f"  Saved: weight_statistics_pooled.csv")

# =============================================================================
# PART 4: AVERAGE VALUES ANALYSIS (LMI and Weights) ACROSS MICE
# =============================================================================
# IMPORTANT: To properly analyze positive vs negative LMI/weights, we compute
# per-mouse averages SEPARATELY for positive and negative cells.
#
# Previous approach (INCORRECT):
#   1. Compute per-mouse mean across ALL cells (mixing positive and negative)
#   2. Filter mice where mean > 0 or mean < 0
#   3. This could exclude mice with mostly negative cells even if they have
#      some positive cells, leading to incorrect variance estimates
#
# Corrected approach:
#   1. For each mouse and cell type, select ONLY positive cells, compute mean
#   2. For each mouse and cell type, select ONLY negative cells, compute mean
#   3. This ensures we're measuring the actual positive/negative signal strength,
#      not an average that mixes both polarities
# =============================================================================

print("\nPART 4: AVERAGE VALUES ANALYSIS")
print("-" * 80)

# Compute per-mouse mean LMI for wS2 and wM1
# Compute SEPARATELY for positive and negative LMI values
print("Computing per-mouse mean LMI values...")
lmi_means_per_mouse = []
lmi_means_per_mouse_pos = []
lmi_means_per_mouse_neg = []

for mouse_id in lmi_df['mouse_id'].unique():
    mouse_data = lmi_df[lmi_df['mouse_id'] == mouse_id]
    reward_group = mouse_data['reward_group'].iloc[0]

    for cell_type in cell_types:  # wS2 and wM1
        type_data = mouse_data[mouse_data['cell_type_group'] == cell_type]

        if len(type_data) >= MIN_CELLS_THRESHOLD:
            mean_lmi = type_data['lmi'].mean()
            lmi_means_per_mouse.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_lmi': mean_lmi,
                'n_cells': len(type_data)
            })
        
        # Compute mean for POSITIVE LMI cells only
        type_data_pos = type_data[type_data['lmi'] > 0]
        if len(type_data_pos) >= MIN_CELLS_THRESHOLD:
            mean_lmi_pos = type_data_pos['lmi'].mean()
            lmi_means_per_mouse_pos.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_lmi': mean_lmi_pos,
                'n_cells': len(type_data_pos)
            })
        
        # Compute mean for NEGATIVE LMI cells only
        type_data_neg = type_data[type_data['lmi'] < 0]
        if len(type_data_neg) >= MIN_CELLS_THRESHOLD:
            mean_lmi_neg = type_data_neg['lmi'].mean()
            lmi_means_per_mouse_neg.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_lmi': mean_lmi_neg,
                'n_cells': len(type_data_neg)
            })

df_lmi_means = pd.DataFrame(lmi_means_per_mouse)
df_lmi_means.to_csv(os.path.join(output_dir, 'lmi_mean_values_per_mouse.csv'), index=False)
print(f"  Saved: lmi_mean_values_per_mouse.csv")
print(f"  Included {df_lmi_means['mouse_id'].nunique()} mice")

df_lmi_means_pos = pd.DataFrame(lmi_means_per_mouse_pos)
df_lmi_means_pos.to_csv(os.path.join(output_dir, 'lmi_mean_values_per_mouse_positive_only.csv'), index=False)
print(f"  Saved: lmi_mean_values_per_mouse_positive_only.csv")
print(f"  Included {df_lmi_means_pos['mouse_id'].nunique()} mice for positive LMI")

df_lmi_means_neg = pd.DataFrame(lmi_means_per_mouse_neg)
df_lmi_means_neg.to_csv(os.path.join(output_dir, 'lmi_mean_values_per_mouse_negative_only.csv'), index=False)
print(f"  Saved: lmi_mean_values_per_mouse_negative_only.csv")
print(f"  Included {df_lmi_means_neg['mouse_id'].nunique()} mice for negative LMI")

# Compute per-mouse mean weights for wS2 and wM1
# Compute SEPARATELY for positive and negative weights
print("Computing per-mouse mean classifier weights...")
weight_means_per_mouse = []
weight_means_per_mouse_pos = []
weight_means_per_mouse_neg = []

for mouse_id in weights_df['mouse_id'].unique():
    mouse_data = weights_df[weights_df['mouse_id'] == mouse_id]
    reward_group = mouse_data['reward_group'].iloc[0]

    for cell_type in cell_types:  # wS2 and wM1
        type_data = mouse_data[mouse_data['cell_type_group'] == cell_type]

        if len(type_data) >= MIN_CELLS_THRESHOLD:
            mean_weight = type_data['classifier_weight'].mean()
            weight_means_per_mouse.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_weight': mean_weight,
                'n_cells': len(type_data)
            })
        
        # Compute mean for POSITIVE weights only
        type_data_pos = type_data[type_data['classifier_weight'] > 0]
        if len(type_data_pos) >= MIN_CELLS_THRESHOLD:
            mean_weight_pos = type_data_pos['classifier_weight'].mean()
            weight_means_per_mouse_pos.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_weight': mean_weight_pos,
                'n_cells': len(type_data_pos)
            })
        
        # Compute mean for NEGATIVE weights only
        type_data_neg = type_data[type_data['classifier_weight'] < 0]
        if len(type_data_neg) >= MIN_CELLS_THRESHOLD:
            mean_weight_neg = type_data_neg['classifier_weight'].mean()
            weight_means_per_mouse_neg.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'cell_type': cell_type,
                'mean_weight': mean_weight_neg,
                'n_cells': len(type_data_neg)
            })

df_weight_means = pd.DataFrame(weight_means_per_mouse)
df_weight_means.to_csv(os.path.join(output_dir, 'weight_mean_values_per_mouse.csv'), index=False)
print(f"  Saved: weight_mean_values_per_mouse.csv")
print(f"  Included {df_weight_means['mouse_id'].nunique()} mice")

df_weight_means_pos = pd.DataFrame(weight_means_per_mouse_pos)
df_weight_means_pos.to_csv(os.path.join(output_dir, 'weight_mean_values_per_mouse_positive_only.csv'), index=False)
print(f"  Saved: weight_mean_values_per_mouse_positive_only.csv")
print(f"  Included {df_weight_means_pos['mouse_id'].nunique()} mice for positive weights")

df_weight_means_neg = pd.DataFrame(weight_means_per_mouse_neg)
df_weight_means_neg.to_csv(os.path.join(output_dir, 'weight_mean_values_per_mouse_negative_only.csv'), index=False)
print(f"  Saved: weight_mean_values_per_mouse_negative_only.csv")
print(f"  Included {df_weight_means_neg['mouse_id'].nunique()} mice for negative weights")

# Statistical testing: Mann-Whitney U comparing wS2 vs wM1 for each reward group
# Now computed separately for positive and negative values
print("\nComputing statistics for average values (Mann-Whitney U test)...")

# LMI statistics - using the correctly computed positive/negative-only dataframes
lmi_mean_stats_pos = []
lmi_mean_stats_neg = []

for reward_group in ['R+', 'R-']:
    # Positive LMI statistics
    data_rg_pos = df_lmi_means_pos[df_lmi_means_pos['reward_group'] == reward_group]
    wS2_means_pos = data_rg_pos[data_rg_pos['cell_type'] == 'wS2']['mean_lmi'].values
    wM1_means_pos = data_rg_pos[data_rg_pos['cell_type'] == 'wM1']['mean_lmi'].values

    if len(wS2_means_pos) > 0 and len(wM1_means_pos) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means_pos, wM1_means_pos, alternative='two-sided')
        lmi_mean_stats_pos.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means_pos),
            'n_wM1': len(wM1_means_pos)
        })
    else:
        lmi_mean_stats_pos.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means_pos),
            'n_wM1': len(wM1_means_pos)
        })
    
    # Negative LMI statistics
    data_rg_neg = df_lmi_means_neg[df_lmi_means_neg['reward_group'] == reward_group]
    wS2_means_neg = data_rg_neg[data_rg_neg['cell_type'] == 'wS2']['mean_lmi'].values
    wM1_means_neg = data_rg_neg[data_rg_neg['cell_type'] == 'wM1']['mean_lmi'].values

    if len(wS2_means_neg) > 0 and len(wM1_means_neg) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means_neg, wM1_means_neg, alternative='two-sided')
        lmi_mean_stats_neg.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means_neg),
            'n_wM1': len(wM1_means_neg)
        })
    else:
        lmi_mean_stats_neg.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means_neg),
            'n_wM1': len(wM1_means_neg)
        })

# Keep the original combined stats for backwards compatibility
lmi_mean_stats = []
for reward_group in ['R+', 'R-']:
    data_rg = df_lmi_means[df_lmi_means['reward_group'] == reward_group]

    wS2_means = data_rg[data_rg['cell_type'] == 'wS2']['mean_lmi'].values
    wM1_means = data_rg[data_rg['cell_type'] == 'wM1']['mean_lmi'].values

    if len(wS2_means) > 0 and len(wM1_means) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means, wM1_means, alternative='two-sided')
        lmi_mean_stats.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means),
            'n_wM1': len(wM1_means)
        })
    else:
        lmi_mean_stats.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means),
            'n_wM1': len(wM1_means)
        })

df_lmi_mean_stats = pd.DataFrame(lmi_mean_stats)
df_lmi_mean_stats.to_csv(os.path.join(output_dir, 'lmi_mean_values_statistics.csv'), index=False)
print(f"  Saved: lmi_mean_values_statistics.csv")

df_lmi_mean_stats_pos = pd.DataFrame(lmi_mean_stats_pos)
df_lmi_mean_stats_pos.to_csv(os.path.join(output_dir, 'lmi_mean_values_statistics_positive.csv'), index=False)
print(f"  Saved: lmi_mean_values_statistics_positive.csv")

df_lmi_mean_stats_neg = pd.DataFrame(lmi_mean_stats_neg)
df_lmi_mean_stats_neg.to_csv(os.path.join(output_dir, 'lmi_mean_values_statistics_negative.csv'), index=False)
print(f"  Saved: lmi_mean_values_statistics_negative.csv")

# Weight statistics - using the correctly computed positive/negative-only dataframes
weight_mean_stats_pos = []
weight_mean_stats_neg = []

for reward_group in ['R+', 'R-']:
    # Positive weight statistics
    data_rg_pos = df_weight_means_pos[df_weight_means_pos['reward_group'] == reward_group]
    wS2_means_pos = data_rg_pos[data_rg_pos['cell_type'] == 'wS2']['mean_weight'].values
    wM1_means_pos = data_rg_pos[data_rg_pos['cell_type'] == 'wM1']['mean_weight'].values

    if len(wS2_means_pos) > 0 and len(wM1_means_pos) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means_pos, wM1_means_pos, alternative='two-sided')
        weight_mean_stats_pos.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means_pos),
            'n_wM1': len(wM1_means_pos)
        })
    else:
        weight_mean_stats_pos.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means_pos),
            'n_wM1': len(wM1_means_pos)
        })
    
    # Negative weight statistics
    data_rg_neg = df_weight_means_neg[df_weight_means_neg['reward_group'] == reward_group]
    wS2_means_neg = data_rg_neg[data_rg_neg['cell_type'] == 'wS2']['mean_weight'].values
    wM1_means_neg = data_rg_neg[data_rg_neg['cell_type'] == 'wM1']['mean_weight'].values

    if len(wS2_means_neg) > 0 and len(wM1_means_neg) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means_neg, wM1_means_neg, alternative='two-sided')
        weight_mean_stats_neg.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means_neg),
            'n_wM1': len(wM1_means_neg)
        })
    else:
        weight_mean_stats_neg.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means_neg),
            'n_wM1': len(wM1_means_neg)
        })

# Keep the original combined stats for backwards compatibility
weight_mean_stats = []
for reward_group in ['R+', 'R-']:
    data_rg = df_weight_means[df_weight_means['reward_group'] == reward_group]

    wS2_means = data_rg[data_rg['cell_type'] == 'wS2']['mean_weight'].values
    wM1_means = data_rg[data_rg['cell_type'] == 'wM1']['mean_weight'].values

    if len(wS2_means) > 0 and len(wM1_means) > 0:
        statistic, pvalue = mannwhitneyu(wS2_means, wM1_means, alternative='two-sided')
        weight_mean_stats.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'stars': pvalue_to_stars(pvalue),
            'n_wS2': len(wS2_means),
            'n_wM1': len(wM1_means)
        })
    else:
        weight_mean_stats.append({
            'reward_group': reward_group,
            'test_type': 'mann_whitney',
            'statistic': np.nan,
            'pvalue': np.nan,
            'stars': 'NA',
            'n_wS2': len(wS2_means),
            'n_wM1': len(wM1_means)
        })

df_weight_mean_stats = pd.DataFrame(weight_mean_stats)
df_weight_mean_stats.to_csv(os.path.join(output_dir, 'weight_mean_values_statistics.csv'), index=False)
print(f"  Saved: weight_mean_values_statistics.csv")

df_weight_mean_stats_pos = pd.DataFrame(weight_mean_stats_pos)
df_weight_mean_stats_pos.to_csv(os.path.join(output_dir, 'weight_mean_values_statistics_positive.csv'), index=False)
print(f"  Saved: weight_mean_values_statistics_positive.csv")

df_weight_mean_stats_neg = pd.DataFrame(weight_mean_stats_neg)
df_weight_mean_stats_neg.to_csv(os.path.join(output_dir, 'weight_mean_values_statistics_negative.csv'), index=False)
print(f"  Saved: weight_mean_values_statistics_negative.csv")

# =============================================================================
# PART 5: CUMULATIVE DISTRIBUTION ANALYSIS (KS TEST)
# =============================================================================
# This analysis compares the full distributions of LMI/weights between wS2 and wM1
# without arbitrary thresholding. The Kolmogorov-Smirnov test quantifies whether
# wS2 cells tend to have systematically higher values than wM1 cells.
# =============================================================================

print("\nPART 5: CUMULATIVE DISTRIBUTION ANALYSIS")
print("-" * 80)

# Function to compute KS test and save results
def compute_ks_test_for_distributions(df, value_column, reward_groups, cell_types_to_compare,
                                     positive_only=False, negative_only=False):
    """
    Compute Kolmogorov-Smirnov test comparing distributions between cell types.

    Parameters:
    -----------
    df : DataFrame with cell-level data
    value_column : column name containing values to compare (e.g., 'lmi', 'classifier_weight')
    reward_groups : list of reward groups to analyze
    cell_types_to_compare : list of cell types to compare (e.g., ['wS2', 'wM1'])
    positive_only : if True, only consider positive values
    negative_only : if True, only consider negative values

    Returns:
    --------
    DataFrame with KS test results
    """
    results = []

    for reward_group in reward_groups:
        data_rg = df[df['reward_group'] == reward_group]

        # Apply filters for positive/negative
        if positive_only:
            data_rg = data_rg[data_rg[value_column] > 0]
        elif negative_only:
            data_rg = data_rg[data_rg[value_column] < 0]

        # Get distributions for each cell type
        wS2_values = data_rg[data_rg['cell_type_group'] == 'wS2'][value_column].values
        wM1_values = data_rg[data_rg['cell_type_group'] == 'wM1'][value_column].values

        if len(wS2_values) > 0 and len(wM1_values) > 0:
            # Two-sided KS test: tests if distributions are different
            statistic_2sided, pvalue_2sided = ks_2samp(wS2_values, wM1_values, alternative='two-sided')

            # One-sided KS test: tests if wS2 distribution is shifted toward higher values
            statistic_greater, pvalue_greater = ks_2samp(wS2_values, wM1_values, alternative='greater')

            # One-sided KS test: tests if wS2 distribution is shifted toward lower values
            statistic_less, pvalue_less = ks_2samp(wS2_values, wM1_values, alternative='less')

            results.append({
                'reward_group': reward_group,
                'value_type': 'positive' if positive_only else ('negative' if negative_only else 'all'),
                'n_wS2': len(wS2_values),
                'n_wM1': len(wM1_values),
                'mean_wS2': np.mean(wS2_values),
                'mean_wM1': np.mean(wM1_values),
                'median_wS2': np.median(wS2_values),
                'median_wM1': np.median(wM1_values),
                'ks_statistic_2sided': statistic_2sided,
                'ks_pvalue_2sided': pvalue_2sided,
                'ks_stars_2sided': pvalue_to_stars(pvalue_2sided),
                'ks_statistic_greater': statistic_greater,
                'ks_pvalue_greater': pvalue_greater,
                'ks_stars_greater': pvalue_to_stars(pvalue_greater),
                'ks_statistic_less': statistic_less,
                'ks_pvalue_less': pvalue_less,
                'ks_stars_less': pvalue_to_stars(pvalue_less)
            })
        else:
            results.append({
                'reward_group': reward_group,
                'value_type': 'positive' if positive_only else ('negative' if negative_only else 'all'),
                'n_wS2': len(wS2_values),
                'n_wM1': len(wM1_values),
                'mean_wS2': np.nan,
                'mean_wM1': np.nan,
                'median_wS2': np.nan,
                'median_wM1': np.nan,
                'ks_statistic_2sided': np.nan,
                'ks_pvalue_2sided': np.nan,
                'ks_stars_2sided': 'NA',
                'ks_statistic_greater': np.nan,
                'ks_pvalue_greater': np.nan,
                'ks_stars_greater': 'NA',
                'ks_statistic_less': np.nan,
                'ks_pvalue_less': np.nan,
                'ks_stars_less': 'NA'
            })

    return pd.DataFrame(results)

# LMI: KS test for all values
print("Computing KS test for LMI distributions...")
df_lmi_ks_all = compute_ks_test_for_distributions(
    lmi_df, 'lmi', ['R+', 'R-'], ['wS2', 'wM1']
)
df_lmi_ks_all.to_csv(os.path.join(output_dir, 'lmi_ks_test_all.csv'), index=False)
print(f"  Saved: lmi_ks_test_all.csv")

# LMI: KS test for positive values only
df_lmi_ks_pos = compute_ks_test_for_distributions(
    lmi_df, 'lmi', ['R+', 'R-'], ['wS2', 'wM1'], positive_only=True
)
df_lmi_ks_pos.to_csv(os.path.join(output_dir, 'lmi_ks_test_positive.csv'), index=False)
print(f"  Saved: lmi_ks_test_positive.csv")

# LMI: KS test for negative values only
df_lmi_ks_neg = compute_ks_test_for_distributions(
    lmi_df, 'lmi', ['R+', 'R-'], ['wS2', 'wM1'], negative_only=True
)
df_lmi_ks_neg.to_csv(os.path.join(output_dir, 'lmi_ks_test_negative.csv'), index=False)
print(f"  Saved: lmi_ks_test_negative.csv")

# Weights: KS test for all values
print("Computing KS test for weight distributions...")
df_weight_ks_all = compute_ks_test_for_distributions(
    weights_df, 'classifier_weight', ['R+', 'R-'], ['wS2', 'wM1']
)
df_weight_ks_all.to_csv(os.path.join(output_dir, 'weight_ks_test_all.csv'), index=False)
print(f"  Saved: weight_ks_test_all.csv")

# Weights: KS test for positive values only
df_weight_ks_pos = compute_ks_test_for_distributions(
    weights_df, 'classifier_weight', ['R+', 'R-'], ['wS2', 'wM1'], positive_only=True
)
df_weight_ks_pos.to_csv(os.path.join(output_dir, 'weight_ks_test_positive.csv'), index=False)
print(f"  Saved: weight_ks_test_positive.csv")

# Weights: KS test for negative values only
df_weight_ks_neg = compute_ks_test_for_distributions(
    weights_df, 'classifier_weight', ['R+', 'R-'], ['wS2', 'wM1'], negative_only=True
)
df_weight_ks_neg.to_csv(os.path.join(output_dir, 'weight_ks_test_negative.csv'), index=False)
print(f"  Saved: weight_ks_test_negative.csv")

# =============================================================================
# PART 6: VISUALIZATIONS (POOLED DATA ONLY)
# =============================================================================

print("\nPART 6: VISUALIZATIONS (POOLED DATA ONLY)")

# ====================================================================================
# Figure 1: LMI Significant - POOLED ONLY
# ====================================================================================
print("Creating Figure 1: LMI Significant (Pooled)...")

sig_data_pooled = df_lmi_props[df_lmi_props['criterion'].isin(['significant_pos_lmi', 'significant_neg_lmi'])]

# Create figure with 2 columns
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Pooled data
for row_idx, reward_group in enumerate(['R+', 'R-']):
    for col_idx, criterion in enumerate(['significant_pos_lmi', 'significant_neg_lmi']):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        plot_data = sig_data_pooled[
            (sig_data_pooled['reward_group'] == reward_group) &
            (sig_data_pooled['criterion'] == criterion)
        ]
        sns.barplot(data=plot_data, x='cell_type', y='proportion',
                   palette=cell_type_colors, order=cell_types,
                   ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5, hue='cell_type',)
        ax.set_xlabel('')
        ax.set_ylabel('Proportion' if col_idx == 0 else '')
        ax.set_ylim(0, 0.5)
        if row_idx == 0:
            ax.set_title(f"{'Positive' if 'pos' in criterion else 'Negative'} LMI", fontsize=11, fontweight='bold')

        # Add significance stars
        stat_result = df_lmi_stats_pooled[
            (df_lmi_stats_pooled['reward_group'] == reward_group) &
            (df_lmi_stats_pooled['criterion'] == criterion)
        ]
        if len(stat_result) > 0:
            stars = stat_result.iloc[0]['stars']
            ax.text(0.5, 0.47, stars, ha='center', va='center', fontsize=12, fontweight='bold',
                   transform=ax.transData)

fig.suptitle('Proportion of wS2 and wM1 Neurons with Significant LMI (Pooled)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_lmi_significant_pooled.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_lmi_significant_pooled.svg")

# ====================================================================================
# Figure 2: LMI Percentiles - Within Cell Type (POOLED, Top 20% Only)
# ====================================================================================
print("Creating Figure 2: LMI Percentiles Within (Pooled, Top 20% Only)...")

# Define criteria - only top 20%
lmi_pct_criteria = ['top_20pct_pos_lmi', 'top_20pct_neg_lmi']

# Create simple title mapping
lmi_pct_titles = {
    'top_20pct_pos_lmi': 'Top 20% Positive LMI',
    'top_20pct_neg_lmi': 'Top 20% Negative LMI'
}

pct_data = df_lmi_props[~df_lmi_props['criterion'].isin(['significant_pos_lmi', 'significant_neg_lmi'])]
g = sns.catplot(
    data=pct_data, x='cell_type', y='proportion',
    col='criterion', row='reward_group', col_order=lmi_pct_criteria,
    kind='bar', palette=cell_type_colors, order=cell_types,
    height=3, aspect=1.0, legend=False, edgecolor='black', linewidth=1.5, alpha=0.7, hue='cell_type',
)
g.set_titles("")  # Remove default titles
g.set_axis_labels("", "Proportion")
g.set(ylim=(0, 0.3))
# Set custom titles only for top row
for col_idx, criterion in enumerate(lmi_pct_criteria):
    g.axes[0, col_idx].set_title(lmi_pct_titles[criterion], fontsize=11, fontweight='bold')

# Add significance stars
for row_idx, reward_group in enumerate(['R+', 'R-']):
    for col_idx, criterion in enumerate(lmi_pct_criteria):
        stat_result = df_lmi_stats_pooled[
            (df_lmi_stats_pooled['reward_group'] == reward_group) &
            (df_lmi_stats_pooled['criterion'] == criterion)
        ]
        if len(stat_result) > 0:
            stars = stat_result.iloc[0]['stars']
            g.axes[row_idx, col_idx].text(0.5, 0.28, stars, ha='center', va='center',
                                          fontsize=10, fontweight='bold')

g.figure.suptitle('Proportion of Each Projection Neuron Type in Top 20% LMI\n(Within Cell Type - Pooled)',
               fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
g.savefig(os.path.join(output_dir, 'figure_lmi_percentile_within_pooled.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_lmi_percentile_within_pooled.svg")

# ====================================================================================
# Figure 3: LMI Percentiles - Composition (POOLED, Top 20% Only)
# ====================================================================================
print("Creating Figure 3: LMI Percentiles Composition (Pooled, Top 20% Only)...")

# Filter to only wS2 and wM1 for composition plots
pct_comp_data = df_lmi_comp[
    (~df_lmi_comp['criterion'].isin(['significant_pos_lmi', 'significant_neg_lmi'])) &
    (df_lmi_comp['cell_type'].isin(cell_types))  # Only wS2 and wM1
]
g = sns.catplot(
    data=pct_comp_data, x='cell_type', y='proportion',
    col='criterion', row='reward_group', col_order=lmi_pct_criteria,
    kind='bar', palette=cell_type_colors, order=cell_types,
    height=3, aspect=1.0, legend=False, edgecolor='black', linewidth=1.5, alpha=0.7, hue='cell_type',
)
g.set_titles("")
g.set_axis_labels("", "Proportion")
g.set(ylim=(0, 1.0))
for col_idx, criterion in enumerate(lmi_pct_criteria):
    g.axes[0, col_idx].set_title(lmi_pct_titles[criterion], fontsize=11, fontweight='bold')

# Add significance stars
for row_idx, reward_group in enumerate(['R+', 'R-']):
    for col_idx, criterion in enumerate(lmi_pct_criteria):
        stat_result = df_lmi_stats_pooled[
            (df_lmi_stats_pooled['reward_group'] == reward_group) &
            (df_lmi_stats_pooled['criterion'] == criterion)
        ]
        if len(stat_result) > 0:
            stars = stat_result.iloc[0]['stars']
            g.axes[row_idx, col_idx].text(0.5, 0.95, stars, ha='center', va='center',
                                          fontsize=10, fontweight='bold')

g.figure.suptitle('Cell Type Composition of Top 20% LMI\n(Of Top 20% Cells, what % are wS2 vs wM1 - Pooled)',
               fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
g.savefig(os.path.join(output_dir, 'figure_lmi_percentile_composition_pooled.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_lmi_percentile_composition_pooled.svg")

# ====================================================================================
# Figure 4: Weight Percentiles - Within Cell Type (POOLED, Top 20% Only)
# ====================================================================================
print("Creating Figure 4: Weight Percentiles Within (Pooled, Top 20% Only)...")

# Create simple title mapping for weight criteria - only top 20%
weight_pct_titles = {
    'top_20pct_pos_weight': 'Top 20% Positive Weight',
    'top_20pct_neg_weight': 'Top 20% Negative Weight'
}

g = sns.catplot(
    data=df_weight_props, x='cell_type', y='proportion',
    col='criterion', row='reward_group', col_order=weight_criteria,
    kind='bar', palette=cell_type_colors, order=cell_types,
    height=3, aspect=1.0, legend=False, edgecolor='black', linewidth=1.5, alpha=0.7, hue='cell_type',
)
g.set_titles("")
g.set_axis_labels("", "Proportion")
g.set(ylim=(0, 0.3))
for col_idx, criterion in enumerate(weight_criteria):
    g.axes[0, col_idx].set_title(weight_pct_titles[criterion], fontsize=11, fontweight='bold')

# Add significance stars
for row_idx, reward_group in enumerate(['R+', 'R-']):
    for col_idx, criterion in enumerate(weight_criteria):
        stat_result = df_weight_stats_pooled[
            (df_weight_stats_pooled['reward_group'] == reward_group) &
            (df_weight_stats_pooled['criterion'] == criterion)
        ]
        if len(stat_result) > 0:
            stars = stat_result.iloc[0]['stars']
            g.axes[row_idx, col_idx].text(0.5, 0.28, stars, ha='center', va='center',
                                          fontsize=10, fontweight='bold')

g.figure.suptitle('Proportion of Each Projection Neuron Type in Top 20% Decoder Weights\n(Within Cell Type - Pooled)',
               fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
g.savefig(os.path.join(output_dir, 'figure_weight_percentile_within_pooled.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_weight_percentile_within_pooled.svg")

# ====================================================================================
# Figure 5: Weight Percentiles - Composition (POOLED, Top 20% Only)
# ====================================================================================
print("Creating Figure 5: Weight Percentiles Composition (Pooled, Top 20% Only)...")

# Filter to only wS2 and wM1 for composition plots
weight_comp_data = df_weight_comp[df_weight_comp['cell_type'].isin(cell_types)]
g = sns.catplot(
    data=weight_comp_data, x='cell_type', y='proportion',
    col='criterion', row='reward_group', col_order=weight_criteria,
    kind='bar', palette=cell_type_colors, order=cell_types,
    height=3, aspect=1.0, legend=False, edgecolor='black', linewidth=1.5, alpha=0.7, hue='cell_type',
)
g.set_titles("")
g.set_axis_labels("", "Proportion")
g.set(ylim=(0, 0.5))
for col_idx, criterion in enumerate(weight_criteria):
    g.axes[0, col_idx].set_title(weight_pct_titles[criterion], fontsize=11, fontweight='bold')

# Add significance stars
for row_idx, reward_group in enumerate(['R+', 'R-']):
    for col_idx, criterion in enumerate(weight_criteria):
        stat_result = df_weight_stats_pooled[
            (df_weight_stats_pooled['reward_group'] == reward_group) &
            (df_weight_stats_pooled['criterion'] == criterion)
        ]
        if len(stat_result) > 0:
            stars = stat_result.iloc[0]['stars']
            g.axes[row_idx, col_idx].text(0.5, 0.47, stars, ha='center', va='center',
                                          fontsize=10, fontweight='bold')

g.figure.suptitle('Cell Type Composition of Top 20% Decoder Weights\n(Of Top 20% Cells, what % are wS2 vs wM1 - Pooled)',
               fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
g.savefig(os.path.join(output_dir, 'figure_weight_percentile_composition_pooled.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_weight_percentile_composition_pooled.svg")

# ====================================================================================
# Figure 6: Cell Counts Summary
# ====================================================================================
print("Creating Figure 6: Cell Counts Summary...")

# Calculate total counts and per-mouse means
total_counts = []
per_mouse_counts = []

for cell_type in cell_types:
    total_lmi = len(lmi_df[lmi_df['cell_type_group'] == cell_type])
    counts_per_mouse = lmi_df[lmi_df['cell_type_group'] == cell_type].groupby('mouse_id').size()
    mean_per_mouse = counts_per_mouse.mean()
    sem_per_mouse = counts_per_mouse.sem()

    total_counts.append({'cell_type': cell_type, 'count': total_lmi})
    per_mouse_counts.append({'cell_type': cell_type, 'mean': mean_per_mouse, 'sem': sem_per_mouse})

df_total = pd.DataFrame(total_counts)
df_per_mouse = pd.DataFrame(per_mouse_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Total and Per-Mouse Counts of Projection Neuron Types', fontsize=14, fontweight='bold')

# Left: Total counts
sns.barplot(data=df_total, x='cell_type', y='count', palette=cell_type_colors, order=cell_types,
            ax=ax1, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_title('Total Counts (All Mice)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Cell Type', fontsize=11)
ax1.set_ylabel('Total Count', fontsize=11)

# Right: Mean per mouse with error bars
ax2.bar(range(len(cell_types)), df_per_mouse['mean'], yerr=df_per_mouse['sem'],
        color=[cell_type_colors[ct] for ct in cell_types],
        alpha=0.7, edgecolor='black', linewidth=1.5,
        error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 5})
ax2.set_title('Mean Counts Per Mouse', fontsize=12, fontweight='bold')
ax2.set_xlabel('Cell Type', fontsize=11)
ax2.set_ylabel('Mean Count Per Mouse ± SEM', fontsize=11)
ax2.set_xticks(range(len(cell_types)))
ax2.set_xticklabels(cell_types)

plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_cell_counts.svg'), format='svg', dpi=300, bbox_inches='tight')
# plt.close()
print("  Saved: figure_cell_counts.svg")



# ====================================================================================
# Figure 7a: Average Positive LMI and Positive Classifier Weight Values (wS2 vs wM1)
# ====================================================================================
print("Creating Figure 7a: Average Positive LMI and Positive Classifier Weight Values...")

# Use the correctly computed positive-only dataframes (no need to filter and take abs)
# The values are already positive and computed from positive cells only
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: Average positive LMI values
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])
    plot_data = df_lmi_means_pos[df_lmi_means_pos['reward_group'] == reward_group]

    sns.barplot(data=plot_data, x='cell_type', y='mean_lmi',
               palette=cell_type_colors, order=cell_types,
               ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5,
               errorbar='ci', err_kws={'linewidth': 2}, hue='cell_type')

    ax.set_xlabel('Cell Type' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Mean Positive LMI', fontsize=11)
    ax.set_title(f'{reward_group} - Avg Positive LMI' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')

    # Add significance stars - use the correct stats dataframe
    stat_result = df_lmi_mean_stats_pos[df_lmi_mean_stats_pos['reward_group'] == reward_group]
    if len(stat_result) > 0:
        stars = stat_result.iloc[0]['stars']
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95
        ax.text(0.5, y_pos, stars, ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transData)

# Panel 2: Average positive Weight values
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])
    plot_data = df_weight_means_pos[df_weight_means_pos['reward_group'] == reward_group]

    sns.barplot(data=plot_data, x='cell_type', y='mean_weight',
               palette=cell_type_colors, order=cell_types,
               ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5,
               errorbar='ci', err_kws={'linewidth': 2}, hue='cell_type')

    ax.set_xlabel('Cell Type' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Mean Positive Classifier Weight', fontsize=11)
    ax.set_title(f'{reward_group} - Avg Positive Weights' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')

    # Add significance stars - use the correct stats dataframe
    stat_result = df_weight_mean_stats_pos[df_weight_mean_stats_pos['reward_group'] == reward_group]
    if len(stat_result) > 0:
        stars = stat_result.iloc[0]['stars']
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95
        ax.text(0.5, y_pos, stars, ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transData)

fig.suptitle('Average Positive LMI and Classifier Weight Values: wS2 vs wM1\n(Mean of per-mouse means, computed from positive cells only)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_average_positive_values.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_average_positive_values.svg")

# ====================================================================================
# Figure 7b: Average Negative LMI and Negative Classifier Weight Values (wS2 vs wM1)
# ====================================================================================
print("Creating Figure 7b: Average Negative LMI and Negative Classifier Weight Values...")

# Use the correctly computed negative-only dataframes
# Take absolute values for display purposes (to show magnitude)
df_lmi_means_neg_plot = df_lmi_means_neg.copy()
df_lmi_means_neg_plot['mean_lmi'] = df_lmi_means_neg_plot['mean_lmi'].abs()
df_weight_means_neg_plot = df_weight_means_neg.copy()
df_weight_means_neg_plot['mean_weight'] = df_weight_means_neg_plot['mean_weight'].abs()

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: Average negative LMI values (absolute)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])
    plot_data = df_lmi_means_neg_plot[df_lmi_means_neg_plot['reward_group'] == reward_group]

    sns.barplot(data=plot_data, x='cell_type', y='mean_lmi',
               palette=cell_type_colors, order=cell_types,
               ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5,
               errorbar='ci', err_kws={'linewidth': 2}, hue='cell_type')

    ax.set_xlabel('Cell Type' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Mean Negative LMI (abs)', fontsize=11)
    ax.set_title(f'{reward_group} - Avg Negative LMI' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')

    # Add significance stars - use the correct stats dataframe
    stat_result = df_lmi_mean_stats_neg[df_lmi_mean_stats_neg['reward_group'] == reward_group]
    if len(stat_result) > 0:
        stars = stat_result.iloc[0]['stars']
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95
        ax.text(0.5, y_pos, stars, ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transData)

# Panel 2: Average negative Weight values (absolute)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])
    plot_data = df_weight_means_neg_plot[df_weight_means_neg_plot['reward_group'] == reward_group]

    sns.barplot(data=plot_data, x='cell_type', y='mean_weight',
               palette=cell_type_colors, order=cell_types,
               ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5,
               errorbar='ci', err_kws={'linewidth': 2}, hue='cell_type')

    ax.set_xlabel('Cell Type' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Mean Negative Classifier Weight (abs)', fontsize=11)
    ax.set_title(f'{reward_group} - Avg Negative Weights' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')

    # Add significance stars - use the correct stats dataframe
    stat_result = df_weight_mean_stats_neg[df_weight_mean_stats_neg['reward_group'] == reward_group]
    if len(stat_result) > 0:
        stars = stat_result.iloc[0]['stars']
        ylim = ax.get_ylim()
        y_pos = ylim[1] * 0.95
        ax.text(0.5, y_pos, stars, ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transData)

fig.suptitle('Average Negative LMI and Classifier Weight Values (Absolute): wS2 vs wM1\n(Mean of per-mouse means, computed from negative cells only)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_average_negative_values.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_average_negative_values.svg")

# ====================================================================================
# Figure 8: Cumulative Distribution Functions - LMI (Positive and Negative)
# ====================================================================================
print("Creating Figure 8: Cumulative Distribution Functions - LMI...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# LMI: Positive values only
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])

    data_rg = lmi_df[lmi_df['reward_group'] == reward_group]
    data_pos = data_rg[data_rg['lmi'] > 0]

    for cell_type in cell_types:
        values = data_pos[data_pos['cell_type_group'] == cell_type]['lmi'].values
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                   color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('LMI Value' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Positive LMI' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Add KS test result
    ks_result = df_lmi_ks_pos[df_lmi_ks_pos['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars_2sided = ks_result.iloc[0]['ks_stars_2sided']
        stars_greater = ks_result.iloc[0]['ks_stars_greater']
        pval_greater = ks_result.iloc[0]['ks_pvalue_greater']
        ax.text(0.98, 0.02, f'KS test (2-sided): {stars_2sided}\nKS test (wS2>wM1): {stars_greater} (p={pval_greater:.4f})',
               ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# LMI: Negative values only (plot absolute values)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])

    data_rg = lmi_df[lmi_df['reward_group'] == reward_group]
    data_neg = data_rg[data_rg['lmi'] < 0]

    for cell_type in cell_types:
        values = np.abs(data_neg[data_neg['cell_type_group'] == cell_type]['lmi'].values)
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                   color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('|LMI| Value' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Negative LMI (abs)' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Add KS test result
    ks_result = df_lmi_ks_neg[df_lmi_ks_neg['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars_2sided = ks_result.iloc[0]['ks_stars_2sided']
        stars_greater = ks_result.iloc[0]['ks_stars_greater']
        pval_greater = ks_result.iloc[0]['ks_pvalue_greater']
        ax.text(0.98, 0.02, f'KS test (2-sided): {stars_2sided}\nKS test (wS2>wM1): {stars_greater} (p={pval_greater:.4f})',
               ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Cumulative Distribution Functions: LMI (wS2 vs wM1)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_cdf_lmi.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_cdf_lmi.svg")

# ====================================================================================
# Figure 9: Cumulative Distribution Functions - Classifier Weights (Positive and Negative)
# ====================================================================================
print("Creating Figure 9: Cumulative Distribution Functions - Classifier Weights...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Weights: Positive values only
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 0])

    data_rg = weights_df[weights_df['reward_group'] == reward_group]
    data_pos = data_rg[data_rg['classifier_weight'] > 0]

    for cell_type in cell_types:
        values = data_pos[data_pos['cell_type_group'] == cell_type]['classifier_weight'].values
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                   color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('Classifier Weight' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Positive Weights' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Add KS test result
    ks_result = df_weight_ks_pos[df_weight_ks_pos['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars_2sided = ks_result.iloc[0]['ks_stars_2sided']
        stars_greater = ks_result.iloc[0]['ks_stars_greater']
        pval_greater = ks_result.iloc[0]['ks_pvalue_greater']
        ax.text(0.98, 0.02, f'KS test (2-sided): {stars_2sided}\nKS test (wS2>wM1): {stars_greater} (p={pval_greater:.4f})',
               ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Weights: Negative values only (plot absolute values)
for row_idx, reward_group in enumerate(['R+', 'R-']):
    ax = fig.add_subplot(gs[row_idx, 1])

    data_rg = weights_df[weights_df['reward_group'] == reward_group]
    data_neg = data_rg[data_rg['classifier_weight'] < 0]

    for cell_type in cell_types:
        values = np.abs(data_neg[data_neg['cell_type_group'] == cell_type]['classifier_weight'].values)
        if len(values) > 0:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=cell_type,
                   color=cell_type_colors[cell_type], linewidth=2, alpha=0.8)

    ax.set_xlabel('|Classifier Weight|' if row_idx == 1 else '', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{reward_group} - Negative Weights (abs)' if row_idx == 0 else f'{reward_group}',
                fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Add KS test result
    ks_result = df_weight_ks_neg[df_weight_ks_neg['reward_group'] == reward_group]
    if len(ks_result) > 0:
        stars_2sided = ks_result.iloc[0]['ks_stars_2sided']
        stars_greater = ks_result.iloc[0]['ks_stars_greater']
        pval_greater = ks_result.iloc[0]['ks_pvalue_greater']
        ax.text(0.98, 0.02, f'KS test (2-sided): {stars_2sided}\nKS test (wS2>wM1): {stars_greater} (p={pval_greater:.4f})',
               ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('Cumulative Distribution Functions: Classifier Weights (wS2 vs wM1)',
             fontsize=14, fontweight='bold', y=0.98)
sns.despine()
plt.savefig(os.path.join(output_dir, 'figure_cdf_weights.svg'), format='svg', dpi=300, bbox_inches='tight')
print("  Saved: figure_cdf_weights.svg")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
