"""
TCA Analysis of Day 0 Learning Plasticity using Tensorly

Uses tensorly's CP (CANDECOMP/PARAFAC) decomposition to decompose day 0 neural
activity into a small number of interpretable components.

Simpler and faster than sliceTCA - good starting point for exploration.

Author: Claude Code
Date: 2025-11-24
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import scipy.ndimage as spnd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
import tensorly as tl
from tensorly.decomposition import parafac
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import utils_io as io
from utils import utils_imaging

# Plotting parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_theme(context='talk', style='ticks', palette='deep')

# Color palettes
palette = sns.color_palette()
component_colors = sns.color_palette("Set2", 8)


# =============================================================================
# FUNCTION: Load Day 0 tensor
# =============================================================================

def load_day0_tensor(mouse_list, cell_list_dict, reward_dict, trial_type='whisker',
                     time_window=(-0.5, 5.0), baseline_subtract=True):
    """
    Load day 0 data and construct tensor: Neurons × Time × Trials.

    Parameters
    ----------
    mouse_list : list
        List of mouse IDs (should be ordered: R+ mice first, then R- mice)
    cell_list_dict : dict
        {mouse_id: [roi1, roi2, ...]}
    reward_dict : dict
        {mouse_id: 'R+' or 'R-'}
    trial_type : str
        'all', 'hits', 'misses', or 'whisker'
    time_window : tuple
        (start, end) in seconds
    baseline_subtract : bool
        Whether to use baseline-subtracted data

    Returns
    -------
    tensor : ndarray
        Shape (n_neurons, n_time, n_trials)
    metadata : dict
        Information about the data
    """
    print(f"\n{'='*70}")
    print(f"Loading Day 0 Data - Trial Type: {trial_type}")
    print(f"{'='*70}\n")

    folder = os.path.join(io.processed_dir, 'mice')
    all_data = []
    cell_info = []
    trial_info_list = []

    for mouse_id in mouse_list:
        if mouse_id not in cell_list_dict or len(cell_list_dict[mouse_id]) == 0:
            continue

        print(f"  Loading {mouse_id}...")

        try:
            # Load stimulus-aligned data
            file_name = 'tensor_xarray_learning_data.nc'
            xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name,
                                                   substracted=baseline_subtract)

            # Select day 0 only
            xarr = xarr.sel(trial=xarr['day'] == 0)

            # Select cells for this mouse
            cell_list = cell_list_dict[mouse_id]
            xarr = xarr.sel(cell=xarr['roi'].isin(cell_list))

            # Filter by trial type
            if trial_type == 'hits':
                xarr = xarr.sel(trial=(xarr['whisker_stim'] == 1) & (xarr['lick_flag'] == 1))
            elif trial_type == 'misses':
                xarr = xarr.sel(trial=(xarr['whisker_stim'] == 1) & (xarr['lick_flag'] == 0))
            elif trial_type == 'whisker':
                xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)

            # Select time window
            xarr = xarr.sel(time=slice(time_window[0], time_window[1]))

            if len(xarr.trial) == 0 or len(xarr.cell) == 0:
                print(f"    No data for {mouse_id}, skipping")
                continue

            # Store trial info
            for trial_idx in range(len(xarr.trial)):
                trial_info = {
                    'mouse_id': mouse_id,
                    'trial_in_session': trial_idx,
                    'lick_flag': int(xarr['lick_flag'].isel(trial=trial_idx).values),
                    'whisker_stim': int(xarr['whisker_stim'].isel(trial=trial_idx).values),
                }
                trial_info_list.append(trial_info)

            # Convert to numpy: (n_cells, n_trials, n_time)
            data = xarr.values
            # Transpose to (n_cells, n_time, n_trials)
            data = data.transpose(0, 2, 1)

            all_data.append(data)

            # Store cell info
            for roi in xarr['roi'].values:
                cell_info.append({
                    'mouse_id': mouse_id,
                    'roi': int(roi),
                    'cell_id': f"{mouse_id}_{int(roi)}",
                    'reward_group': reward_dict[mouse_id]
                })

            print(f"    Added {len(xarr.cell)} cells, {len(xarr.trial)} trials")

        except Exception as e:
            print(f"    Error loading {mouse_id}: {e}")
            continue

    if len(all_data) == 0:
        raise ValueError("No data loaded!")

    # Concatenate across mice
    # Pad trials to max_trials
    max_trials = max([d.shape[2] for d in all_data])

    padded_data = []
    for data in all_data:
        n_cells, n_time, n_trials = data.shape
        if n_trials < max_trials:
            padding = np.full((n_cells, n_time, max_trials - n_trials), np.nan)
            data = np.concatenate([data, padding], axis=2)
        padded_data.append(data)

    # Concatenate along cell dimension: (n_cells_total, n_time, max_trials)
    tensor = np.concatenate(padded_data, axis=0)

    # Remove trials with NaN values
    nan_trials = np.any(np.isnan(tensor), axis=(0, 1))
    tensor = tensor[:, :, ~nan_trials]

    # Get time axis
    time_axis = xarr['time'].values

    # Create metadata
    metadata = {
        'cell_info': pd.DataFrame(cell_info),
        'trial_info': pd.DataFrame(trial_info_list),
        'time_axis': time_axis,
        'n_neurons': tensor.shape[0],
        'n_time': tensor.shape[1],
        'n_trials': tensor.shape[2],
        'trial_type': trial_type,
        'time_window': time_window
    }

    print(f"\n  Final tensor shape: {tensor.shape} (neurons, time, trials)")
    print(f"  Neurons: {metadata['n_neurons']}")
    print(f"  Time points: {metadata['n_time']}")
    print(f"  Trials: {metadata['n_trials']}")

    return tensor, metadata


# =============================================================================
# FUNCTION: Prepare tensor for TCA
# =============================================================================

def prepare_tensor_for_tca(tensor, metadata, smooth_sigma=2, normalize=True):
    """
    Prepare tensor for TCA analysis.

    Parameters
    ----------
    tensor : ndarray
        Raw tensor (neurons, time, trials)
    metadata : dict
        Metadata
    smooth_sigma : float
        Gaussian smoothing sigma for temporal smoothing
    normalize : bool
        Whether to z-score normalize

    Returns
    -------
    tensor_prepared : ndarray
        Prepared tensor
    """
    print(f"\n{'='*70}")
    print(f"Preparing Tensor for TCA")
    print(f"{'='*70}\n")

    tensor_prep = tensor.copy()

    # Gaussian smoothing along time axis (axis=1)
    if smooth_sigma > 0:
        print(f"  Applying Gaussian smoothing (sigma={smooth_sigma})...")
        tensor_prep = spnd.gaussian_filter1d(tensor_prep.astype('float32'),
                                            sigma=smooth_sigma, axis=1)

    # Z-score normalization (for each neuron across time and trials)
    if normalize:
        print(f"  Z-scoring neurons...")
        # Calculate mean and std across time and trials for each neuron
        for neuron_idx in range(tensor_prep.shape[0]):
            neuron_data = tensor_prep[neuron_idx, :, :].flatten()
            mean_val = np.nanmean(neuron_data)
            std_val = np.nanstd(neuron_data)
            if std_val > 0:
                tensor_prep[neuron_idx, :, :] = (tensor_prep[neuron_idx, :, :] - mean_val) / std_val

    print(f"  Final tensor shape: {tensor_prep.shape}")
    print(f"  Value range: [{np.nanmin(tensor_prep):.3f}, {np.nanmax(tensor_prep):.3f}]")

    return tensor_prep


# =============================================================================
# FUNCTION: Fit TCA (CP decomposition)
# =============================================================================

def fit_tca(tensor, n_components=5, max_iter=100, tol=1e-6, init='random', random_state=0):
    """
    Fit TCA using tensorly's CP decomposition (PARAFAC).

    Parameters
    ----------
    tensor : ndarray
        Prepared tensor (neurons, time, trials)
    n_components : int
        Number of components
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    init : str
        Initialization method ('random' or 'svd')
    random_state : int
        Random seed

    Returns
    -------
    factors : list
        [neuron_factors, time_factors, trial_factors]
    errors : list
        Reconstruction errors per iteration
    """
    print(f"\n{'='*70}")
    print(f"Fitting TCA (CP Decomposition)")
    print(f"  Components: {n_components}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Initialization: {init}")
    print(f"{'='*70}\n")

    # Set random seed
    np.random.seed(random_state)

    # Fit CP decomposition
    # PARAFAC returns (weights, factors) where factors is a list [mode0, mode1, mode2]
    weights, factors = parafac(tensor, rank=n_components,
                               n_iter_max=max_iter,
                               init=init,
                               tol=tol,
                               random_state=random_state,
                               return_errors=False)

    print(f"\n  Decomposition completed!")
    print(f"  Factor shapes:")
    print(f"    Neuron factors: {factors[0].shape}")
    print(f"    Time factors: {factors[1].shape}")
    print(f"    Trial factors: {factors[2].shape}")
    print(f"    Weights: {weights.shape}")

    return weights, factors


# =============================================================================
# FUNCTION: Analyze components
# =============================================================================

def analyze_components(weights, factors, metadata):
    """
    Analyze TCA components to identify learning-related patterns.

    Returns
    -------
    component_stats : pd.DataFrame
        Statistics for each component
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Components")
    print(f"{'='*70}\n")

    neuron_factors = factors[0]  # (n_neurons, n_components)
    time_factors = factors[1]    # (n_time, n_components)
    trial_factors = factors[2]   # (n_trials, n_components)

    n_components = len(weights)
    stats_list = []

    time_axis = metadata['time_axis']

    for comp_idx in range(n_components):
        # Get factors for this component
        neuron_factor = neuron_factors[:, comp_idx]
        time_factor = time_factors[:, comp_idx]
        trial_factor = trial_factors[:, comp_idx]
        weight = weights[comp_idx]

        # Temporal statistics
        peak_time_idx = np.argmax(time_factor)
        peak_time = time_axis[peak_time_idx]
        temporal_class = 'early' if peak_time < 0.3 else 'late'

        # Trial progression statistics
        trial_numbers = np.arange(len(trial_factor))
        correlation, p_value = stats.spearmanr(trial_numbers, trial_factor)

        # Is it learning-related?
        is_learning_related = (p_value < 0.05) and (abs(correlation) > 0.2)
        learning_direction = 'increasing' if correlation > 0 else 'decreasing'

        # Neuron statistics
        neuron_threshold = 0.1
        n_neurons_strong = np.sum(neuron_factor > neuron_threshold)

        stats_list.append({
            'component': comp_idx,
            'weight': weight,
            'peak_time': peak_time,
            'temporal_class': temporal_class,
            'trial_correlation': correlation,
            'trial_correlation_p': p_value,
            'is_learning_related': is_learning_related,
            'learning_direction': learning_direction if is_learning_related else 'none',
            'n_neurons_strong': n_neurons_strong,
        })

        print(f"  Component {comp_idx} (weight={weight:.3f}):")
        print(f"    Peak time: {peak_time:.3f}s ({temporal_class})")
        print(f"    Trial correlation: {correlation:.3f} (p={p_value:.4f})")
        print(f"    Learning-related: {is_learning_related}")
        print()

    component_stats = pd.DataFrame(stats_list)
    return component_stats


# =============================================================================
# FUNCTION: Cluster cells
# =============================================================================

def cluster_cells_by_components(factors, metadata, n_clusters=4):
    """
    Cluster cells based on their component loadings.

    Returns
    -------
    cell_clusters : pd.DataFrame
        Cell information with cluster assignments
    """
    print(f"\n{'='*70}")
    print(f"Clustering Cells into Subpopulations")
    print(f"{'='*70}\n")

    # Get neuron factors
    neuron_factors = factors[0]

    # Hierarchical clustering
    linkage_matrix = linkage(neuron_factors, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Add to cell info
    cell_info = metadata['cell_info'].copy()
    cell_info['cluster'] = cluster_labels

    # Characterize each cluster
    print(f"  Cluster sizes:")
    for cluster_id in range(1, n_clusters + 1):
        n_cells = np.sum(cluster_labels == cluster_id)
        cluster_cells = cluster_labels == cluster_id
        cluster_loadings = neuron_factors[cluster_cells, :].mean(axis=0)
        dominant_comp = np.argmax(cluster_loadings)
        print(f"    Cluster {cluster_id}: {n_cells} cells, dominant component: {dominant_comp}")

    return cell_info


# =============================================================================
# FUNCTION: Plot TCA results
# =============================================================================

def plot_tca_overview(weights, factors, metadata, component_stats, reward_filter='all', figsize=(20, 12)):
    """
    Create overview figure showing all TCA components.

    Neurons are ordered consistently across all components, grouped by their
    dominant component to reveal segregation vs. overlap.

    Parameters
    ----------
    reward_filter : str
        'all', 'R+', or 'R-' to indicate which reward group is being analyzed
    """
    neuron_factors = factors[0]
    time_factors = factors[1]
    trial_factors = factors[2]

    n_components = len(weights)
    n_neurons = neuron_factors.shape[0]

    fig, axes = plt.subplots(n_components, 3, figsize=figsize)
    if n_components == 1:
        axes = axes.reshape(1, -1)

    time_axis = metadata['time_axis']
    trial_axis = np.arange(trial_factors.shape[0])

    # =========================================================================
    # Keep neurons in original order (already sorted by mice: R+ then R-)
    # Create color array based on reward group
    # =========================================================================

    # Get reward group for each neuron from metadata
    reward_groups = metadata['cell_info']['reward_group'].values

    # Create color array: green for R+, magenta for R-
    neuron_colors = np.array(['green' if rg == 'R+' else 'magenta' for rg in reward_groups])

    # Find boundary between R+ and R- mice
    reward_boundary = np.where(reward_groups == 'R-')[0]
    if len(reward_boundary) > 0:
        reward_boundary = reward_boundary[0]
    else:
        reward_boundary = None

    # =========================================================================
    # Plot components
    # =========================================================================

    for comp_idx in range(n_components):
        stats_row = component_stats.iloc[comp_idx]
        is_learning = stats_row['is_learning_related']
        color = component_colors[comp_idx % len(component_colors)]

        # Plot temporal factor
        ax = axes[comp_idx, 0]
        ax.plot(time_axis, time_factors[:, comp_idx], color=color, linewidth=2)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(stats_row['peak_time'], color=color, linestyle=':', alpha=0.5)
        ax.set_ylabel(f'Component {comp_idx}\n(w={stats_row["weight"]:.2f})', fontweight='bold')
        ax.set_title(f'Temporal Profile ({stats_row["temporal_class"]})')
        if comp_idx == n_components - 1:
            ax.set_xlabel('Time (s)')

        # Plot trial factor
        ax = axes[comp_idx, 1]
        ax.plot(trial_axis, trial_factors[:, comp_idx], color=color, linewidth=2, alpha=0.7)
        ax.scatter(trial_axis, trial_factors[:, comp_idx], color=color, s=10, alpha=0.5)

        # Add trend line if learning-related
        if is_learning:
            z = np.polyfit(trial_axis, trial_factors[:, comp_idx], 1)
            p = np.poly1d(z)
            ax.plot(trial_axis, p(trial_axis), "r--", alpha=0.8, linewidth=2,
                   label=f'r={stats_row["trial_correlation"]:.2f}')
            ax.legend()

        ax.set_title(f'Trial Progression ({"Learning" if is_learning else "Stable"})')
        if comp_idx == n_components - 1:
            ax.set_xlabel('Trial number')

        # Plot neuron factor - BARS WITH SHARED X-AXIS
        ax = axes[comp_idx, 2]

        # Get neuron loadings in ORIGINAL ORDER (actual weights, not absolute)
        neuron_loadings = neuron_factors[:, comp_idx]

        # Create bar plot with reward group colors
        x_pos = np.arange(n_neurons)
        ax.bar(x_pos, neuron_loadings, color=neuron_colors, alpha=0.7, width=1.0, edgecolor='none')

        # Add vertical line at R+/R- boundary
        if reward_boundary is not None:
            ax.axvline(reward_boundary - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
            # Add labels
            if comp_idx == 0:
                ax.text(reward_boundary/2, ax.get_ylim()[1], 'R+',
                       ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
                ax.text(reward_boundary + (n_neurons-reward_boundary)/2, ax.get_ylim()[1], 'R-',
                       ha='center', va='bottom', fontsize=10, color='magenta', fontweight='bold')

        # Add zero line
        ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

        # Formatting
        ax.set_ylabel('Weight', fontsize=10)
        if reward_filter == 'all':
            ax.set_title(f'Neuron Weights (R+ mice | R- mice)')
        elif reward_filter == 'R+':
            ax.set_title(f'Neuron Weights (R+ mice only)')
        else:  # R-
            ax.set_title(f'Neuron Weights (R- mice only)')

        # Share x-axis across all components
        if comp_idx < n_components - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Neurons (grouped by mice & reward)')
            # Show fewer x-ticks for readability
            n_ticks = min(10, n_neurons)
            tick_positions = np.linspace(0, n_neurons-1, n_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions, fontsize=8)

        ax.set_xlim(-0.5, n_neurons-0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # ========================================================================
    # PARAMETER: Which reward group to analyze
    # Options: 'all', 'R+', 'R-'
    # ========================================================================
    REWARD_GROUP_FILTER = 'R+'  # Change to 'R+' or 'R-' to analyze specific group

    print("\n" + "="*70)
    print(f"TCA ANALYSIS OF DAY 0 LEARNING PLASTICITY (Tensorly CP)")
    print(f"Reward group filter: {REWARD_GROUP_FILTER}")
    print("="*70)

    # Load session database to get reward group info
    print("\nLoading session database...")
    _, _, all_mice, db = io.select_sessions_from_db(
        io.db_path,
        io.nwb_dir,
        two_p_imaging='yes',
        experimenters=['AR', 'GF', 'MI']
    )
    mice_info = db[['mouse_id', 'reward_group']].drop_duplicates()

    # Create reward group dictionary
    reward_dict = dict(zip(mice_info['mouse_id'], mice_info['reward_group']))

    # Load LMI data to get cell lists
    print("Loading LMI data...")
    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

    # Get ALL cells for each mouse (not just positive LMI)
    # Only include mice that are in the database
    cell_list_dict = {}
    for mouse_id in lmi_df['mouse_id'].unique():
        if mouse_id in reward_dict:  # Only include mice with reward group info
            mouse_cells = lmi_df[lmi_df['mouse_id'] == mouse_id]
            cell_list_dict[mouse_id] = mouse_cells['roi'].tolist()

    # Sort mice by reward group (R+ first, then R-)
    mouse_list_R_plus = [m for m in cell_list_dict.keys() if reward_dict[m] == 'R+']
    mouse_list_R_minus = [m for m in cell_list_dict.keys() if reward_dict[m] == 'R-']

    # Apply reward group filter
    if REWARD_GROUP_FILTER == 'R+':
        mouse_list = mouse_list_R_plus
        print(f"\n  Filtering to R+ mice only")
    elif REWARD_GROUP_FILTER == 'R-':
        mouse_list = mouse_list_R_minus
        print(f"\n  Filtering to R- mice only")
    else:
        mouse_list = mouse_list_R_plus + mouse_list_R_minus
        print(f"\n  Using all mice (R+ and R-)")
        
    # Mouse AR177 seemsto have specific components. Exclude for now.
    if 'AR177' in mouse_list:
        mouse_list.remove('AR177')
        print(f"  Excluding mouse AR177 due to specific components") 

    print(f"  R+ Mice ({len(mouse_list_R_plus)}): {mouse_list_R_plus}")
    print(f"  R- Mice ({len(mouse_list_R_minus)}): {mouse_list_R_minus}")
    print(f"  Selected mice ({len(mouse_list)}): {mouse_list}")
    print(f"  Total cells (ALL cells, not just LMI+): {sum([len(cells) for mouse_id, cells in cell_list_dict.items() if mouse_id in mouse_list])}")

    # Define output path with reward group indicator
    output_dir = io.results_dir
    if REWARD_GROUP_FILTER == 'all':
        suffix = 'all'
    else:
        suffix = REWARD_GROUP_FILTER.replace('+', 'plus').replace('-', 'minus')
    pdf_path = os.path.join(output_dir, f'tca_day0_tensorly_{suffix}.pdf')

    # Create PDF
    with PdfPages(pdf_path) as pdf:

        # =================================================================
        # Analysis for Whisker Trials
        # =================================================================

        print("\n" + "="*70)
        print("ANALYSIS: WHISKER TRIALS (DAY 0)")
        print("="*70)

        # Load data
        tensor, metadata = load_day0_tensor(
            mouse_list, cell_list_dict, reward_dict,
            trial_type='whisker',
            time_window=(-0.5, 4.0),
            baseline_subtract=True
        )

        # Prepare tensor
        tensor_prep = prepare_tensor_for_tca(tensor, metadata,
                                            smooth_sigma=2,
                                            normalize=True)

        # Fit TCA
        weights, factors = fit_tca(
            tensor_prep,
            n_components=5, 
            max_iter=500,    # Should be fast!
            init='random',
            random_state=0
        )

        # Analyze components
        component_stats = analyze_components(weights, factors, metadata)

        # Cluster cells
        cell_clusters = cluster_cells_by_components(factors, metadata, n_clusters=4)

        # Visualize
        print("\nGenerating visualizations...")

        # Main overview plot
        fig = plot_tca_overview(weights, factors, metadata, component_stats, reward_filter=REWARD_GROUP_FILTER)
        title_suffix = f'({REWARD_GROUP_FILTER} mice)' if REWARD_GROUP_FILTER != 'all' else '(All mice: R+ & R-)'
        fig.suptitle(f'TCA (CP): Day 0 Whisker Trials {title_suffix}',
                    fontsize=16, fontweight='bold', y=0.995)
        fig.savefig(os.path.join(output_dir, f'tca_day0_tensorly_{suffix}.svg'), format='svg', bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # # Save component stats
        # stats_path = os.path.join(output_dir, f'tca_tensorly_component_stats_{suffix}.csv')
        # component_stats.to_csv(stats_path, index=False)
        # print(f"\n  Component stats saved to: {stats_path}")

        # # Save cell clusters
        # clusters_path = os.path.join(output_dir, f'tca_tensorly_cell_clusters_{suffix}.csv')
        # cell_clusters.to_csv(clusters_path, index=False)
        # print(f"  Cell clusters saved to: {clusters_path}")

        # # Save factors
        # factors_path = os.path.join(output_dir, f'tca_tensorly_factors_{suffix}.npz')
        # np.savez(factors_path,
        #          weights=weights,
        #          neuron_factors=factors[0],
        #          time_factors=factors[1],
        #          trial_factors=factors[2])
        # print(f"  Factors saved to: {factors_path}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {pdf_path}\n")
