"""
TCA Analysis of Day 0 Learning Plasticity using sliceTCA

Uses the slicetca library to decompose day 0 neural activity and identify:
1. Temporal patterns during trials (when plasticity occurs)
2. Trial progression patterns (when plasticity emerges)
3. Cell subpopulations with different learning dynamics

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
import torch
import slicetca
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

def load_day0_tensor(mouse_list, cell_list_dict, trial_type='whisker',
                     time_window=(-0.5, 2.0), baseline_subtract=True):
    """
    Load day 0 data and construct tensor: Time × Neurons × Trials.
    (Note: slicetca uses this dimension order)

    Parameters
    ----------
    mouse_list : list
        List of mouse IDs
    cell_list_dict : dict
        {mouse_id: [roi1, roi2, ...]}
    trial_type : str
        'all', 'hits', 'misses', or 'whisker'
    time_window : tuple
        (start, end) in seconds
    baseline_subtract : bool
        Whether to use baseline-subtracted data

    Returns
    -------
    tensor : ndarray
        Shape (n_time, n_neurons, n_trials)
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
            # Transpose to (n_cells, n_time, n_trials) first
            data = data.transpose(0, 2, 1)

            all_data.append(data)

            # Store cell info
            for roi in xarr['roi'].values:
                cell_info.append({
                    'mouse_id': mouse_id,
                    'roi': int(roi),
                    'cell_id': f"{mouse_id}_{int(roi)}"
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

    # Remove trials with NaN values (trials not present in all mice)
    nan_trials = np.any(np.isnan(tensor), axis=(0, 1))
    tensor = tensor[:, :, ~nan_trials]

    # Transpose to slicetca format: (n_time, n_neurons, n_trials)
    tensor = tensor.transpose(1, 0, 2)

    # Get time axis
    time_axis = xarr['time'].values

    # Create metadata
    metadata = {
        'cell_info': pd.DataFrame(cell_info),
        'trial_info': pd.DataFrame(trial_info_list),
        'time_axis': time_axis,
        'n_time': tensor.shape[0],
        'n_neurons': tensor.shape[1],
        'n_trials': tensor.shape[2],
        'trial_type': trial_type,
        'time_window': time_window
    }

    print(f"\n  Final tensor shape: {tensor.shape} (time, neurons, trials)")
    print(f"  Time points: {metadata['n_time']}")
    print(f"  Neurons: {metadata['n_neurons']}")
    print(f"  Trials: {metadata['n_trials']}")

    return tensor, metadata


# =============================================================================
# FUNCTION: Prepare tensor for TCA
# =============================================================================

def prepare_tensor_for_tca(tensor, metadata, smooth_sigma=2, normalize=True):
    """
    Prepare tensor for sliceTCA analysis.

    Parameters
    ----------
    tensor : ndarray
        Raw tensor (time, neurons, trials)
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

    # Gaussian smoothing along time axis
    if smooth_sigma > 0:
        print(f"  Applying Gaussian smoothing (sigma={smooth_sigma})...")
        tensor_prep = spnd.gaussian_filter1d(tensor_prep.astype('float32'),
                                            sigma=smooth_sigma, axis=0)

    # Z-score normalization (for each neuron across time and trials)
    if normalize:
        print(f"  Z-scoring neurons...")
        # Calculate mean and std across time and trials for each neuron
        mean_vals = np.nanmean(tensor_prep, axis=(0, 2), keepdims=True)
        std_vals = np.nanstd(tensor_prep, axis=(0, 2), keepdims=True)
        # Avoid division by zero
        std_vals[std_vals == 0] = 1.0
        tensor_prep = (tensor_prep - mean_vals) / std_vals

    print(f"  Final tensor shape: {tensor_prep.shape}")
    print(f"  Value range: [{np.nanmin(tensor_prep):.3f}, {np.nanmax(tensor_prep):.3f}]")

    return tensor_prep


# =============================================================================
# FUNCTION: Fit sliceTCA
# =============================================================================

def fit_slicetca(tensor, n_components=(3, 3, 3), learning_rate=5e-3,
                max_iter=10000, positive=True, seed=0):
    """
    Fit sliceTCA model.

    Parameters
    ----------
    tensor : ndarray
        Prepared tensor (time, neurons, trials)
    n_components : tuple
        (n_trial_components, n_neuron_components, n_time_components)
    learning_rate : float
        Learning rate for optimization
    max_iter : int
        Maximum iterations
    positive : bool
        Enforce non-negative factors
    seed : int
        Random seed

    Returns
    -------
    components : dict
        Extracted components
    model : slicetca model
        Fitted model
    """
    print(f"\n{'='*70}")
    print(f"Fitting sliceTCA")
    print(f"  Components: {n_components} (trial, neuron, time)")
    print(f"  Max iterations: {max_iter}")
    print(f"  Positive constraint: {positive}")
    print(f"{'='*70}\n")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Convert to torch tensor
    tensor_torch = torch.tensor(tensor, dtype=torch.float, device=device)

    # Fit model
    components, model = slicetca.decompose(
        tensor_torch,
        number_components=n_components,
        positive=positive,
        learning_rate=learning_rate,
        min_std=1e-3,
        max_iter=max_iter,
        seed=seed
    )

    print(f"\n  Model fitting completed!")

    return components, model


# =============================================================================
# FUNCTION: Analyze components
# =============================================================================

def analyze_components(model, metadata):
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

    # Get factor matrices from model
    trial_factors = model.factors[0].detach().cpu().numpy()  # (n_trials, n_components)
    neuron_factors = model.factors[1].detach().cpu().numpy()  # (n_neurons, n_components)
    time_factors = model.factors[2].detach().cpu().numpy()  # (n_time, n_components)

    n_components = trial_factors.shape[1]
    stats_list = []

    time_axis = metadata['time_axis']

    for comp_idx in range(n_components):
        # Get factors for this component
        trial_factor = trial_factors[:, comp_idx]
        neuron_factor = neuron_factors[:, comp_idx]
        time_factor = time_factors[:, comp_idx]

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
            'peak_time': peak_time,
            'temporal_class': temporal_class,
            'trial_correlation': correlation,
            'trial_correlation_p': p_value,
            'is_learning_related': is_learning_related,
            'learning_direction': learning_direction if is_learning_related else 'none',
            'n_neurons_strong': n_neurons_strong,
        })

        print(f"  Component {comp_idx}:")
        print(f"    Peak time: {peak_time:.3f}s ({temporal_class})")
        print(f"    Trial correlation: {correlation:.3f} (p={p_value:.4f})")
        print(f"    Learning-related: {is_learning_related}")
        print()

    component_stats = pd.DataFrame(stats_list)
    return component_stats


# =============================================================================
# FUNCTION: Cluster cells
# =============================================================================

def cluster_cells_by_components(model, metadata, n_clusters=4):
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
    neuron_factors = model.factors[1].detach().cpu().numpy()

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
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TCA ANALYSIS OF DAY 0 LEARNING PLASTICITY (sliceTCA)")
    print("="*70)

    # Load LMI data to get cell lists
    print("\nLoading LMI data...")
    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

    # Get positive LMI cells for each mouse
    cell_list_dict = {}
    for mouse_id in lmi_df['mouse_id'].unique():
        mouse_cells = lmi_df[(lmi_df['mouse_id'] == mouse_id) & (lmi_df['lmi'] >= 0)]
        cell_list_dict[mouse_id] = mouse_cells['roi'].tolist()

    mouse_list = list(cell_list_dict.keys())
    print(f"  Mice: {mouse_list}")
    print(f"  Total cells: {sum([len(cells) for cells in cell_list_dict.values()])}")

    # Define output path
    output_dir = io.results_dir
    pdf_path = os.path.join(output_dir, 'tca_day0_slicetca.pdf')

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
            mouse_list, cell_list_dict,
            trial_type='whisker',
            time_window=(-0.5, 2.0),
            baseline_subtract=True
        )

        # Prepare tensor
        tensor_prep = prepare_tensor_for_tca(tensor, metadata,
                                            smooth_sigma=2,
                                            normalize=True)

        # Fit TCA
        components, model = fit_slicetca(
            tensor_prep,
            n_components=(3, 3, 3),  # (trial, neuron, time)
            learning_rate=5e-3,
            max_iter=10000,
            positive=True,
            seed=0
        )

        # Analyze components
        component_stats = analyze_components(model, metadata)

        # Cluster cells
        cell_clusters = cluster_cells_by_components(model, metadata, n_clusters=4)

        # Visualize using slicetca's plotting
        print("\nGenerating visualizations...")

        # Main slicetca plot
        fig = plt.figure(figsize=(20, 12))
        time_ticks = np.linspace(0, len(metadata['time_axis'])-1, 4)
        time_labels = np.linspace(metadata['time_window'][0],
                                 metadata['time_window'][1], 4)

        axes = slicetca.plot(
            model,
            variables=('trial', 'neuron', 'time'),
            ticks=(None, None, time_ticks),
            tick_labels=(None, None, time_labels),
            quantile=0.99
        )
        plt.suptitle('sliceTCA: Day 0 Whisker Trials (Positive LMI Cells)',
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(plt.gcf(), bbox_inches='tight')
        plt.close()

        # Save component stats
        stats_path = os.path.join(output_dir, 'tca_slicetca_component_stats.csv')
        component_stats.to_csv(stats_path, index=False)
        print(f"\n  Component stats saved to: {stats_path}")

        # Save cell clusters
        clusters_path = os.path.join(output_dir, 'tca_slicetca_cell_clusters.csv')
        cell_clusters.to_csv(clusters_path, index=False)
        print(f"  Cell clusters saved to: {clusters_path}")

        # Save model
        model_path = os.path.join(output_dir, 'tca_slicetca_model_day0.pt')
        torch.save(model, model_path)
        print(f"  Model saved to: {model_path}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {pdf_path}\n")
