"""
TCA Analysis of Day 0 Learning Plasticity

This script uses Tensor Component Analysis (TCA) to discover:
1. When plasticity occurs during trials (temporal structure)
2. When plasticity emerges during learning (trial progression)
3. Which cells participate in different components (cell subpopulations)

The analysis builds a tensor: Neurons × Time × Trials (day 0 only)
and decomposes it to find interpretable components that reveal
learning-related neural dynamics.

Author: Claude Code
Date: 2025-11-22
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import NMF
from joblib import Parallel, delayed
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
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palettes
colors = sns.color_palette("husl", 8)
component_colors = sns.color_palette("Set2", 8)


# =============================================================================
# HELPER: Non-negative Tensor Factorization
# =============================================================================

class SimpleTensorDecomposition:
    """
    Simple non-negative tensor decomposition using alternating least squares.
    For a 3-way tensor X (neurons × time × trials), finds:
        X ≈ sum_r (neuron_r ⊗ time_r ⊗ trial_r)

    This is a simplified implementation using multiplicative updates.
    """

    def __init__(self, n_components=5, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.neuron_factors = None
        self.time_factors = None
        self.trial_factors = None
        self.reconstruction_error = []

    def _initialize_factors(self, shape):
        """Initialize factor matrices with small random positive values."""
        np.random.seed(self.random_state)
        n_neurons, n_time, n_trials = shape

        self.neuron_factors = np.abs(np.random.randn(n_neurons, self.n_components)) + 0.1
        self.time_factors = np.abs(np.random.randn(n_time, self.n_components)) + 0.1
        self.trial_factors = np.abs(np.random.randn(n_trials, self.n_components)) + 0.1

        # Normalize
        self.neuron_factors /= np.linalg.norm(self.neuron_factors, axis=0)
        self.time_factors /= np.linalg.norm(self.time_factors, axis=0)
        self.trial_factors /= np.linalg.norm(self.trial_factors, axis=0)

    def _reconstruct(self):
        """Reconstruct tensor from factors."""
        reconstruction = np.zeros((self.neuron_factors.shape[0],
                                  self.time_factors.shape[0],
                                  self.trial_factors.shape[0]))

        for r in range(self.n_components):
            # Outer product of the three factors
            component = np.einsum('i,j,k->ijk',
                                 self.neuron_factors[:, r],
                                 self.time_factors[:, r],
                                 self.trial_factors[:, r])
            reconstruction += component

        return reconstruction

    def _compute_error(self, X):
        """Compute reconstruction error."""
        X_recon = self._reconstruct()
        error = np.linalg.norm(X - X_recon) / np.linalg.norm(X)
        return error

    def fit(self, X):
        """
        Fit tensor decomposition using alternating least squares.

        Parameters
        ----------
        X : ndarray, shape (n_neurons, n_time, n_trials)
            Input tensor
        """
        # Sanity check: prevent memory explosion
        n_neurons, n_time, n_trials = X.shape
        kron_size = n_time * n_trials
        memory_gb = (kron_size ** 2) * 8 / 1e9
        if memory_gb > 100:
            raise ValueError(
                f"Tensor dimensions too large for TCA! "
                f"Shape: {X.shape}, Kronecker product: {kron_size}, "
                f"Required memory: ~{memory_gb:.1f} GB. "
                f"Did you forget to downsample time?"
            )

        self._initialize_factors(X.shape)

        # DEBUG: Print factor shapes after initialization
        print(f"  [DEBUG] After initialization:")
        print(f"    X.shape: {X.shape}")
        print(f"    neuron_factors.shape: {self.neuron_factors.shape}")
        print(f"    time_factors.shape: {self.time_factors.shape}")
        print(f"    trial_factors.shape: {self.trial_factors.shape}")

        for iteration in range(self.max_iter):
            # Update neuron factors (mode-1)
            X_1 = X.reshape(n_neurons, -1)  # Unfold along neuron dimension
            temp = np.kron(self.trial_factors, self.time_factors)

            if iteration == 0:
                print(f"  [DEBUG] First iteration, mode-1 update:")
                print(f"    X_1.shape: {X_1.shape}, temp.shape: {temp.shape}")

            # Break computation into steps and use inv instead of pinv
            gram = temp.T @ temp
            if iteration == 0:
                print(f"    gram.shape: {gram.shape}")

            # Add small regularization for numerical stability
            gram_reg = gram + 1e-8 * np.eye(gram.shape[0])
            gram_inv = np.linalg.inv(gram_reg)

            if iteration == 0:
                print(f"    gram_inv.shape: {gram_inv.shape}, computing X_1 @ temp @ gram_inv...")

            self.neuron_factors = np.maximum(X_1 @ temp @ gram_inv, 1e-10)

            # Update time factors (mode-2)
            X_2 = X.transpose(1, 0, 2).reshape(n_time, -1)
            temp = np.kron(self.trial_factors, self.neuron_factors)
            gram = temp.T @ temp
            gram_reg = gram + 1e-8 * np.eye(gram.shape[0])
            self.time_factors = np.maximum(X_2 @ temp @ np.linalg.inv(gram_reg), 1e-10)

            # Update trial factors (mode-3)
            X_3 = X.transpose(2, 0, 1).reshape(n_trials, -1)
            temp = np.kron(self.time_factors, self.neuron_factors)
            gram = temp.T @ temp
            gram_reg = gram + 1e-8 * np.eye(gram.shape[0])
            self.trial_factors = np.maximum(X_3 @ temp @ np.linalg.inv(gram_reg), 1e-10)

            # Normalize factors
            for r in range(self.n_components):
                norm = np.linalg.norm(self.neuron_factors[:, r])
                self.neuron_factors[:, r] /= norm
                self.time_factors[:, r] *= norm

            # Compute error
            error = self._compute_error(X)
            self.reconstruction_error.append(error)

            if iteration > 0 and abs(self.reconstruction_error[-2] - error) < self.tol:
                print(f"  Converged at iteration {iteration}")
                break

        return self

    def transform(self, X):
        """Get component activations for new data."""
        return {
            'neuron': self.neuron_factors,
            'time': self.time_factors,
            'trial': self.trial_factors
        }


# =============================================================================
# FUNCTION: Load and prepare data
# =============================================================================

def load_day0_tensor(mouse_list, cell_list_dict, trial_type='all',
                     time_window=(-1, 5), baseline_subtract=True):
    """
    Load day 0 data and construct tensor: Neurons × Time × Trials.

    Parameters
    ----------
    mouse_list : list
        List of mouse IDs to include
    cell_list_dict : dict
        {mouse_id: [roi1, roi2, ...]}
    trial_type : str
        'all', 'hits', 'misses', or 'whisker'
    time_window : tuple
        (start, end) time in seconds
    baseline_subtract : bool
        Whether to use baseline-subtracted data

    Returns
    -------
    tensor : ndarray
        Shape (n_neurons, n_time, n_trials)
    metadata : dict
        Information about cells, time axis, trials
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
            # 'all' includes all trials

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

            # Convert to numpy and add to list
            data = xarr.values  # Shape: (n_cells, n_trials, n_time)
            data = data.transpose(0, 2, 1)  # Reshape to (n_cells, n_time, n_trials)

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

    # Concatenate all data across mice
    # all_data is list of (n_cells_i, n_time, n_trials_i)
    # Goal: (n_cells_total, n_time, max_trials) with NaN padding

    # Find maximum number of trials across all mice
    max_trials = max([d.shape[2] for d in all_data])

    # Pad trials to max_trials for each mouse's data
    padded_data = []
    for data in all_data:
        n_cells, n_time, n_trials = data.shape
        if n_trials < max_trials:
            # Pad with NaN for missing trials
            padding = np.full((n_cells, n_time, max_trials - n_trials), np.nan)
            data = np.concatenate([data, padding], axis=2)
        padded_data.append(data)

    # Concatenate along cell dimension
    tensor = np.concatenate(padded_data, axis=0)  # (n_cells_total, n_time, max_trials)

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

    print(f"\n  Final tensor shape: {tensor.shape}")
    print(f"  Neurons: {metadata['n_neurons']}")
    print(f"  Time points: {metadata['n_time']}")
    print(f"  Trials: {metadata['n_trials']}")

    return tensor, metadata


# =============================================================================
# FUNCTION: Prepare tensor for TCA
# =============================================================================

def prepare_tensor_for_tca(tensor, metadata, remove_nan=True, z_score=True, time_bin_size=10):
    """
    Prepare tensor for TCA analysis.

    Parameters
    ----------
    tensor : ndarray
        Raw tensor (neurons, time, trials)
    metadata : dict
        Metadata from load_day0_tensor
    remove_nan : bool
        Remove trials with NaN values
    z_score : bool
        Z-score normalize each neuron
    time_bin_size : int
        Bin time points to reduce memory (1=no binning, 3=bin every 3 points)

    Returns
    -------
    tensor_prepared : ndarray
        Prepared tensor
    metadata_updated : dict
        Updated metadata
    """
    print(f"\n{'='*70}")
    print(f"Preparing Tensor for TCA")
    print(f"{'='*70}\n")

    tensor_prep = tensor.copy()

    # Downsample time dimension to reduce memory
    if time_bin_size > 1:
        print(f"  Downsampling time by factor {time_bin_size} to reduce memory...")
        n_neurons, n_time, n_trials = tensor_prep.shape
        n_time_binned = n_time // time_bin_size

        # Bin by averaging
        tensor_binned = np.zeros((n_neurons, n_time_binned, n_trials))
        for i in range(n_time_binned):
            start_idx = i * time_bin_size
            end_idx = min((i + 1) * time_bin_size, n_time)
            tensor_binned[:, i, :] = np.nanmean(tensor_prep[:, start_idx:end_idx, :], axis=1)

        tensor_prep = tensor_binned

        # Update time axis
        time_axis_binned = metadata['time_axis'][::time_bin_size][:n_time_binned]
        metadata['time_axis'] = time_axis_binned
        metadata['time_bin_size'] = time_bin_size

        print(f"    Original time points: {n_time}")
        print(f"    Binned time points: {n_time_binned}")

    # Remove trials with NaN
    if remove_nan:
        nan_trials = np.any(np.isnan(tensor_prep), axis=(0, 1))
        tensor_prep = tensor_prep[:, :, ~nan_trials]
        print(f"  Removed {nan_trials.sum()} trials with NaN values")
        print(f"  Remaining trials: {tensor_prep.shape[2]}")

    # Z-score normalize each neuron (across time and trials)
    if z_score:
        print(f"  Z-scoring neurons...")
        for neuron_idx in range(tensor_prep.shape[0]):
            neuron_data = tensor_prep[neuron_idx, :, :].flatten()
            if np.std(neuron_data) > 0:
                tensor_prep[neuron_idx, :, :] = (tensor_prep[neuron_idx, :, :] - np.mean(neuron_data)) / np.std(neuron_data)

    # Make non-negative (shift to minimum = 0)
    print(f"  Shifting to non-negative values...")
    min_val = np.min(tensor_prep)
    tensor_prep = tensor_prep - min_val

    metadata_updated = metadata.copy()
    metadata_updated['n_trials'] = tensor_prep.shape[2]
    metadata_updated['n_time'] = tensor_prep.shape[1]
    metadata_updated['preprocessed'] = {
        'remove_nan': remove_nan,
        'z_score': z_score,
        'min_shift': min_val,
        'time_bin_size': time_bin_size
    }

    print(f"  Final tensor shape: {tensor_prep.shape}")
    print(f"  Value range: [{np.min(tensor_prep):.3f}, {np.max(tensor_prep):.3f}]")

    # Estimate memory requirement
    memory_gb = (tensor_prep.shape[1] * tensor_prep.shape[2]) ** 2 * 8 / 1e9
    print(f"  Estimated TCA memory: ~{memory_gb:.2f} GB")

    return tensor_prep, metadata_updated


# =============================================================================
# FUNCTION: Fit TCA with multiple initializations
# =============================================================================

# Helper function at module level to avoid joblib closure issues
def _fit_single_tca_run(tensor, n_components, max_iter, run_id):
    """Fit a single TCA run - must be module-level for proper joblib serialization."""
    model = SimpleTensorDecomposition(
        n_components=n_components,
        max_iter=max_iter,
        random_state=run_id
    )
    model.fit(tensor)
    final_error = model.reconstruction_error[-1]
    return model, final_error


def fit_tca_multiple_runs(tensor, n_components=5, n_runs=5, max_iter=100, n_jobs=35):
    """
    Fit TCA with multiple random initializations and select best model.

    Parameters
    ----------
    tensor : ndarray
        Prepared tensor (neurons, time, trials)
    n_components : int
        Number of components
    n_runs : int
        Number of random initializations
    max_iter : int
        Maximum iterations per run
    n_jobs : int
        Number of parallel jobs. If 1, runs sequentially.
        If -1, uses all available cores.

    Returns
    -------
    best_model : SimpleTensorDecomposition
        Best fitted model (lowest error)
    all_models : list
        All fitted models
    """
    print(f"\n{'='*70}")
    print(f"Fitting TCA: {n_components} components, {n_runs} runs")
    if n_jobs != 1:
        print(f"Parallelization: {n_jobs} jobs")
    print(f"{'='*70}\n")

    # Verify tensor shape before fitting
    print(f"  Input tensor shape to TCA: {tensor.shape}")
    print(f"  Expected Kronecker product size: {tensor.shape[1] * tensor.shape[2]}\n")

    # Run TCA fits (parallel or sequential)
    if n_jobs == 1:
        # Sequential execution
        results = []
        for run in range(n_runs):
            model, error = _fit_single_tca_run(tensor, n_components, max_iter, run)
            print(f"  Run {run+1}/{n_runs}... Final error: {error:.4f}")
            results.append((model, error))
    else:
        # Parallel execution - pass tensor explicitly to avoid closure issues
        print(f"  Running {n_runs} initializations in parallel...")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_fit_single_tca_run)(tensor, n_components, max_iter, run)
            for run in range(n_runs)
        )

    # Unpack results
    all_models = [r[0] for r in results]
    errors = [r[1] for r in results]

    # Select best model
    best_idx = np.argmin(errors)
    best_model = all_models[best_idx]

    print(f"\n  Best model: Run {best_idx+1} (error: {errors[best_idx]:.4f})")
    print(f"  Error range: {np.min(errors):.4f} - {np.max(errors):.4f}")
    print(f"  Error std: {np.std(errors):.4f}")

    return best_model, all_models


# =============================================================================
# FUNCTION: Analyze components
# =============================================================================

def analyze_components(model, metadata):
    """
    Analyze TCA components to identify learning-related patterns.

    Parameters
    ----------
    model : SimpleTensorDecomposition
        Fitted TCA model
    metadata : dict
        Data metadata

    Returns
    -------
    component_stats : pd.DataFrame
        Statistics for each component
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Components")
    print(f"{'='*70}\n")

    stats_list = []

    for comp_idx in range(model.n_components):
        # Get factors
        neuron_factor = model.neuron_factors[:, comp_idx]
        time_factor = model.time_factors[:, comp_idx]
        trial_factor = model.trial_factors[:, comp_idx]

        # Temporal statistics
        time_axis = metadata['time_axis']
        peak_time_idx = np.argmax(time_factor)
        peak_time = time_axis[peak_time_idx]

        # Is it early (< 300ms) or late (> 300ms)?
        temporal_class = 'early' if peak_time < 0.3 else 'late'

        # Trial progression statistics
        trial_numbers = np.arange(len(trial_factor))
        correlation, p_value = stats.spearmanr(trial_numbers, trial_factor)

        # Is it learning-related? (significant correlation with trial number)
        is_learning_related = (p_value < 0.05) and (abs(correlation) > 0.2)
        learning_direction = 'increasing' if correlation > 0 else 'decreasing'

        # Neuron statistics
        top_neurons_idx = np.argsort(neuron_factor)[-10:]  # Top 10 neurons
        top_neurons_loading = neuron_factor[top_neurons_idx].mean()

        stats_list.append({
            'component': comp_idx,
            'peak_time': peak_time,
            'temporal_class': temporal_class,
            'trial_correlation': correlation,
            'trial_correlation_p': p_value,
            'is_learning_related': is_learning_related,
            'learning_direction': learning_direction if is_learning_related else 'none',
            'n_neurons_strong': np.sum(neuron_factor > 0.1),  # Neurons with loading > 0.1
            'top_neurons_loading': top_neurons_loading
        })

        print(f"  Component {comp_idx}:")
        print(f"    Peak time: {peak_time:.3f}s ({temporal_class})")
        print(f"    Trial correlation: {correlation:.3f} (p={p_value:.4f})")
        print(f"    Learning-related: {is_learning_related} ({learning_direction})")
        print(f"    Strong neurons: {np.sum(neuron_factor > 0.1)}")
        print()

    component_stats = pd.DataFrame(stats_list)
    return component_stats


# =============================================================================
# FUNCTION: Cluster cells into subpopulations
# =============================================================================

def cluster_cells_by_components(model, metadata, n_clusters=4):
    """
    Cluster cells based on their component loadings.

    Parameters
    ----------
    model : SimpleTensorDecomposition
        Fitted TCA model
    metadata : dict
        Data metadata
    n_clusters : int
        Number of clusters

    Returns
    -------
    cell_clusters : pd.DataFrame
        Cell information with cluster assignments
    """
    print(f"\n{'='*70}")
    print(f"Clustering Cells into Subpopulations")
    print(f"{'='*70}\n")

    # Get neuron factors (neurons × components)
    neuron_factors = model.neuron_factors

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
        # Find dominant component for this cluster
        cluster_cells = cluster_labels == cluster_id
        cluster_loadings = neuron_factors[cluster_cells, :].mean(axis=0)
        dominant_comp = np.argmax(cluster_loadings)

        print(f"    Cluster {cluster_id}: {n_cells} cells, dominant component: {dominant_comp}")

    return cell_info


# =============================================================================
# FUNCTION: Plot TCA results
# =============================================================================

def plot_tca_overview(model, metadata, component_stats, figsize=(20, 12)):
    """
    Create overview figure showing all TCA components.

    Parameters
    ----------
    model : SimpleTensorDecomposition
        Fitted TCA model
    metadata : dict
        Data metadata
    component_stats : pd.DataFrame
        Component statistics

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_components = model.n_components
    fig, axes = plt.subplots(n_components, 3, figsize=figsize)

    time_axis = metadata['time_axis']
    trial_axis = np.arange(model.trial_factors.shape[0])

    for comp_idx in range(n_components):
        # Get component info
        stats_row = component_stats.iloc[comp_idx]
        is_learning = stats_row['is_learning_related']
        color = component_colors[comp_idx % len(component_colors)]

        # Plot temporal factor
        ax = axes[comp_idx, 0] if n_components > 1 else axes[0]
        ax.plot(time_axis, model.time_factors[:, comp_idx], color=color, linewidth=2)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(stats_row['peak_time'], color=color, linestyle=':', alpha=0.5)
        ax.set_ylabel(f'Component {comp_idx}\nLoading', fontweight='bold')
        ax.set_title(f'Temporal Profile ({stats_row["temporal_class"]})')
        if comp_idx == n_components - 1:
            ax.set_xlabel('Time (s)')

        # Plot trial factor
        ax = axes[comp_idx, 1] if n_components > 1 else axes[1]
        ax.plot(trial_axis, model.trial_factors[:, comp_idx], color=color, linewidth=2, alpha=0.7)
        ax.scatter(trial_axis, model.trial_factors[:, comp_idx], color=color, s=10, alpha=0.5)

        # Add trend line if learning-related
        if is_learning:
            z = np.polyfit(trial_axis, model.trial_factors[:, comp_idx], 1)
            p = np.poly1d(z)
            ax.plot(trial_axis, p(trial_axis), "r--", alpha=0.8, linewidth=2,
                   label=f'r={stats_row["trial_correlation"]:.2f}')
            ax.legend()

        ax.set_title(f'Trial Progression ({"Learning" if is_learning else "Stable"})')
        if comp_idx == n_components - 1:
            ax.set_xlabel('Trial number')

        # Plot neuron factor (sorted)
        ax = axes[comp_idx, 2] if n_components > 1 else axes[2]
        sorted_loadings = np.sort(model.neuron_factors[:, comp_idx])[::-1]
        ax.plot(sorted_loadings, color=color, linewidth=1.5)
        ax.axhline(0.1, color='k', linestyle='--', alpha=0.3, label='Threshold=0.1')
        ax.set_title(f'Neuron Loadings (n={stats_row["n_neurons_strong"]} strong)')
        ax.legend()
        if comp_idx == n_components - 1:
            ax.set_xlabel('Neuron (sorted)')

    plt.tight_layout()
    return fig


def plot_learning_components(model, metadata, component_stats, figsize=(15, 10)):
    """
    Focus plot on learning-related components.

    Parameters
    ----------
    model : SimpleTensorDecomposition
        Fitted TCA model
    metadata : dict
        Data metadata
    component_stats : pd.DataFrame
        Component statistics

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Find learning-related components
    learning_comps = component_stats[component_stats['is_learning_related']].index.tolist()

    if len(learning_comps) == 0:
        print("  No learning-related components found!")
        return None

    fig, axes = plt.subplots(2, len(learning_comps), figsize=figsize)
    if len(learning_comps) == 1:
        axes = axes.reshape(-1, 1)

    time_axis = metadata['time_axis']
    trial_axis = np.arange(model.trial_factors.shape[0])

    for idx, comp_idx in enumerate(learning_comps):
        stats_row = component_stats.iloc[comp_idx]
        color = component_colors[comp_idx % len(component_colors)]

        # Temporal profile
        ax = axes[0, idx]
        ax.plot(time_axis, model.time_factors[:, comp_idx], color=color, linewidth=3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3, label='Stimulus')
        ax.axvline(stats_row['peak_time'], color=color, linestyle=':', alpha=0.7,
                  label=f'Peak: {stats_row["peak_time"]:.2f}s')
        ax.fill_between(time_axis, 0, model.time_factors[:, comp_idx], alpha=0.3, color=color)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Component Loading', fontsize=11)
        ax.set_title(f'Component {comp_idx}: {stats_row["temporal_class"].capitalize()} Response',
                    fontsize=12, fontweight='bold')
        ax.legend()

        # Trial progression
        ax = axes[1, idx]
        ax.plot(trial_axis, model.trial_factors[:, comp_idx], color=color,
               linewidth=2, label='Component strength')
        ax.scatter(trial_axis, model.trial_factors[:, comp_idx], color=color, s=20, alpha=0.6)

        # Fit line
        z = np.polyfit(trial_axis, model.trial_factors[:, comp_idx], 1)
        p = np.poly1d(z)
        ax.plot(trial_axis, p(trial_axis), "r--", alpha=0.8, linewidth=2.5,
               label=f'Trend: r={stats_row["trial_correlation"]:.3f}, p={stats_row["trial_correlation_p"]:.4f}')

        ax.set_xlabel('Trial Number (Day 0)', fontsize=11)
        ax.set_ylabel('Component Strength', fontsize=11)
        ax.set_title(f'{stats_row["learning_direction"].capitalize()} During Learning',
                    fontsize=12, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    return fig


def plot_cell_clustering(cell_clusters, model, component_stats, figsize=(16, 10)):
    """
    Visualize cell clustering based on component loadings.

    Parameters
    ----------
    cell_clusters : pd.DataFrame
        Cell cluster assignments
    model : SimpleTensorDecomposition
        Fitted TCA model
    component_stats : pd.DataFrame
        Component statistics

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_clusters = cell_clusters['cluster'].nunique()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Component loadings heatmap
    ax = axes[0, 0]
    # Sort cells by cluster
    sorted_idx = np.argsort(cell_clusters['cluster'].values)
    loadings_sorted = model.neuron_factors[sorted_idx, :]

    im = ax.imshow(loadings_sorted.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Cells (sorted by cluster)', fontsize=11)
    ax.set_ylabel('Component', fontsize=11)
    ax.set_title('Component Loadings Across Cells', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Loading')

    # Add cluster boundaries
    cluster_changes = np.where(np.diff(cell_clusters.iloc[sorted_idx]['cluster'].values))[0] + 0.5
    for change in cluster_changes:
        ax.axvline(change, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Plot 2: Cluster sizes
    ax = axes[0, 1]
    cluster_counts = cell_clusters['cluster'].value_counts().sort_index()
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=component_colors[:n_clusters])
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.set_title('Cell Population per Cluster', fontsize=12, fontweight='bold')

    # Plot 3: Cluster profiles (average loadings)
    ax = axes[1, 0]
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = cell_clusters['cluster'] == cluster_id
        cluster_profile = model.neuron_factors[cluster_mask.values, :].mean(axis=0)
        ax.plot(cluster_profile, marker='o', label=f'Cluster {cluster_id}',
               color=component_colors[cluster_id-1], linewidth=2)

    ax.set_xlabel('Component', fontsize=11)
    ax.set_ylabel('Average Loading', fontsize=11)
    ax.set_title('Component Profiles by Cluster', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xticks(range(model.n_components))

    # Plot 4: Cluster composition by dominant component
    ax = axes[1, 1]
    cluster_dominant = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = cell_clusters['cluster'] == cluster_id
        cluster_profile = model.neuron_factors[cluster_mask.values, :].mean(axis=0)
        dominant_comp = np.argmax(cluster_profile)
        cluster_dominant.append(dominant_comp)

    # Create a mapping
    comp_names = [f'C{i}\n({component_stats.iloc[i]["temporal_class"]})'
                 for i in range(model.n_components)]

    bars = ax.bar(range(1, n_clusters + 1), [component_stats.iloc[d]['peak_time'] for d in cluster_dominant],
                 color=[component_colors[d] for d in cluster_dominant])
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Dominant Component Peak Time (s)', fontsize=11)
    ax.set_title('Temporal Preference by Cluster', fontsize=12, fontweight='bold')
    ax.axhline(0.3, color='k', linestyle='--', alpha=0.3, label='Early/Late boundary')
    ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TCA ANALYSIS OF DAY 0 LEARNING PLASTICITY")
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
    pdf_path = os.path.join(output_dir, 'tca_day0_plasticity.pdf')

    # Create PDF
    with PdfPages(pdf_path) as pdf:

        # =================================================================
        # Analysis for All Whisker Trials
        # =================================================================

        print("\n" + "="*70)
        print("ANALYSIS 1: ALL WHISKER TRIALS")
        print("="*70)

        # Load data
        tensor, metadata = load_day0_tensor(
            mouse_list, cell_list_dict,
            trial_type='whisker',
            time_window=(-0.5, 2.0),
            baseline_subtract=True
        )

        # Prepare tensor
        tensor_prep, metadata = prepare_tensor_for_tca(tensor, metadata)

        # Debug: Print actual tensor shape after preparation
        print(f"\n{'='*70}")
        print(f"TENSOR SHAPE AFTER PREPARATION: {tensor_prep.shape}")
        print(f"  Neurons: {tensor_prep.shape[0]}")
        print(f"  Time points: {tensor_prep.shape[1]}")
        print(f"  Trials: {tensor_prep.shape[2]}")
        print(f"  Expected temp.T @ temp shape: ({tensor_prep.shape[1] * tensor_prep.shape[2]}, {tensor_prep.shape[1] * tensor_prep.shape[2]})")
        print(f"  Expected memory: ~{(tensor_prep.shape[1] * tensor_prep.shape[2])**2 * 8 / 1e12:.3f} TiB")
        print(f"{'='*70}\n")

        # Fit TCA with parallel processing
        # n_jobs=20 uses 20 cores, n_runs=20 for robust model selection
        # Adjust n_jobs based on your server (use -1 for all cores)
        model, all_models = fit_tca_multiple_runs(
            tensor_prep,
            n_components=5,
            n_runs=3,  # Start with 5 runs
            max_iter=100,
            n_jobs=1  # Run sequentially firt to avoid joblib cache issues
        )

        # Analyze components
        component_stats = analyze_components(model, metadata)

        # Cluster cells
        cell_clusters = cluster_cells_by_components(model, metadata, n_clusters=4)

        # Visualize
        print("\nGenerating visualizations...")

        # Overview
        fig = plot_tca_overview(model, metadata, component_stats)
        fig.suptitle('TCA Overview: All Whisker Trials (Day 0)', fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Learning components
        fig = plot_learning_components(model, metadata, component_stats)
        if fig is not None:
            fig.suptitle('Learning-Related Components (Day 0)', fontsize=16, fontweight='bold', y=0.995)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Cell clustering
        fig = plot_cell_clustering(cell_clusters, model, component_stats)
        fig.suptitle('Cell Subpopulations Based on TCA Components', fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Save component stats
        stats_path = os.path.join(output_dir, 'tca_component_stats.csv')
        component_stats.to_csv(stats_path, index=False)
        print(f"\n  Component stats saved to: {stats_path}")

        # Save cell clusters
        clusters_path = os.path.join(output_dir, 'tca_cell_clusters.csv')
        cell_clusters.to_csv(clusters_path, index=False)
        print(f"  Cell clusters saved to: {clusters_path}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput saved to: {pdf_path}\n")
