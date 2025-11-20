
"""
This script detects reactivation events during no-stimulus trials and analyzes their
relationship with behavioral performance across days and mice.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, linregress
from scipy.signal import find_peaks
from joblib import Parallel, delayed

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *

# ============================================================================
# PARAMETERS
# ============================================================================

sampling_rate = 30  # Hz
win = (0, 0.300)  # Window for template: stimulus onset to 300ms after
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
n_map_trials = 40  # Number of mapping trials to use

# Template and event detection parameters
threshold_dff = 0.05  # 5% dff threshold for including cells in template
threshold_corr = 0.45  # Correlation threshold for event detection
min_event_distance_ms = 200  # Minimum distance between events (ms)
min_event_distance_frames = int(min_event_distance_ms / 1000 * sampling_rate)  # 3 frames at 30Hz

# Visualization parameters
time_per_row = 200  # seconds per row in correlation trace plots (rows calculated dynamically)

# Parallel processing parameters
n_jobs = 30  # Number of parallel jobs for processing mice (set to -1 to use all available cores)

# Load database and available mice
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

# Separate mice by reward group
r_plus_mice = []
r_minus_mice = []

for mouse in all_mice:
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if reward_group == 'R+':
            r_plus_mice.append(mouse)
        elif reward_group == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_whisker_template(mouse, day, threshold_dff=0.05, verbose=True):
    """
    Create whisker response template from mapping data for a specific day.

    Parameters
    ----------
    mouse : str
        Mouse ID
    day : int
        Day number (-2, -1, 0, 1, 2)
    threshold_dff : float
        Minimum absolute dff response for including cells (default: 0.05 = 5%)
    verbose : bool
        Print information about template creation

    Returns
    -------
    template : np.ndarray
        Template vector (n_cells,)
    cells_mask : np.ndarray
        Boolean mask indicating which cells pass threshold
    """
    if verbose:
        print(f"\n  Creating template for {mouse}, Day {day}")

    # Load mapping data
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray_map = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

    # Select the specific day
    xarray_day = xarray_map.sel(trial=xarray_map['day'] == day)

    # Select last n_map_trials for this day
    xarray_day = xarray_day.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Average over time window
    d = xarray_day.sel(time=slice(win[0], win[1])).mean(dim='time')

    # Handle NaN values
    d = d.fillna(0)

    # Average across trials to get template
    template = d.mean(dim='trial').values

    # Filter cells by threshold
    cells_mask = template >= threshold_dff

    # Create filtered template (set non-responsive cells to 0)
    template_filtered = template.copy()
    template_filtered[~cells_mask] = 0

    if verbose:
        print(f"    Total cells: {len(template)}")
        print(f"    Cells above {threshold_dff*100}% threshold: {cells_mask.sum()} ({cells_mask.sum()/len(template)*100:.1f}%)")
        print(f"    Template mean: {np.mean(template_filtered[cells_mask]):.4f}")

    return template_filtered, cells_mask


def compute_template_correlation(data, template):
    """
    Compute correlation between neural activity and template at each timepoint.

    Parameters
    ----------
    data : np.ndarray
        Neural activity data (n_cells, n_timepoints)
    template : np.ndarray
        Template vector (n_cells,)

    Returns
    -------
    correlations : np.ndarray
        Correlation values at each timepoint (n_timepoints,)
    """
    n_cells, n_timepoints = data.shape
    correlations = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        activity = data[:, t]
        # Handle potential NaN or constant values
        if np.std(activity) > 0 and np.std(template) > 0:
            correlations[t] = np.corrcoef(template, activity)[0, 1]
        else:
            correlations[t] = 0

    return correlations


def detect_reactivation_events(correlations, threshold=0.3, min_distance=3):
    """
    Detect reactivation events as threshold crossings (bottom-to-top).

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries
    threshold : float
        Correlation threshold for event detection
    min_distance : int
        Minimum distance between events (in frames)

    Returns
    -------
    event_indices : np.ndarray
        Indices of detected events
    """
    # Find where correlation crosses threshold from below
    above_threshold = correlations >= threshold
    crossings = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

    if len(crossings) == 0:
        return np.array([])

    # Enforce minimum distance between events
    filtered_events = [crossings[0]]
    for event_idx in crossings[1:]:
        if event_idx - filtered_events[-1] >= min_distance:
            filtered_events.append(event_idx)

    return np.array(filtered_events)


def map_events_to_blocks(event_indices, nostim_trials, n_timepoints_per_trial):
    """
    Map event indices to block IDs and count events per block.

    Parameters
    ----------
    event_indices : np.ndarray
        Indices of detected events in concatenated timeseries
    nostim_trials : xarray.DataArray
        No-stim trial data with block_id coordinate
    n_timepoints_per_trial : int
        Number of timepoints per trial

    Returns
    -------
    events_per_block : dict
        Dictionary mapping block_id to event count
    event_blocks : list
        List of block IDs for each event (for plotting)
    """
    # Get block IDs for each no_stim trial
    block_ids = nostim_trials['block_id'].values

    # Map event indices to trial numbers
    event_trials = event_indices // n_timepoints_per_trial

    # Map trials to blocks
    event_blocks = []
    for trial_idx in event_trials:
        if trial_idx < len(block_ids):
            event_blocks.append(block_ids[trial_idx])

    # Count events per block
    unique_blocks = np.unique(block_ids)
    events_per_block = {int(block): 0 for block in unique_blocks}

    for block in event_blocks:
        if block in events_per_block:
            events_per_block[int(block)] += 1

    return events_per_block, event_blocks


def get_block_boundaries(nostim_trials, n_timepoints_per_trial):
    """
    Find indices where block transitions occur in concatenated no_stim data.

    Parameters
    ----------
    nostim_trials : xarray.DataArray
        No-stim trial data with block_id coordinate
    n_timepoints_per_trial : int
        Number of timepoints per trial

    Returns
    -------
    boundaries : list
        List of indices where blocks change
    """
    block_ids = nostim_trials['block_id'].values
    boundaries = []

    for i in range(1, len(block_ids)):
        if block_ids[i] != block_ids[i-1]:
            boundaries.append(i * n_timepoints_per_trial)

    return boundaries


def extract_performance_per_block(nostim_trials):
    """
    Extract whisker hit rate (hr_w) per block from no_stim trial data.

    Parameters
    ----------
    nostim_trials : xarray.DataArray
        No-stim trial data with hr_w and block_id coordinates

    Returns
    -------
    hr_per_block : dict
        Dictionary mapping block_id to hit rate
    """
    block_ids = nostim_trials['block_id'].values
    hr_w = nostim_trials['hr_w'].values

    unique_blocks = np.unique(block_ids)
    hr_per_block = {}

    for block in unique_blocks:
        # Get hr_w values for this block (should be constant per block)
        block_mask = block_ids == block
        hr_values = hr_w[block_mask]
        # Take the first non-NaN value (they should all be the same per block)
        valid_hr = hr_values[~np.isnan(hr_values)]
        if len(valid_hr) > 0:
            hr_per_block[int(block)] = valid_hr[0]

    return hr_per_block


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_correlation_traces(correlations, events, block_boundaries, mouse, day,
                           save_path=None, time_per_row=200, sampling_rate=30,
                           ylim=None):
    """
    Plot correlation traces split across multiple rows with event markers.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries
    events : np.ndarray
        Event indices
    block_boundaries : list
        Indices where blocks change
    mouse : str
        Mouse ID
    day : int
        Day number
    save_path : str, optional
        Path to save figure
    time_per_row : float
        Time duration per row (seconds)
    sampling_rate : float
        Sampling rate (Hz)
    ylim : tuple, optional
        Y-axis limits (min, max) to use across all days
    """
    # Calculate time axis
    time_axis = np.arange(len(correlations)) / sampling_rate
    total_time = time_axis[-1] if len(time_axis) > 0 else 0

    # Calculate number of rows needed based on total time
    n_rows = int(np.ceil(total_time / time_per_row))
    if n_rows == 0:
        n_rows = 1

    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Calculate frames per row
    frames_per_row = int(time_per_row * sampling_rate)

    for row_idx in range(n_rows):
        ax = axes[row_idx]

        # Define time window - always show full time_per_row duration
        t_start = row_idx * time_per_row
        t_end = (row_idx + 1) * time_per_row

        # Get data indices for this window
        idx_start = row_idx * frames_per_row
        idx_end = min((row_idx + 1) * frames_per_row, len(correlations))

        if idx_start >= len(correlations):
            # No data for this row
            ax.set_xlim(t_start, t_end)
            ax.set_ylabel('Correlation', fontsize=9)
            ax.grid(True, alpha=0.2)
            if ylim is not None:
                ax.set_ylim(ylim)
            continue

        # Get data for this window
        time_window = time_axis[idx_start:idx_end]
        corr_window = correlations[idx_start:idx_end]

        # Plot correlation
        ax.plot(time_window, corr_window, 'k-', linewidth=0.8, alpha=0.7)
        ax.axhline(threshold_corr, color='gray', linestyle='--', linewidth=0.5, alpha=0.5,
                  label=f'Threshold ({threshold_corr})' if row_idx == 0 else '')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add event markers (red lines)
        for event_idx in events:
            event_time = event_idx / sampling_rate
            if t_start <= event_time <= t_end:
                ax.axvline(event_time, color='red', linewidth=1.0, alpha=0.5)

        # Formatting - always use full time window
        ax.set_xlim(t_start, t_end)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(True, alpha=0.2)

        # Set y-limits (consistent across days if provided)
        if ylim is not None:
            ax.set_ylim(ylim)

        if row_idx == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=9)

        # Add row label
        ax.text(0.01, 0.95, f'Row {row_idx+1}/{n_rows}', transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if row_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Overall title
    fig.suptitle(f'{mouse} - Day {day} - Correlation Traces\n'
                f'{len(events)} events detected, Mean r = {np.mean(correlations):.3f}',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_events_vs_performance_per_block(events_per_block, hr_per_block, mouse, day, save_path=None):
    """
    Plot relationship between reactivation events and performance per block.
    Dual-axis plot with performance on top and events on bottom.

    Parameters
    ----------
    events_per_block : dict
        Number of events per block
    hr_per_block : dict
        Hit rate per block
    mouse : str
        Mouse ID
    day : int
        Day number
    save_path : str, optional
        Path to save figure
    """
    # Align data (only blocks with both events and performance data)
    common_blocks = set(events_per_block.keys()) & set(hr_per_block.keys())

    if len(common_blocks) == 0:
        print(f"Warning: No common blocks with both event and performance data for {mouse} Day {day}")
        return None

    blocks = sorted(common_blocks)
    events = [events_per_block[b] for b in blocks]
    hr = [hr_per_block[b] for b in blocks]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Performance (hr_w)
    ax1.plot(blocks, hr, 'o-', color='#2ca02c', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_ylabel('Whisker Hit Rate (hr_w)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{mouse} - Day {day}\nPerformance and Reactivation Events per Block',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])

    # Bottom panel: Reactivation events
    ax2.plot(blocks, events, 'o-', color='#d62728', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xlabel('Block ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Reactivation Events', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # Add correlation statistics in text box
    if len(events) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(events, hr)
        stats_text = f'Correlation:\nr = {r_value:.3f}, p = {p_value:.4f}'
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_events_per_day(events_by_day, mouse, save_path=None):
    """
    Plot bar chart of events per day for a single mouse.

    Parameters
    ----------
    events_by_day : dict
        Number of events per day {day: count}
    mouse : str
        Mouse ID
    save_path : str, optional
        Path to save figure
    """
    days_list = sorted(events_by_day.keys())
    events_list = [events_by_day[d] for d in days_list]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ['#1f77b4' if d != 0 else '#ff7f0e' for d in days_list]
    ax.bar(days_list, events_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Formatting
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Number of Reactivation Events', fontsize=11)
    ax.set_title(f'{mouse} - Reactivation Events Across Days', fontsize=12, fontweight='bold')
    ax.set_xticks(days_list)
    ax.set_xticklabels([str(d) for d in days_list])
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Day 0
    ax.axvline(0, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Day 0')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_mouse_reactivation(mouse, days=[-2, -1, 0, 1, 2], verbose=True):
    """
    Analyze reactivation events for a single mouse across multiple days.

    Parameters
    ----------
    mouse : str
        Mouse ID
    days : list
        List of days to analyze
    verbose : bool
        Print progress information

    Returns
    -------
    results : dict
        Nested dictionary containing all analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    results = {
        'mouse': mouse,
        'days': {}
    }

    for day in days:
        if verbose:
            print(f"\nProcessing Day {day}...")

        try:
            # Step 1: Create template
            template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=verbose)

            # Step 2: Load learning data
            folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
            file_name = 'tensor_xarray_learning_data.nc'
            xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

            # Select this day
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)

            # Select no_stim trials only
            nostim_trials = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

            n_nostim_trials = len(nostim_trials.trial)
            if verbose:
                print(f"  No-stim trials: {n_nostim_trials}")

            if n_nostim_trials == 0:
                if verbose:
                    print(f"  Warning: No no-stim trials for Day {day}, skipping...")
                continue

            # Step 3: Prepare data and compute correlations
            n_cells, n_trials, n_timepoints = nostim_trials.shape
            data = nostim_trials.values.reshape(n_cells, -1)
            data = np.nan_to_num(data, nan=0.0)

            correlations = compute_template_correlation(data, template)

            if verbose:
                print(f"  Correlation: mean={np.mean(correlations):.3f}, max={np.max(correlations):.3f}")

            # Step 4: Detect events
            events = detect_reactivation_events(correlations, threshold_corr, min_event_distance_frames)

            if verbose:
                print(f"  Events detected: {len(events)}")

            # Step 5: Map events to blocks
            events_per_block, event_blocks = map_events_to_blocks(events, nostim_trials, n_timepoints)

            # Step 6: Get block boundaries
            block_boundaries = get_block_boundaries(nostim_trials, n_timepoints)

            # Step 7: Extract performance
            hr_per_block = extract_performance_per_block(nostim_trials)

            # Store results
            results['days'][day] = {
                'template': template,
                'cells_mask': cells_mask,
                'correlations': correlations,
                'events': events,
                'events_per_block': events_per_block,
                'hr_per_block': hr_per_block,
                'block_boundaries': block_boundaries,
                'n_trials': n_nostim_trials,
                'n_timepoints': n_timepoints,
                'total_events': len(events),
                'session_hr_mean': np.mean(list(hr_per_block.values())) if hr_per_block else np.nan
            }

        except Exception as e:
            if verbose:
                print(f"  Error processing Day {day}: {str(e)}")
            continue

    return results


def generate_mouse_pdf(results, save_dir):
    """
    Generate multi-page PDF report for a single mouse.

    Parameters
    ----------
    results : dict
        Analysis results from analyze_mouse_reactivation
    save_dir : str
        Directory to save PDF
    """
    mouse = results['mouse']
    pdf_path = os.path.join(save_dir, f'{mouse}_reactivation_analysis.pdf')

    # Calculate global y-limits across all days for consistent y-axis
    all_correlations = []
    for day in days:
        if day in results['days']:
            all_correlations.extend(results['days'][day]['correlations'])

    if len(all_correlations) > 0:
        ylim = (np.min(all_correlations), np.max(all_correlations))
    else:
        ylim = None

    with PdfPages(pdf_path) as pdf:
        # Pages 1-5: Correlation traces for all days (in order: -2, -1, 0, +1, +2)
        for day in days:
            if day in results['days']:
                day_data = results['days'][day]
                fig = plot_correlation_traces(
                    day_data['correlations'],
                    day_data['events'],
                    day_data['block_boundaries'],
                    mouse, day,
                    time_per_row=time_per_row,
                    sampling_rate=sampling_rate,
                    ylim=ylim
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Page 6: Day 0 events vs performance per block
        if 0 in results['days']:
            day0 = results['days'][0]
            fig = plot_events_vs_performance_per_block(
                day0['events_per_block'],
                day0['hr_per_block'],
                mouse, 0
            )
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Page 7: Events per day
        events_by_day = {day: results['days'][day]['total_events']
                        for day in results['days'].keys()}
        fig = plot_events_per_day(events_by_day, mouse)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"  PDF saved: {pdf_path}")


# ============================================================================
# ACROSS-MICE ANALYSIS FUNCTIONS
# ============================================================================

def plot_session_level_across_mice(all_results, save_path):
    """
    Plot Day 0 session-level reactivation vs performance across mice.

    Parameters
    ----------
    all_results : dict
        Dictionary with results for all mice
    save_path : str
        Path to save SVG figure
    """
    # Extract data
    mice_list = []
    total_events_list = []
    session_hr_list = []

    for mouse, results in all_results.items():
        if 0 in results['days']:
            mice_list.append(mouse)
            total_events_list.append(results['days'][0]['total_events'])
            session_hr_list.append(results['days'][0]['session_hr_mean'])

    if len(mice_list) == 0:
        print("Warning: No Day 0 data available for session-level analysis")
        return None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot
    colors = plt.cm.tab10(range(len(mice_list)))
    for i, (mouse, events, hr) in enumerate(zip(mice_list, total_events_list, session_hr_list)):
        ax.scatter(events, hr, s=120, alpha=0.7, c=[colors[i]],
                  edgecolors='black', linewidth=1, label=mouse)

    # Regression line
    if len(total_events_list) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(total_events_list, session_hr_list)
        x_line = np.linspace(min(total_events_list), max(total_events_list), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Linear fit')

        # Add statistics text
        stats_text = f'n = {len(mice_list)} mice\n'
        stats_text += f'r = {r_value:.3f}\n'
        stats_text += f'p = {p_value:.4f}\n'
        stats_text += f'R² = {r_value**2:.3f}'

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    # Formatting
    ax.set_xlabel('Total Reactivation Events (Day 0 Session)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Session Hit Rate (hr_w)', fontsize=12, fontweight='bold')
    ax.set_title('Reactivation vs Performance\nSession Level (Across Mice)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


def plot_block_level_across_mice(all_results, save_path):
    """
    Plot Day 0 block-level reactivation vs performance averaged across mice.
    Dual-axis plot with performance on top and events on bottom.

    Parameters
    ----------
    all_results : dict
        Dictionary with results for all mice
    save_path : str
        Path to save SVG figure
    """
    # Build dataframe for easier processing with seaborn
    data_rows = []

    for mouse, results in all_results.items():
        if 0 in results['days']:
            day0 = results['days'][0]
            events_per_block = day0['events_per_block']
            hr_per_block = day0['hr_per_block']

            # Get common blocks
            common_blocks = set(events_per_block.keys()) & set(hr_per_block.keys())

            for block in common_blocks:
                data_rows.append({
                    'mouse': mouse,
                    'block_id': block,
                    'events': events_per_block[block],
                    'hr_w': hr_per_block[block]
                })

    if len(data_rows) == 0:
        print("Warning: No Day 0 block data available for analysis")
        return None

    df = pd.DataFrame(data_rows)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: Performance (hr_w) with confidence interval
    sns.lineplot(data=df, x='block_id', y='hr_w', ax=ax1,
                color='#2ca02c', linewidth=2.5, marker='o', markersize=8,
                errorbar=('ci', 95), err_style='band', alpha=0.8)
    ax1.set_ylabel('Whisker Hit Rate (hr_w)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance and Reactivation Events per Block\n(Averaged Across Mice, Day 0)',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])

    # Bottom panel: Reactivation events with confidence interval
    sns.lineplot(data=df, x='block_id', y='events', ax=ax2,
                color='#d62728', linewidth=2.5, marker='o', markersize=8,
                errorbar=('ci', 95), err_style='band', alpha=0.8)
    ax2.set_xlabel('Block ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reactivation Events', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # Add sample size info
    n_mice = df['mouse'].nunique()
    n_blocks_total = len(df)
    info_text = f'n = {n_mice} mice\n{n_blocks_total} blocks total'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Calculate correlation between average events and average hr_w per block
    block_avg = df.groupby('block_id').agg({'events': 'mean', 'hr_w': 'mean'}).reset_index()
    if len(block_avg) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(block_avg['events'], block_avg['hr_w'])
        stats_text = f'Correlation (block averages):\nr = {r_value:.3f}, p = {p_value:.4f}'
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


def plot_events_per_day_across_mice(all_results, save_path):
    """
    Plot bar chart of events per day averaged across mice.

    Parameters
    ----------
    all_results : dict
        Dictionary with results for all mice
    save_path : str
        Path to save SVG figure
    """
    # Collect events by day for each mouse
    events_by_day_by_mouse = {day: [] for day in days}

    for mouse, results in all_results.items():
        for day in days:
            if day in results['days']:
                events_by_day_by_mouse[day].append(results['days'][day]['total_events'])

    # Calculate mean and SEM
    days_list = sorted(events_by_day_by_mouse.keys())
    mean_events = []
    sem_events = []

    for day in days_list:
        data = events_by_day_by_mouse[day]
        if len(data) > 0:
            mean_events.append(np.mean(data))
            sem_events.append(np.std(data) / np.sqrt(len(data)))
        else:
            mean_events.append(0)
            sem_events.append(0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Bar plot
    colors = ['#1f77b4' if d != 0 else '#ff7f0e' for d in days_list]
    bars = ax.bar(days_list, mean_events, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1, yerr=sem_events,
                  capsize=5, error_kw={'linewidth': 2})

    # Add individual data points
    for day_idx, day in enumerate(days_list):
        data = events_by_day_by_mouse[day]
        if len(data) > 0:
            x_positions = [day] * len(data)
            # Add small jitter for visibility
            x_jitter = np.random.normal(0, 0.05, len(data))
            ax.scatter(x_positions + x_jitter, data, color='black',
                      s=40, alpha=0.5, zorder=10)

    # Formatting
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Reactivation Events', fontsize=12, fontweight='bold')
    ax.set_title(f'Reactivation Events Across Days\n(Mean ± SEM, n={len(all_results)} mice)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(days_list)
    ax.set_xticklabels([str(d) for d in days_list])
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Day 0
    ax.axvline(0, color='orange', linestyle='--', linewidth=2, alpha=0.5,
              label='Day 0 (Learning)', zorder=0)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  SVG saved: {save_path}")
        plt.close()

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_single_mouse(mouse, save_dir, verbose=False):
    """
    Process a single mouse (analysis + PDF generation).

    Parameters
    ----------
    mouse : str
        Mouse ID
    save_dir : str
        Directory to save PDF
    verbose : bool
        Print progress information

    Returns
    -------
    tuple
        (mouse_id, results_dict)
    """
    if verbose:
        print(f"\nProcessing {mouse}...")

    results = analyze_mouse_reactivation(mouse, days=days, verbose=verbose)
    generate_mouse_pdf(results, save_dir)

    if verbose:
        print(f"  Completed {mouse}")

    return (mouse, results)


def process_mouse_group(mice_list, group_name, save_dir, n_jobs=30):
    """
    Process a group of mice and generate their figures in parallel.

    Parameters
    ----------
    mice_list : list
        List of mouse IDs to process
    group_name : str
        Name of the group (e.g., 'R+', 'R-')
    save_dir : str
        Directory to save results
    n_jobs : int
        Number of parallel jobs (default: 30)

    Returns
    -------
    all_results : dict
        Dictionary with results for all mice in the group
    """
    print("\n" + "="*60)
    print(f"ANALYZING {group_name} MICE")
    print("="*60)
    print(f"Mice in group: {mice_list}")
    print(f"Processing {len(mice_list)} mice in parallel using {n_jobs} cores...")

    # Process all mice in parallel
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, save_dir, verbose=False)
        for mouse in mice_list
    )

    # Convert list of tuples to dictionary
    all_results = dict(results_list)

    print(f"\nCompleted processing {len(all_results)} {group_name} mice")

    # Generate across-mice SVG figures for this group (sequential, fast)
    print("\n" + "="*60)
    print(f"GENERATING ACROSS-MICE FIGURES FOR {group_name}")
    print("="*60)

    # Figure 1: Session-level analysis
    svg_path = os.path.join(save_dir, f'across_mice_session_level_{group_name}.svg')
    plot_session_level_across_mice(all_results, svg_path)

    # Figure 2: Block-level analysis
    svg_path = os.path.join(save_dir, f'across_mice_block_level_{group_name}.svg')
    plot_block_level_across_mice(all_results, svg_path)

    # Figure 3: Events per day
    svg_path = os.path.join(save_dir, f'across_mice_events_per_day_{group_name}.svg')
    plot_events_per_day_across_mice(all_results, svg_path)

    return all_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION EVENT DETECTION AND ANALYSIS")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Correlation threshold: {threshold_corr}")
    print(f"  Min event distance: {min_event_distance_ms}ms ({min_event_distance_frames} frames)")
    print(f"  DFF threshold: {threshold_dff*100}%")
    print(f"  Days: {days}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"\nMice groups:")
    print(f"  R+ mice ({len(r_plus_mice)}): {r_plus_mice}")
    print(f"  R- mice ({len(r_minus_mice)}): {r_minus_mice}")

    # Create output directory
    save_dir = os.path.join(io.results_dir, 'reactivation')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nResults will be saved to: {save_dir}")

    # Process R+ mice in parallel
    r_plus_results = process_mouse_group(r_plus_mice, 'R+', save_dir, n_jobs=n_jobs)

    # Process R- mice in parallel
    r_minus_results = process_mouse_group(r_minus_mice, 'R-', save_dir, n_jobs=n_jobs)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(r_plus_results)} R+ mice")
    print(f"Processed {len(r_minus_results)} R- mice")
    print(f"Total: {len(r_plus_results) + len(r_minus_results)} mice")
    print(f"\nPDFs saved to: {save_dir}")
    print(f"SVG figures saved to: {save_dir}")
