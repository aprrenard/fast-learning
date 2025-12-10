"""
Single-cell plasticity analysis during day 0 whisker learning.

This script fits sigmoid models to trial-by-trial responses of LMI-significant cells
and identifies cells showing online plasticity. Cells are ranked by response amplitude
and filtered by statistical significance (p < 0.05 via likelihood ratio test vs flat model).

Output: CSV with amplitude metrics, distribution plots, and 4 stratified PDF reports
        showing top 50 significant cells per group (R+/-, LMI+/-) sorted by amplitude.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2, mannwhitneyu
from scipy.optimize import curve_fit
from scipy.special import expit
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *

# =============================================================================
# PARAMETERS
# =============================================================================

# Analysis parameters
SAMPLING_RATE = 30  # Hz
RESPONSE_WIN = (0, 0.300)  # 0-300ms response window
RESPONSE_TYPE = 'mean'  # 'mean' or 'peak' within response window
AMPLITUDE_TYPE = 'absolute'  # 'absolute' or 'relative' - how to compute amplitude
MIN_TRIALS = 20  # Minimum whisker trials required for fitting
ALPHA = 0.05  # Significance threshold
N_CORES = 35  # Number of cores for parallel processing (one per mouse)

# LMI thresholds for cell selection
LMI_POSITIVE_THRESHOLD = 0.975  # Top 2.5% LMI cells
LMI_NEGATIVE_THRESHOLD = 0.025  # Bottom 2.5% LMI cells

# Output directory
OUTPUT_DIR = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/plasticity'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_day0_data(mouse_id, response_type='mean', response_win=(0, 0.300)):
    """
    Load and prepare day 0 whisker trial data for a single mouse.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak' - how to compute response from window
    response_win : tuple
        (start, end) time window in seconds

    Returns
    -------
    responses : np.ndarray
        Shape (n_cells, n_trials) - response values
    trial_indices : np.ndarray
        Shape (n_trials,) - trial_w values (whisker trial numbers)
    roi_ids : np.ndarray
        Shape (n_cells,) - ROI identifiers
    """
    # Load xarray
    folder = os.path.join(io.processed_dir, 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

    # Select day 0 whisker trials
    xarray = xarray.sel(trial=xarray['day'] == 0)
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)

    # Extract trial indices
    trial_indices = xarray['trial_w'].values

    # Compute response per trial
    xarray_win = xarray.sel(time=slice(*response_win))

    if response_type == 'mean':
        responses_xarr = xarray_win.mean(dim='time')
    elif response_type == 'peak':
        responses_xarr = xarray_win.max(dim='time')
    else:
        raise ValueError(f"response_type must be 'mean' or 'peak', got {response_type}")

    # Extract ROI identifiers
    roi_ids = xarray['roi'].values

    # Convert to numpy array: (n_cells, n_trials)
    responses = responses_xarr.values

    return responses, trial_indices, roi_ids


# =============================================================================
# MODEL FITTING FUNCTIONS
# =============================================================================

def sigmoid_4pl(x, baseline, max_val, inflection, slope_param):
    """
    4-parameter logistic (sigmoid) function using scipy's expit.

    Parameters
    ----------
    x : array-like
        Independent variable (trial numbers)
    baseline : float
        Bottom asymptote (minimum response)
    max_val : float
        Top asymptote (maximum response)
    inflection : float
        Inflection point (trial number where change is steepest)
    slope_param : float
        Slope parameter (controls steepness)

    Returns
    -------
    y : array-like
        Sigmoid function output
    """
    # Use scipy's expit (logistic sigmoid) for numerical stability
    # expit(x) = 1 / (1 + exp(-x))
    return baseline + (max_val - baseline) * expit((x - inflection) / slope_param)


def fit_sigmoid_model(x, y):
    """
    Fit 4-parameter logistic sigmoid model to data.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict or None
        Returns None if fitting fails, otherwise dict with:
        {
            'baseline': float,
            'max_val': float,
            'inflection': float,
            'slope_param': float,
            'predictions': np.ndarray,
            'residuals': np.ndarray,
            'n_params': int,  # Always 4
            'fit_success': bool
        }
    """
    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 5:  # Need more points for 4 parameters
        return None

    # Compute initial parameter estimates
    y_min, y_max = np.nanmin(y_clean), np.nanmax(y_clean)
    y_range = y_max - y_min

    # Initial guesses
    p0 = [
        y_min,  # baseline
        y_max,  # max_val
        np.median(x_clean),  # inflection (middle trial)
        (x_clean[-1] - x_clean[0]) / 4  # slope_param (quarter of trial range)
    ]

    # Bounds to ensure numerical stability
    bounds = (
        [y_min - y_range, y_min - y_range, x_clean[0], 0.1],  # Lower bounds
        [y_max + y_range, y_max + y_range, x_clean[-1], (x_clean[-1] - x_clean[0]) * 2]  # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                sigmoid_4pl, x_clean, y_clean,
                p0=p0, bounds=bounds, maxfev=5000
            )

        predictions = sigmoid_4pl(x_clean, *popt)
        residuals = y_clean - predictions

        # Compute robust amplitude by evaluating fitted curve over trial range
        # This is more robust than |max_val - baseline| which can be affected by outlier parameters
        trial_range = np.linspace(x_clean[0], x_clean[-1], 100)
        predictions_range = sigmoid_4pl(trial_range, *popt)

        # Absolute amplitude: max - min of fitted curve
        amplitude_absolute = predictions_range.max() - predictions_range.min()

        # Relative amplitude: normalized by baseline (avoid division by zero)
        baseline_val = popt[0]
        if abs(baseline_val) > 1e-10:  # Avoid division by very small values
            amplitude_relative = amplitude_absolute / abs(baseline_val)
        else:
            amplitude_relative = amplitude_absolute  # Fallback to absolute if baseline ~0

        # Compute parameter standard errors from covariance matrix
        try:
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = None

        return {
            'baseline': popt[0],
            'max_val': popt[1],
            'inflection': popt[2],
            'slope_param': popt[3],
            'predictions': predictions,
            'residuals': residuals,
            'n_params': 4,
            'fit_success': True,
            'x_clean': x_clean,
            'y_clean': y_clean,
            'amplitude_absolute': amplitude_absolute,
            'amplitude_relative': amplitude_relative,
            'pcov': pcov,
            'perr': perr
        }

    except (RuntimeError, ValueError):
        # Fitting failed
        return None


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_pseudo_r_squared(residuals, y):
    """
    Compute pseudo-R² for sigmoid model.

    Pseudo-R² = 1 - (RSS / TSS)
    where TSS = total sum of squares

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    y : np.ndarray
        Observed values

    Returns
    -------
    pseudo_r2 : float
    """
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals ** 2)

    if tss == 0:
        return 0.0

    return 1 - (rss / tss)


def fit_flat_model(y):
    """
    Fit flat (constant mean) model for null hypothesis comparison.

    Parameters
    ----------
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict
    """
    mask = ~np.isnan(y)
    y_clean = y[mask]

    mean_val = np.nanmean(y_clean)
    predictions = np.full_like(y_clean, mean_val)
    residuals = y_clean - predictions

    return {
        'mean': mean_val,
        'predictions': predictions,
        'residuals': residuals,
        'n_params': 1,
        'y_clean': y_clean
    }


def likelihood_ratio_test(residuals_null, residuals_alt, df_diff):
    """
    Perform likelihood ratio test between nested models.

    Parameters
    ----------
    residuals_null : np.ndarray
        Residuals from null (flat) model
    residuals_alt : np.ndarray
        Residuals from alternative (sigmoid) model
    df_diff : int
        Difference in degrees of freedom

    Returns
    -------
    p_value : float
    """
    n = len(residuals_null)
    rss_null = np.sum(residuals_null ** 2)
    rss_alt = np.sum(residuals_alt ** 2)

    if rss_alt <= 0:
        return 0.0

    lr_stat = n * np.log(rss_null / rss_alt)
    p_value = 1 - chi2.cdf(lr_stat, df_diff)

    return p_value


# =============================================================================
# SINGLE-CELL ANALYSIS
# =============================================================================

def analyze_single_cell(x, y, min_trials=20, amplitude_type='absolute'):
    """
    Fit sigmoid model to single cell's trial-by-trial responses.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)
    min_trials : int
        Minimum number of trials required
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results : dict or None
        Returns None if insufficient data or fitting failed
        Otherwise returns dict with sigmoid fit results
    """
    # Check data quality
    mask = ~np.isnan(y)
    n_valid = np.sum(mask)

    if n_valid < min_trials:
        return None

    # Fit sigmoid model
    sigmoid_fit = fit_sigmoid_model(x, y)
    if sigmoid_fit is None or not sigmoid_fit.get('fit_success', False):
        return None

    # Fit flat model for significance test
    flat_fit = fit_flat_model(y)

    # Compute pseudo-R²
    pseudo_r2 = compute_pseudo_r_squared(sigmoid_fit['residuals'], sigmoid_fit['y_clean'])

    # Perform likelihood ratio test
    p_value = likelihood_ratio_test(
        flat_fit['residuals'],
        sigmoid_fit['residuals'],
        df_diff=sigmoid_fit['n_params'] - flat_fit['n_params']
    )

    # Select amplitude type (absolute or relative)
    if amplitude_type == 'relative':
        amplitude = sigmoid_fit['amplitude_relative']
    else:  # Default to 'absolute'
        amplitude = sigmoid_fit['amplitude_absolute']

    return {
        'p_value': p_value,
        'pseudo_r2': pseudo_r2,
        'amplitude': amplitude,
        'amplitude_absolute': sigmoid_fit['amplitude_absolute'],
        'amplitude_relative': sigmoid_fit['amplitude_relative'],
        'baseline': sigmoid_fit['baseline'],
        'max_val': sigmoid_fit['max_val'],
        'inflection': sigmoid_fit['inflection'],
        'slope_param': sigmoid_fit['slope_param'],
        'predictions': sigmoid_fit['predictions'],
        'residuals': sigmoid_fit['residuals'],
        'x_clean': sigmoid_fit['x_clean'],
        'y_clean': sigmoid_fit['y_clean']
    }


# =============================================================================
# MOUSE-LEVEL PROCESSING
# =============================================================================

def process_mouse(mouse_id, response_type='mean', response_win=(0, 0.300),
                  min_trials=20, amplitude_type='absolute'):
    """
    Process all cells for a single mouse using sigmoid model.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak'
    response_win : tuple
        (start, end) time window
    min_trials : int
        Minimum trials required for fitting
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results_df : pd.DataFrame
        Columns: mouse_id, roi, reward_group, p_value, pseudo_r2, amplitude,
                 amplitude_absolute, amplitude_relative, baseline, max_val,
                 inflection, slope_param, n_trials
    """
    print(f"Processing {mouse_id}...")

    # Load data
    responses, trial_indices, roi_ids = load_day0_data(mouse_id, response_type, response_win)

    # Get reward group
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Process each cell
    results = []
    n_cells = responses.shape[0]

    for i, roi in enumerate(roi_ids):
        y = responses[i, :]
        x = trial_indices

        cell_results = analyze_single_cell(x, y, min_trials, amplitude_type)

        if cell_results is not None:
            results.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'reward_group': reward_group,
                'p_value': cell_results['p_value'],
                'pseudo_r2': cell_results['pseudo_r2'],
                'amplitude': cell_results['amplitude'],
                'amplitude_absolute': cell_results['amplitude_absolute'],
                'amplitude_relative': cell_results['amplitude_relative'],
                'baseline': cell_results['baseline'],
                'max_val': cell_results['max_val'],
                'inflection': cell_results['inflection'],
                'slope_param': cell_results['slope_param'],
                'n_trials': len(x)
            })

    results_df = pd.DataFrame(results)
    print(f"  Processed {len(results_df)} cells for {mouse_id}")

    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_distributions(results_df, output_dir):
    """
    Plot distributions of key plasticity metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory for plots
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)

    # Overall distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    # 1. Amplitude distribution
    ax = axes[0, 0]
    sns.histplot(data=results_df, x='amplitude', bins=50, ax=ax, color='darkgreen')
    ax.set_xlabel('Amplitude (Range of Fitted Curve)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Response Amplitude')
    ax.axvline(results_df['amplitude'].median(), color='red', linestyle='--', label='Median')
    ax.legend()

    # 2. Pseudo-R²

    ax = axes[0, 1]
    sns.histplot(data=results_df, x='pseudo_r2', bins=50, ax=ax, color='steelblue')
    ax.set_xlabel('Pseudo-R²')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Pseudo-R² (Sigmoid Fit)')
    ax.axvline(results_df['pseudo_r2'].median(), color='red', linestyle='--', label='Median')
    ax.legend()

    # 3. Inflection points
    ax = axes[1, 0]
    sns.histplot(data=results_df, x='inflection', bins=30, ax=ax, color='darkorange')
    ax.set_xlabel('Inflection Point (trial number)')
    ax.set_ylabel('Count')
    ax.set_title(f'Inflection Points (n={len(results_df)})')

    # 4. Proportion significant by reward group
    ax = axes[1, 1]
    sig_props = []
    for group in ['R+', 'R-']:
        df_g = results_df[results_df['reward_group'] == group]
        prop_sig = (df_g['p_value'] < ALPHA).sum() / len(df_g) if len(df_g) > 0 else 0
        sig_props.append({'reward_group': group, 'proportion': prop_sig})
    sig_props_df = pd.DataFrame(sig_props)
    sns.barplot(data=sig_props_df, x='reward_group', y='proportion',
                order=['R+', 'R-'], palette=reward_palette[::-1], ax=ax)
    ax.set_ylabel('Proportion Significant')
    ax.set_xlabel('Reward Group')
    ax.set_title('Significant Plasticity (p < 0.05)')
    ax.set_ylim(0, 1)
    for i, row in sig_props_df.iterrows():
        ax.text(i, row['proportion'] + 0.02, f'{row["proportion"]:.2f}', ha='center')

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'distributions_all.svg'), format='svg', dpi=150)
    plt.close()

    # Distributions by reward group
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)

    for idx, group in enumerate(['R+', 'R-']):
        df_group = results_df[results_df['reward_group'] == group]
        df_group_sig = df_group[df_group['p_value'] < ALPHA]
        color = reward_palette[1] if group == 'R+' else reward_palette[0]

        # Amplitude distribution
        ax = axes[idx, 0]
        sns.histplot(data=df_group, x='amplitude', bins=30, ax=ax, color=color)
        ax.set_xlabel('Amplitude (Range of Fitted Curve)')
        ax.set_ylabel('Count')
        ax.set_title(f'{group}: Amplitude (n={len(df_group)})')

        # Inflection points
        ax = axes[idx, 1]
        sns.histplot(data=df_group, x='inflection', bins=30, ax=ax, color=color)
        ax.set_xlabel('Inflection Point (trial number)')
        ax.set_ylabel('Count')
        ax.set_title(f'{group}: Inflection Points (n={len(df_group)})')

        # Inflection relative to learning trial (significant cells only)
        ax = axes[idx, 2]
        df_group_sig_with_learning = df_group_sig.dropna(subset=['learning_trial'])
        if len(df_group_sig_with_learning) > 0:
            sns.histplot(
                data=df_group_sig_with_learning, x='inflection_relative', kde=True,
                color=color, ax=ax, bins=20
            )
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
                      label='Behavioral learning trial')
            ax.legend(frameon=False)
        ax.set_xlabel('Cellular inflection - Learning trial (trials)')
        ax.set_ylabel('Count')
        ax.set_title(f'{group}: Inflection Timing (n={len(df_group_sig_with_learning)})')

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'distributions_by_reward.svg'), format='svg', dpi=150)
    plt.close()

    print("  ✓ Distribution plots saved")


def plot_amplitude_distributions_by_lmi(results_df, output_dir, alpha=0.05):
    """
    Plot amplitude distributions by reward group for LMI+ and LMI- cells.

    Creates 2-panel figure with overlaid histograms:
    - Left panel: LMI positive cells (R+ vs R-)
    - Right panel: LMI negative cells (R+ vs R-)

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells (must have 'lmi_sign' column)
    output_dir : str
        Output directory for saving figure
    alpha : float
        Significance threshold for filtering cells (default: 0.05)
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)

    # Filter for significant cells only
    sig_cells = results_df[results_df['p_value'] < alpha].copy()

    # Determine global bin edges across all data for consistency
    all_amplitudes = sig_cells['amplitude'].values
    bin_edges = np.histogram_bin_edges(all_amplitudes, bins=20)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Panel 1: LMI Positive cells
    lmi_pos = sig_cells[sig_cells['lmi_sign'] == 'Positive']
    lmi_pos_rplus = lmi_pos[lmi_pos['reward_group'] == 'R+']
    lmi_pos_rminus = lmi_pos[lmi_pos['reward_group'] == 'R-']

    ax = axes[0]
    # Histogram for R- with KDE
    sns.histplot(data=lmi_pos_rminus, x='amplitude', ax=ax, color=reward_palette[0],
                 alpha=0.4, label=f'R- (n={len(lmi_pos_rminus)})', stat='density',
                 bins=bin_edges, kde=True, line_kws={'linewidth': 2})
    # Overlaid histogram for R+ with KDE
    sns.histplot(data=lmi_pos_rplus, x='amplitude', ax=ax, color=reward_palette[1],
                 alpha=0.4, label=f'R+ (n={len(lmi_pos_rplus)})', stat='density',
                 bins=bin_edges, kde=True, line_kws={'linewidth': 2})
    ax.set_xlabel('Amplitude (Relative to Baseline)')
    ax.set_ylabel('Density')
    ax.set_title(f'LMI+ Cells (n={len(lmi_pos)})')
    ax.legend(frameon=False)

    # Panel 2: LMI Negative cells
    lmi_neg = sig_cells[sig_cells['lmi_sign'] == 'Negative']
    lmi_neg_rplus = lmi_neg[lmi_neg['reward_group'] == 'R+']
    lmi_neg_rminus = lmi_neg[lmi_neg['reward_group'] == 'R-']

    ax = axes[1]
    # Histogram for R- with KDE
    sns.histplot(data=lmi_neg_rminus, x='amplitude', ax=ax, color=reward_palette[0],
                 alpha=0.4, label=f'R- (n={len(lmi_neg_rminus)})', stat='density',
                 bins=bin_edges, kde=True, line_kws={'linewidth': 2})
    # Overlaid histogram for R+ with KDE
    sns.histplot(data=lmi_neg_rplus, x='amplitude', ax=ax, color=reward_palette[1],
                 alpha=0.4, label=f'R+ (n={len(lmi_neg_rplus)})', stat='density',
                 bins=bin_edges, kde=True, line_kws={'linewidth': 2})
    ax.set_xlabel('Amplitude (Relative to Baseline)')
    ax.set_ylabel('Density')
    ax.set_title(f'LMI- Cells (n={len(lmi_neg)})')
    ax.legend(frameon=False)

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'amplitude_distributions_by_reward_lmi.svg'), format='svg', dpi=150)
    plt.close()

    print("  ✓ Amplitude distribution plots saved")


def plot_cell_psth_split_by_inflection(ax, mouse_id, roi, inflection_trial, reward_group='R+'):
    """
    Plot PSTH of whisker stimulus responses split by inflection point.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mouse_id : str
        Mouse identifier§
    roi : int
        Cell ROI number
    inflection_trial : int
        Trial index of sigmoid inflection
    reward_group : str, optional
        Reward group ('R+' or 'R-') for color coding (default: 'R+')
    """

    # Load xarray data (baseline-subtracted)¨
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=False
    )

    # Filter for this cell
    xarr_cell = xarr.sel(cell=xarr['roi'] == roi)
    xarr_cell = xarr_cell.sel(trial=xarr_cell['day'] == 0)

    # Filter for whisker stimulus trials (whisker_stim == 1)
    whisker_trials = xarr_cell.sel(trial=xarr_cell['whisker_stim'] == 1)

    # Get trial_w indices
    trial_w_indices = whisker_trials['trial_w'].values

    # Split trials by inflection point
    before_mask = trial_w_indices <= inflection_trial
    after_mask = trial_w_indices > inflection_trial

    trials_before = whisker_trials.isel(trial=before_mask)
    trials_after = whisker_trials.isel(trial=after_mask)
    
    # Convert to DataFrame without computing mean 
    df_before = trials_before.to_dataframe(name='activity').reset_index()
    df_before['activity'] = df_before['activity'] * 100  # Convert to %ΔF/F
    df_before['period'] = f'Before inflection (n={before_mask.sum()})'

    df_after = trials_after.to_dataframe(name='activity').reset_index()
    df_after['activity'] = df_after['activity'] * 100
    df_after['period'] = f'After inflection (n={after_mask.sum()})'

    # Combine
    df_combined = pd.concat([df_before, df_after], ignore_index=True)

    # Use reward-based colors for after-inflection trace
    if reward_group == 'R+':
        colors = ['gray', reward_palette[1]]  # Gray before, green after (R+)
    else:  # R-
        colors = ['gray', reward_palette[0]]  # Light gray before, magenta after (R-)

    # Plot with seaborn - let it compute mean and CI across trials
    sns.lineplot(
        data=df_combined, x='time', y='activity', hue='period',
        errorbar='ci', ax=ax, palette=colors
    )

    # Styling
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Stimulus onset')
    ax.set_xlabel('Time from stimulus (s)', fontsize=9)
    ax.set_ylabel('Activity (%ΔF/F)', fontsize=9)
    ax.set_title('Whisker Stimulus Response', fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    sns.despine(ax=ax)


def create_cell_pdf_report(results_df, output_dir, pdf_name, n_cells=50):
    """
    Create PDF report with individual cell plots showing raw data and sigmoid fits.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory
    pdf_name : str
        Name of PDF file to create
    n_cells : int
        Number of top cells to include in report (default: 50)
    """
    print(f"\n  Generating {pdf_name} for top {n_cells} cells (sorted by amplitude)...")

    # Filter for significant cells and sort by amplitude
    results_significant = results_df[results_df['p_value'] < ALPHA]
    results_sorted = results_significant.sort_values('amplitude', ascending=False).head(n_cells)

    # Create PDF
    pdf_path = os.path.join(output_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for idx, (_, row) in enumerate(results_sorted.iterrows()):
            print(f"  Plotting cell {idx+1}/{len(results_sorted)}: {row['mouse_id']}_{row['roi']}")

            # Load raw data for this cell
            responses, trial_indices, roi_ids = load_day0_data(
                row['mouse_id'], RESPONSE_TYPE, RESPONSE_WIN
            )

            # Find this cell's index
            cell_idx = np.where(roi_ids == row['roi'])[0][0]
            y = responses[cell_idx, :]
            x = trial_indices

            # Create figure
            fig = plt.figure(figsize=(12, 8), dpi=150)
            gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

            # Main plot: Raw data + fitted model
            ax_main = fig.add_subplot(gs[0, :])

            # Plot raw data
            color = reward_palette[1] if row['reward_group'] == 'R+' else reward_palette[0]
            ax_main.scatter(x, y * 100, alpha=0.5, s=30, color=color, label='Raw data')

            # Fit sigmoid model
            sigmoid_fit = fit_sigmoid_model(x, y)

            if sigmoid_fit is not None and sigmoid_fit.get('fit_success', False):
                x_fit = sigmoid_fit['x_clean']
                y_fit = sigmoid_fit['predictions']
                ax_main.plot(x_fit, y_fit * 100, 'darkorange', linewidth=3, label='Sigmoid fit')

            # Add vertical line for behavioral learning trial
            if 'learning_trial' in row and not pd.isna(row['learning_trial']):
                learning_trial = row['learning_trial']
                ax_main.axvline(learning_trial, color='black', linestyle='-',
                                linewidth=2, alpha=0.8, label='Behavioral learning trial')

            ax_main.set_xlabel('Whisker Trial Number (trial_w)', fontsize=12)
            ax_main.set_ylabel('Response (ΔF/F0 %)', fontsize=12)
            ax_main.legend(loc='best', fontsize=10)
            ax_main.grid(True, alpha=0.3)
            sns.despine(ax=ax_main)

            # Info panel
            ax_info = fig.add_subplot(gs[1, 0])
            ax_info.axis('off')

            # Format learning trial info
            learning_trial_text = ""
            if 'learning_trial' in row and not pd.isna(row['learning_trial']):
                learning_trial_text = f"""
Behavioral Learning:
─────────────────
Learning trial: {row['learning_trial']:.0f}
Inflection rel. to learning: {row['inflection_relative']:.1f} trials
"""

            info_text = f"""
Cell Information:
─────────────────
Mouse ID: {row['mouse_id']}
ROI: {row['roi']}
Reward Group: {row['reward_group']}
LMI: {row['lmi']:.3f}
LMI p: {row['lmi_p']:.3f}

Plasticity Metrics:
─────────────────
Amplitude: {row['amplitude']:.2f}
p-value: {row['p_value']:.4e}
Pseudo-R²: {row['pseudo_r2']:.4f}
Significant: {'YES' if row['p_value'] < ALPHA else 'NO'}

Sigmoid Parameters:
─────────────────
Inflection: {row['inflection']:.1f} trials
Baseline: {row['baseline']:.2f}
Max: {row['max_val']:.2f}
Slope param: {row['slope_param']:.2f}
{learning_trial_text}
            """

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

            # PSTH plot (whisker responses before/after inflection)
            ax_psth = fig.add_subplot(gs[1, 1])
            inflection_trial = row['inflection']
            plot_cell_psth_split_by_inflection(ax_psth, row['mouse_id'], row['roi'],
                                                inflection_trial,
                                                reward_group=row['reward_group'])

            # Overall title
            fig.suptitle(f'Cell Plasticity Report #{idx+1} - {row["mouse_id"]}_{row["roi"]} ({row["reward_group"]})',
                        fontsize=14, fontweight='bold')

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  ✓ PDF report saved to: {pdf_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("SINGLE-CELL PLASTICITY ANALYSIS - DAY 0")
    print("="*70)

    # Load mice list
    _, _, mice, db = io.select_sessions_from_db(
        io.db_path, io.nwb_dir, two_p_imaging='yes'
    )

    print(f"\nProcessing {len(mice)} mice in parallel using {N_CORES} cores...")

    # Process all mice in parallel
    all_results = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_mouse)(
            mouse_id,
            response_type=RESPONSE_TYPE,
            response_win=RESPONSE_WIN,
            min_trials=MIN_TRIALS,
            amplitude_type=AMPLITUDE_TYPE
        )
        for mouse_id in mice
    )

    # Filter out None results and empty dataframes
    all_results = [r for r in all_results if r is not None and len(r) > 0]

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Add LMI information
    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
    results_df = results_df.merge(
        lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p']],
        on=['mouse_id', 'roi'],
        how='inner'
    )

    print(f"\nTotal cells before LMI filtering: {len(results_df)}")

    # Filter for LMI-significant cells only
    lmi_positive = results_df[results_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD].copy()
    lmi_negative = results_df[results_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD].copy()

    print(f"LMI+ cells (p >= {LMI_POSITIVE_THRESHOLD}): {len(lmi_positive)}")
    print(f"LMI- cells (p <= {LMI_NEGATIVE_THRESHOLD}): {len(lmi_negative)}")

    # Add LMI sign column
    lmi_positive['lmi_sign'] = 'Positive'
    lmi_negative['lmi_sign'] = 'Negative'

    # Combine for saving
    results_lmi = pd.concat([lmi_positive, lmi_negative], ignore_index=True)

    # Load and merge behavioral learning trial data
    learning_path = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    )
    learning_df = pd.read_csv(learning_path)
    learning_df = learning_df[['mouse_id', 'learning_trial']].dropna(subset=['learning_trial']).drop_duplicates()

    # Merge learning_trial into results
    results_lmi = results_lmi.merge(learning_df, on='mouse_id', how='left')

    # Compute inflection relative to learning trial
    results_lmi['inflection_relative'] = results_lmi['inflection'] - results_lmi['learning_trial']

    # Save results
    csv_path = os.path.join(OUTPUT_DIR, 'plasticity_results_lmi_cells.csv')
    results_lmi.to_csv(csv_path, index=False)
    print(f"\nSaved LMI-filtered results to {csv_path}")

    # Quantify proportions
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (LMI-SIGNIFICANT CELLS)")
    print("="*70)

    n_total = len(results_lmi)
    n_significant = np.sum(results_lmi['p_value'] < ALPHA)

    print(f"\nTotal LMI-significant cells: {n_total}")
    print(f"  - LMI+ cells: {len(lmi_positive)}")
    print(f"  - LMI- cells: {len(lmi_negative)}")
    print(f"\nCells with significant plasticity (p < {ALPHA}): {n_significant} ({100*n_significant/n_total:.1f}%)")

    # By reward group and LMI sign
    print("\nBy group:")
    for lmi_sign in ['Positive', 'Negative']:
        for group in ['R+', 'R-']:
            df_subset = results_lmi[(results_lmi['lmi_sign'] == lmi_sign) &
                                    (results_lmi['reward_group'] == group)]
            n_subset = len(df_subset)
            n_sig_subset = np.sum(df_subset['p_value'] < ALPHA)
            if n_subset > 0:
                print(f"  LMI{lmi_sign[0]}, {group}: {n_sig_subset}/{n_subset} ({100*n_sig_subset/n_subset:.1f}%) significant")

    # Mean amplitude (significant cells only)
    sig_cells = results_lmi[results_lmi['p_value'] < ALPHA]
    if len(sig_cells) > 0:
        print("\nMean amplitude (significant cells only):")
        for lmi_sign in ['Positive', 'Negative']:
            for group in ['R+', 'R-']:
                df_subset = sig_cells[(sig_cells['lmi_sign'] == lmi_sign) &
                                      (sig_cells['reward_group'] == group)]
                if len(df_subset) > 0:
                    mean_amp = df_subset['amplitude'].mean()
                    std_amp = df_subset['amplitude'].std()
                    print(f"  LMI{lmi_sign[0]}, {group}: {mean_amp:.4f} ± {std_amp:.4f}")

    # Inflection timing analysis
    print("\n" + "="*70)
    print("INFLECTION TIMING RELATIVE TO BEHAVIORAL LEARNING")
    print("="*70)
    sig_with_learning = sig_cells.dropna(subset=['learning_trial'])
    if len(sig_with_learning) > 0:
        for reward in ['R+', 'R-']:
            subset = sig_with_learning[sig_with_learning['reward_group'] == reward]
            if len(subset) > 0:
                mean_rel = subset['inflection_relative'].mean()
                std_rel = subset['inflection_relative'].std()
                median_rel = subset['inflection_relative'].median()
                print(f"\n{reward}: mean={mean_rel:.2f} ± {std_rel:.2f} trials, "
                      f"median={median_rel:.2f} trials (n={len(subset)})")

                # Report how many cells have inflection before/after learning
                before = (subset['inflection_relative'] < 0).sum()
                after = (subset['inflection_relative'] >= 0).sum()
                print(f"  Before learning: {before} ({100*before/len(subset):.1f}%)")
                print(f"  After learning: {after} ({100*after/len(subset):.1f}%)")
    else:
        print("\nNo cells with learning trial data available.")

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_distributions(results_lmi, OUTPUT_DIR)
    plot_amplitude_distributions_by_lmi(results_lmi, OUTPUT_DIR, alpha=ALPHA)

    # Generate 4 separate PDF reports
    print("\n" + "="*70)
    print("GENERATING PDF REPORTS (4 SEPARATE FILES)")
    print("="*70)

    # R+ LMI+
    subset = results_lmi[(results_lmi['reward_group'] == 'R+') &
                         (results_lmi['lmi_sign'] == 'Positive')]
    if len(subset) > 0:
        create_cell_pdf_report(subset, OUTPUT_DIR,
                               'plasticity_R+_LMI_positive.pdf',
                               n_cells=min(50, len(subset)))
    else:
        print("  No R+ LMI+ cells found")

    # # R+ LMI-
    # subset = results_lmi[(results_lmi['reward_group'] == 'R+') &
    #                      (results_lmi['lmi_sign'] == 'Negative')]
    # if len(subset) > 0:
    #     create_cell_pdf_report(subset, OUTPUT_DIR,
    #                            'plasticity_R+_LMI_negative.pdf',
    #                            n_cells=min(50, len(subset)))
    # else:
    #     print("  No R+ LMI- cells found")

    # # R- LMI+
    # subset = results_lmi[(results_lmi['reward_group'] == 'R-') &
    #                      (results_lmi['lmi_sign'] == 'Positive')]
    # if len(subset) > 0:
    #     create_cell_pdf_report(subset, OUTPUT_DIR,
    #                            'plasticity_R-_LMI_positive.pdf',
    #                            n_cells=min(50, len(subset)))
    # else:
    #     print("  No R- LMI+ cells found")

    # # R- LMI-
    # subset = results_lmi[(results_lmi['reward_group'] == 'R-') &
    #                      (results_lmi['lmi_sign'] == 'Negative')]
    # if len(subset) > 0:
    #     create_cell_pdf_report(subset, OUTPUT_DIR,
    #                            'plasticity_R-_LMI_negative.pdf',
    #                            n_cells=min(50, len(subset)))
    # else:
    #     print("  No R- LMI- cells found")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
