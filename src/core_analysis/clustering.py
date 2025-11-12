import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
from rastermap import Rastermap, utils
from scipy.stats import zscore

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging 
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from scipy.stats import mannwhitneyu
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d


def make_mouse_vector(raster_dict, isort):
    # Assign a color to each mouse_id
    # Construct vector of mouse_id indicating mouse_id of each neuron.
    mouse_ids = []
    for i, mouse_id in enumerate(raster_dict['mouse_id']):
        n_cells = raster_dict['activity'][i].shape[0]
        mouse_ids.append(np.full(n_cells, mouse_id))
    mouse_id_vector = np.concatenate(mouse_ids, axis=0)
    
    # Use a custom palette: first half warm (reds/oranges), last half cold (blues)
    mouse_sorted = mouse_id_vector[isort]
    unique_mouse_ids = np.unique(mouse_id_vector)
    n = len(unique_mouse_ids)
    n_warm = len([m for m in unique_mouse_ids if m.startswith('AR')]) 
    n_cold = n - n_warm  # Count cold colors (e.g., S2, S1)
    warm_colors = sns.color_palette("autumn", n_warm)
    cold_colors = sns.color_palette("winter", n_cold)[::-1]
    
    mouse_colors = list(warm_colors) + list(cold_colors)
    mouse_id_to_color = {mid: mouse_colors[i] for i, mid in enumerate(unique_mouse_ids)}
    mouse_color_list = [mouse_id_to_color[mid] for mid in mouse_sorted]
    
    return mouse_color_list, unique_mouse_ids, mouse_id_to_color


def make_reward_vector(raster_dict, isort):
    
    # Construct vector of reward groups indicating reward group of each neuron.
    reward_groups = np.array(raster_dict['reward_group'])
    reward_vector = []
    for i, gp in enumerate(reward_groups):
        if gp == 'R+':
            reward_vector.append(np.ones(raster_dict['activity'][i].shape[0]))
        elif gp == 'R-':
            reward_vector.append(np.zeros(raster_dict['activity'][i].shape[0]))
    reward_vector = np.concatenate(reward_vector, axis=0)
    reward_sorted = reward_vector[isort]
    
    return reward_sorted


def make_cluster_vector(clusters, isort):

    cluster_sorted = clusters[isort]
    
    # Ensure the colormap has enough colors by repeating if necessary
    n_clusters = np.unique(cluster_sorted).size
    base_cmap = sns.color_palette('Dark2', 8)  # Dark2 has 8 base colors
    if n_clusters > 8:
        repeated_colors = (base_cmap * ((n_clusters // 8) + 1))[:n_clusters]
    else:
        repeated_colors = base_cmap[:n_clusters]
    cluster_cmap = colors.ListedColormap(repeated_colors)

    return cluster_sorted, cluster_cmap


def plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                    cluster_cmap, cluster_sorted, reward_sorted, reward_cmap,
                    nmappingtrials, nlearningtrials, vmin=0, vmax=0.4, figsize=(12, 20),
                    mice_subset=None):
    """
    Optionally plot only a subset of mice by providing mice_subset (list of mouse IDs).
    """
    # If a subset of mice is provided, filter all relevant arrays
    if mice_subset is not None:
        # Build mouse_id_vector from mouse_color_list and unique_mouse_ids
        color_to_mouse = {tuple(np.round(np.array(v), 6)): k for k, v in mouse_id_to_color.items()}
        mouse_id_vector = [color_to_mouse[tuple(np.round(np.array(color), 6))] for color in mouse_color_list]
        mouse_id_vector = np.array(mouse_id_vector)
        mask = np.isin(mouse_id_vector, mice_subset)
        sn = sn[mask]
        mouse_color_list = [mouse_color_list[i] for i in np.where(mask)[0]]
        cluster_sorted = np.array(cluster_sorted)[mask]
        reward_sorted = np.array(reward_sorted)[mask]
        # Only keep unique_mouse_ids and mouse_id_to_color for the subset
        unique_mouse_ids = [mid for mid in unique_mouse_ids if mid in mice_subset]
        mouse_id_to_color = {mid: mouse_id_to_color[mid] for mid in unique_mouse_ids}

    # Create a common figure with five axes: left for rastermap image, then cluster color bar, reward group color bar, mouse_id color bar, right for activity colorbar
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = GridSpec(1, 5, width_ratios=[0.9, 0.025, 0.025, 0.025, 0.025], wspace=0.1)

    # Rastermap image
    ax_raster = fig.add_subplot(gs[0])
    im = ax_raster.imshow(sn, cmap="grey_r", vmin=vmin, vmax=vmax, aspect="auto")

    # Add vertical lines to separate mapping and learning trials
    edges =  [nmappingtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials]
    for x in np.cumsum(edges):
        ax_raster.axvline(x, color='grey', linestyle='-')
    # Set x-ticks at the end of each mapping or learning session
    tick_positions = np.cumsum(edges)
    ax_raster.set_xticks(tick_positions)
    ax_raster.set_xticklabels(tick_positions)

    # Cluster color bar (middle, less wide)
    ax_cbar = fig.add_subplot(gs[1])
    cbar_img = np.array(cluster_sorted)[:, None]
    ax_cbar.imshow(cbar_img, aspect="auto", cmap=cluster_cmap, vmin=0, vmax=np.unique(cluster_sorted).max())
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([])

    # Reward group color bar (next to cluster color bar)
    ax_reward = fig.add_subplot(gs[2])
    reward_img = np.array(reward_sorted)[:, None]
    ax_reward.imshow(reward_img, aspect="auto", cmap=colors.ListedColormap(reward_cmap), vmin=0, vmax=1)
    ax_reward.set_xticks([])
    ax_reward.set_yticks([])

    # Mouse ID color bar (next to reward group color bar)
    ax_mouse = fig.add_subplot(gs[3])
    mouse_img = np.array(mouse_color_list).reshape(-1, 1, 3)
    ax_mouse.imshow(mouse_img, aspect="auto")
    ax_mouse.set_xticks([])
    ax_mouse.set_yticks([])
    # Optionally, add a legend for mouse IDs
    legend_elements = [Patch(facecolor=mouse_id_to_color[mid], label=str(mid)) for mid in unique_mouse_ids]
    ax_mouse.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Mouse ID", fontsize='small', title_fontsize='small', frameon=False)

    # Activity colorbar (rightmost, for the raster image)
    ax_activity_cbar = fig.add_subplot(gs[4])
    cb = plt.colorbar(im, cax=ax_activity_cbar)
    box = ax_activity_cbar.get_position()
    ax_activity_cbar.set_position([box.x0 + 0.18, box.y0, box.width, box.height])


def plot_rastermap_per_mouse(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                                cluster_cmap, cluster_sorted, reward_sorted, reward_cmap,
                                nmappingtrials, nlearningtrials, pdf_path, figsize=(12, 10)):
    """
    Plot a rastermap for each mouse and save all plots to a multi-page PDF.

    Args:
        sn: (n_neurons, n_timepoints) array, sorted activity.
        mouse_color_list: list of RGB tuples, color for each neuron (sorted).
        unique_mouse_ids: list of mouse IDs.
        mouse_id_to_color: dict mapping mouse_id to color.
        cluster_cmap: matplotlib colormap for clusters.
        cluster_sorted: cluster assignment for each neuron (sorted).
        reward_sorted: reward group for each neuron (sorted).
        reward_cmap: color palette for reward groups.
        nmappingtrials: int, number of mapping trials.
        nlearningtrials: int, number of learning trials.
        pdf_path: str, path to output PDF file.
    """
    # Build mouse_id vector for sorted neurons
    mouse_id_vector = []
    for i, mouse_id in enumerate(unique_mouse_ids):
        n_cells = np.sum(np.array(mouse_color_list) == mouse_id_to_color[mouse_id], axis=1)
        mouse_id_vector.extend([mouse_id] * n_cells.sum())
    mouse_id_vector = np.array(mouse_id_vector)
    # But above is not robust; better to reconstruct from the original raster_dict if available.
    # Instead, reconstruct from mouse_color_list and unique_mouse_ids:
    # For each neuron, find which mouse_id its color matches.
    mouse_id_vector = []
    color_to_mouse = {tuple(np.round(np.array(v), 6)): k for k, v in mouse_id_to_color.items()}
    for color in mouse_color_list:
        mouse_id_vector.append(color_to_mouse[tuple(np.round(np.array(color), 6))])
    mouse_id_vector = np.array(mouse_id_vector)

    with PdfPages(pdf_path) as pdf:
        for mouse_id in unique_mouse_ids:
            idx = np.where(mouse_id_vector == mouse_id)[0]
            if len(idx) == 0:
                continue
            sn_mouse = sn[idx]
            cluster_sorted_mouse = np.array(cluster_sorted)[idx]
            reward_sorted_mouse = np.array(reward_sorted)[idx]
            mouse_color_list_mouse = [mouse_id_to_color[mouse_id]] * len(idx)
            # Plot
            fig = plot_rastermap(
                sn_mouse,
                mouse_color_list_mouse,
                [mouse_id],
                mouse_id_to_color,
                cluster_cmap,
                cluster_sorted_mouse,
                reward_sorted_mouse,
                reward_cmap,
                nmappingtrials,
                nlearningtrials,
                vmin=0,
                vmax=0.4,
                figsize=figsize
            )
            fig.suptitle(f"Mouse {mouse_id}", y=0.98)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                          nlearningtrials, mouse_color_list=None,
                          unique_mouse_ids=None, mouse_id_to_color=None,
                          mice_subset=None, ymin=None, ymax=None):
    """
    Plot average activity for each cluster in separate subplots.
    Optionally, plot only a subset of mice by providing mice_subset (list of mouse IDs).
    """
    # If a subset of mice is provided, filter sn and cluster_sorted accordingly
    if mice_subset is not None and mouse_color_list is not None and unique_mouse_ids is not None and mouse_id_to_color is not None:
        # Build mouse_id_vector from mouse_color_list and unique_mouse_ids
        color_to_mouse = {tuple(np.round(np.array(v), 6)): k for k, v in mouse_id_to_color.items()}
        mouse_id_vector = [color_to_mouse[tuple(np.round(np.array(color), 6))] for color in mouse_color_list]
        mouse_id_vector = np.array(mouse_id_vector)
        mask = np.isin(mouse_id_vector, mice_subset)
        sn = sn[mask]
        cluster_sorted = np.array(cluster_sorted)[mask]

    unique_clusters = np.unique(cluster_sorted)
    n_clusters = unique_clusters.size
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]  # Ensure axes is always iterable

    edges = [nmappingtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials]
    tick_positions = np.cumsum(edges)

    for i, clust in enumerate(unique_clusters):
        idx = np.where(cluster_sorted == clust)[0]
        if len(idx) == 0:
            continue
        cluster_activity = sn[idx]
        mean_activity = cluster_activity.mean(axis=0)
        ax = axes[i]
        ax.plot(mean_activity, label=f'Cluster {clust}', color=cluster_cmap(clust))
        # Add vertical lines to separate mapping and learning trials
        for x in tick_positions:
            ax.axvline(x, color='grey', linestyle='-')
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_ylabel('Mean activity')
        ax.set_title(f'Cluster {clust} (n={len(idx)})')
        ax.legend(loc='upper right', fontsize='small')
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)

    axes[-1].set_xlabel('Time')
    fig.suptitle('Neuron-averaged activity per cluster', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])


# Load and bin psth data.

# Parameters.
sampling_rate = 30
win = (0, 0.300)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']
avg_time_win = True
bin_size = 3  # Binning size in frames.
nmappingtrials = 40
nlearningtrials = 60
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

# Load the data.
# --------------

raster_dict = {'mouse_id': [],
                'activity': [],
                'roi': [],
                'cell_type': [],
                'reward_group': [],
        }

for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr_mapping = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    print('Substracting baseline...')
    xarr_mapping = utils_imaging.substract_baseline(xarr_mapping, 2, baseline_win)
    
    print('Selecting trials...')
    # Select days.
    xarr_mapping = xarr_mapping.sel(trial=xarr_mapping['day'].isin(days))
    # Select n mapping trials.
    xarr_mapping = xarr_mapping.groupby('day').apply(lambda x: x.isel(trial=slice(-nmappingtrials, None)))
    # Select PSTH trace length.
    xarr_mapping = xarr_mapping.sel(time=slice(win[0], win[1]))
    
    enough_trials = True
    for day in [-2, -1, 0, 1, 2]:
        n_trials = xarr_mapping.sel(trial=xarr_mapping['day'] == day)['trial'].size
        if n_trials < nmappingtrials:
            print(f'Warning: not enough mapping trials for mouse {mouse_id} on day {day}. Expected at least {nmappingtrials}, got {n_trials}.')
            enough_trials = False
    if not enough_trials:
        print(f'Skipping mouse {mouse_id} due to insufficient trials.')
        continue
    
    if avg_time_win:
        # Average the time window.
        xarr_mapping = xarr_mapping.mean(dim='time', keep_attrs=True)
    else:
        print('Binning time...')
        # Bin time with blocks of 3 frames (fast method).
        arr = xarr_mapping.values
        n_bins = arr.shape[-1] // bin_size
        arr_binned = arr[..., :n_bins*bin_size].reshape(*arr.shape[:-1], n_bins, bin_size).mean(axis=-1)
        # Update xarr_mapping with binned data and new time coordinates
        new_time = xarr_mapping['time'].values[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
        xarr_mapping = xr.DataArray(
            arr_binned,
            dims=xarr_mapping.dims,
            coords={**{k: v for k, v in xarr_mapping.coords.items() if k != 'time'}, 'time': new_time},
            name=xarr_mapping.name
        )

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr_learning = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)
    
    print('Selecting trials...')
    # Select days.
    xarr_learning = xarr_learning.sel(trial=xarr_learning['day'].isin([0,1,2]))
    # Select PSTH trace length.
    xarr_learning = xarr_learning.sel(time=slice(win[0], win[1]))
    # Select whisker trials.
    xarr_learning = xarr_learning.sel(trial=xarr_learning['whisker_stim']==1)
    # Select n learning trials.
    xarr_learning = xarr_learning.sel(trial=xarr_learning['trial_w'] <= nlearningtrials)
    # Check that the number of trials is correct.
    # Check that each day has at least nlearningtrials
    enough_trials = True
    for day in [0, 1, 2]:
        n_trials = xarr_learning.sel(trial=xarr_learning['day'] == day)['trial'].size
        if n_trials < nlearningtrials:
            print(f'Warning: not enough trials for mouse {mouse_id} on day {day}. Expected at least {nlearningtrials}, got {n_trials}.')
            enough_trials = False
    if not enough_trials:
        print(f'Skipping mouse {mouse_id} due to insufficient trials.')
        continue
    
    if avg_time_win:
        # Average the time window.
        xarr_learning = xarr_learning.mean(dim='time', keep_attrs=True)
    else:
        print('Binning time...')
        # Bin time with blocks of 3 frames (fast method).
        arr = xarr_learning.values
        n_bins = arr.shape[-1] // bin_size
        arr_binned = arr[..., :n_bins*bin_size].reshape(*arr.shape[:-1], n_bins, bin_size).mean(axis=-1)
        # Update xarr_learning with binned data and new time coordinates
        new_time = xarr_learning['time'].values[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
        xarr_learning = xr.DataArray(
            arr_binned,
            dims=xarr_learning.dims,
            coords={**{k: v for k, v in xarr_learning.coords.items() if k != 'time'}, 'time': new_time},
            name=xarr_learning.name
        )

    activity_2d = []
    # Select trials for the current day.
    xarr_day = xarr_mapping.sel(trial=xarr_mapping['day'] == -2)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_mapping.sel(trial=xarr_mapping['day'] == -1)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_learning.sel(trial=xarr_learning['day'] == 0)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_mapping.sel(trial=xarr_mapping['day'] == 0)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_learning.sel(trial=xarr_learning['day'] == 1)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_mapping.sel(trial=xarr_mapping['day'] == 1)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_learning.sel(trial=xarr_learning['day'] == 2)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    xarr_day = xarr_mapping.sel(trial=xarr_mapping['day'] == 2)
    activity_2d.append(xarr_day.values.reshape(xarr_day.shape[0], -1))
    
    activity_2d = np.concatenate(activity_2d, axis=1)

    # Remove artefactual cells.
    good_cells = np.any(activity_2d != 0, axis=1)
    activity_2d = activity_2d[good_cells]
    rois = xarr_mapping['roi'].values[good_cells]
    ct = xarr_mapping['cell_type'].values[good_cells]
    
    raster_dict['mouse_id'].append(mouse_id)
    raster_dict['activity'].append(activity_2d)
    raster_dict['roi'].append(rois)
    raster_dict['cell_type'].append(ct)
    raster_dict['reward_group'].append(reward_group)
    # raster_dict['trial'] = xarr_mapping['trial_w'].values

# Save raster dictionnary.
save_path = os.path.join(io.processed_dir, 'rasters', 'raster_dict_singlebin_300ms.npy')
np.save(save_path, raster_dict)





# Run clustering and plot rasters.
# ################################

# Load raster data.
# -----------------

raster_path = os.path.join(io.processed_dir, 'rasters', 'raster_dict_singlebin_180ms.npy')
raster_dict = np.load(raster_path, allow_pickle=True).item()

# # Remove mice with wrong number of trials
# for key, item in raster_dict.items():
#     raster_dict[key] = [x for i, x in enumerate(item) if i != 4]

# Select subset of mice.
# mice_gp = mice_groups['good_day0']
# mice_gp = mice_groups['meh_day0'] + mice_groups['bad_day0']
mice_gp = mice
raster_dict = {k: [v[i] for i in range(len(v)) if raster_dict['mouse_id'][i] in mice_gp] for k, v in raster_dict.items()}

raster = np.concatenate(raster_dict['activity'], axis=0)

# # z-score activity.
# raster_dict['activity_zscore'] = []
# for activity in raster_dict['activity']:
#     activity = zscore(activity, axis=1, ddof=1)
#     raster_dict['activity_zscore'].append(activity)
# raster_zscore = np.concatenate(raster_dict['activity_zscore'], axis=0)

# Cluster just on mapping trials.
raster_mapping = np.concatenate([raster[:, :nmappingtrials*2],
                          raster[:, nmappingtrials*2 + nlearningtrials:nmappingtrials*3 + nlearningtrials],
                          raster[:, nmappingtrials*3 + nlearningtrials*2:nmappingtrials*4 + nlearningtrials*2],
                          raster[:, nmappingtrials*4 + nlearningtrials*2:],
                          ], axis=1)
raster_mapping = raster_mapping[~(raster_mapping==0).all(axis=1), :]  # Remove empty neurons.



raster_day0 = raster[:, nmappingtrials*2: nmappingtrials*2 + nlearningtrials]
raster_day1 = raster[:, nmappingtrials*3 + nlearningtrials: nmappingtrials*3 + nlearningtrials*2]
raster_day2 = raster[:, nmappingtrials*4 + nlearningtrials*2: nmappingtrials*4 + nlearningtrials*3]


# Clustering.
# -----------

# from sklearn.mixture import GaussianMixture

# nclusters = 3
# gmm = GaussianMixture(
#         n_components=nclusters,
#         covariance_type="full",
#         random_state=42,
#     )

# # X_0 = StandardScaler().fit_transform(raster_day0.T).T 
# # X_0 = StandardScaler().fit_transform(X_0)

# # X = StandardScaler().fit_transform(raster.T).T 
# # X = StandardScaler().fit_transform(X)

# model = gmm.fit(raster_day0)
# clusters = gmm.predict(raster_day0)  # Predict clusters
# isort = np.argsort(clusters, kind='mergesort')  # Sort neurons by cluster
# # Sort neuron according to clusters.
# sn = raster[isort]  # Sort raster according to clusters

# # # Reclustering within each cluster based only on day 0 learning activity
# # reclustered_clusters = np.zeros_like(clusters)
# # cluster_offset = 0  # To keep unique cluster labels after reclustering

# # for clust in np.unique(clusters):
# #     idx = np.where(clusters == clust)[0]
# #     if len(idx) == 0:
# #         continue
# #     # Extract day 0 learning activity for neurons in this cluster
# #     day0_activity = raster[idx, nmappingtrials*2: nmappingtrials*2 + nlearningtrials]
# #     # Remove neurons with all zeros (if any)
# #     valid = ~(day0_activity == 0).all(axis=1)
# #     idx_valid = idx[valid]
# #     day0_activity = day0_activity[valid]
# #     if day0_activity.shape[0] < 2:
# #         # Not enough neurons to cluster, assign same cluster
# #         reclustered_clusters[idx_valid] = cluster_offset
# #         cluster_offset += 1
# #         continue
# #     # Standardize
# #     X2 = StandardScaler().fit_transform(day0_activity.T).T 
# #     X2 = StandardScaler().fit_transform(X2)

# #     # Reclustering: use GMM with 2 clusters (or 1 if very few neurons)
# #     n_subclusters = 3
# #     gmm_sub = GaussianMixture(n_components=n_subclusters, covariance_type="full", random_state=42)
# #     sub_labels = gmm_sub.fit_predict(X2)
# #     # Assign new cluster labels, offset to keep unique
# #     for sub in range(n_subclusters):
# #         reclustered_clusters[idx_valid[sub_labels == sub]] = cluster_offset
# #         cluster_offset += 1

# # # Now reclustered_clusters contains the new cluster assignments after day 0 reclustering
# # # You may want to update isort and sn accordingly:
# # isort = np.argsort(reclustered_clusters, kind='mergesort')
# # sn = raster[isort]
# # clusters = reclustered_clusters





# # Create vectors for mouse_id, reward group, and clusters.
# mouse_color_list, unique_mouse_ids, mouse_id_to_color = make_mouse_vector(raster_dict, isort)
# reward_sorted = make_reward_vector(raster_dict, isort)
# cluster_sorted, cluster_cmap = make_cluster_vector(clusters, isort)
# # Plot rastermap.
# plot_rastermap(X[isort], mouse_color_list, unique_mouse_ids, mouse_id_to_color,
#                    cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
#                    nmappingtrials, nlearningtrials, vmin=0, vmax=0.4)

# plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
#                       nlearningtrials, mouse_color_list=mouse_color_list,
#                       unique_mouse_ids=unique_mouse_ids, mouse_id_to_color=mouse_id_to_color)





# Run rastermap
X = StandardScaler().fit_transform(raster.T).T 
X = StandardScaler().fit_transform(X)
model = Rastermap(n_clusters=3, # number of clusters to compute
                  n_PCs=128, # number of PCs to use
                  locality=0., # locality in sorting to find sequences (this is a value from 0-1)
                  grid_upsample=10, # default value, 10 is good for large recordings
                ).fit(X)
embedding = model.embedding # neurons x 1
isort = model.isort
clusters = model.embedding_clust

# Binning neurons for visibility.
nbin = 1 # number of neurons to bin over
sn = utils.bin1d(raster[isort], bin_size=nbin, axis=0)

# Create vectors for mouse_id, reward group, and clusters.
mouse_color_list, unique_mouse_ids, mouse_id_to_color = make_mouse_vector(raster_dict, isort)
reward_sorted = make_reward_vector(raster_dict, isort)
cluster_sorted, cluster_cmap = make_cluster_vector(clusters, isort)


# Plot rastermap.
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, vmin=0, vmax=0.4)
# output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
# output_dir = io.adjust_path_to_host(output_dir)
# svg_file = f'rastermap_ncluster_{nclusters}.svg'
# plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                      nlearningtrials)
# output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
# output_dir = io.adjust_path_to_host(output_dir)
# svg_file = f'cluster_averages_ncluster_{nclusters}.svg'
# plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)






# Raster plot for each mouse.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
output_dir = io.adjust_path_to_host(output_dir)
pdf_path = os.path.join(output_dir, f'rastermap_per_mouse_ncluster_{nclusters}.pdf')
plot_rastermap_per_mouse(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                                cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                                nmappingtrials, nlearningtrials, pdf_path)



# Get the rewarded mice (reward_group == 'R+')
mice_subset = [mouse_id for mouse_id, reward_group in zip(raster_dict['mouse_id'], raster_dict['reward_group']) if reward_group == 'R+']
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, vmin=0, vmax=0.4, mice_subset=mice_subset)

ymin= -0.02
ymax= 0.12
plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                      nlearningtrials,  mouse_color_list=mouse_color_list,
                      unique_mouse_ids=unique_mouse_ids, mouse_id_to_color=mouse_id_to_color,
                      mice_subset=mice_subset, ymin=ymin, ymax=ymax)

# Get the rewarded mice (reward_group == 'R+')
mice_subset = [mouse_id for mouse_id, reward_group in zip(raster_dict['mouse_id'], raster_dict['reward_group']) if reward_group == 'R+']
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, vmin=0, vmax=0.4, mice_subset=mice_subset)

plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                      nlearningtrials,  mouse_color_list=mouse_color_list,
                      unique_mouse_ids=unique_mouse_ids, mouse_id_to_color=mouse_id_to_color,
                      mice_subset=mice_subset, ymin=ymin, ymax=ymax)


# # Save the figure.
# save_path = os.path.join(io.processed_dir, 'rasters', 'rastermap_singlebin_180ms.pdf')
# with PdfPages(save_path) as pdf:
#     plt.savefig(pdf, format='pdf', bbox_inches='tight')
#     plt.close()




# Rastermap on day 0 activity.
# ############################


# Load raster data.
save_path = os.path.join(io.processed_dir, 'rasters', 'raster_dict_singlebin_180ms.npy')
raster_dict = np.load(save_path, allow_pickle=True).item()

# mice_gp = mice
mice_gp = mice_groups['good_day0']
# mice_gp = mice_groups['meh_day0'] + mice_groups['bad_day0']
# mice_gbadmice

raster_dict = {k: [v[i] for i in range(len(v)) if raster_dict['mouse_id'][i] in mice_gp] for k, v in raster_dict.items()}

# # z-score activity.
# raster_dict['activity_zscore'] = []
# for activity in raster_dict['activity']:
#     activity = zscore(activity, axis=1, ddof=1)
#     raster_dict['activity_zscore'].append(activity)

# # Select subset of mice.
# mice = mice_groups['gradual_day0']
# raster_dict = {k: [v[i] for i in range(len(v)) if raster_dict['mouse_id'][i] in mice] for k, v in raster_dict.items()}

raster = np.concatenate(raster_dict['activity'], axis=0)
raster[np.isnan(raster)] = 1
# raster_zscore = np.concatenate(raster_dict['activity_zscore'], axis=0)

np.all(raster==1, axis=0).sum() # Check number of empty neurons

raster_day0 = raster[:, nmappingtrials*2: nmappingtrials*2 + nlearningtrials]
raster_day1 = raster[:, nmappingtrials*3 + nlearningtrials: nmappingtrials*3 + nlearningtrials*2]
raster_day2 = raster[:, nmappingtrials*4 + nlearningtrials*2: nmappingtrials*4 + nlearningtrials*3]

# Weird cell.
raster_day2 = raster_day2[np.std(raster_day2, axis=1)!=0] # Check number of empty neurons


# Run rastermap
n_clusters = 3
model = Rastermap(n_clusters=n_clusters, # number of clusters to compute
                  n_PCs=64, # number of PCs to use
                  locality=0., # locality in sorting to find sequences (this is a value from 0-1)
                  grid_upsample=0, # default value, 10 is good for large recordings
                ).fit(raster_day0)
embedding = model.embedding # neurons x 1
isort = model.isort
clusters = model.embedding_clust
nclusters = np.unique(clusters).size


# Binning neurons for visibility.
nbin = 1 # number of neurons to bin over
sn = utils.bin1d(raster[isort], bin_size=nbin, axis=0)

# Create vectors for mouse_id, reward group, and clusters.
mouse_color_list, unique_mouse_ids, mouse_id_to_color = make_mouse_vector(raster_dict, isort)
reward_sorted = make_reward_vector(raster_dict, isort)
cluster_sorted, cluster_cmap = make_cluster_vector(clusters, isort)

# # Save clustering results.
# # Associate each (mouse_id, roi) with a cluster_id and save as a DataFrame
# mouse_id_vector = []
# roi_vector = []
# for i, mouse_id in enumerate(raster_dict['mouse_id']):
#     rois = raster_dict['roi'][i]
#     n_cells = rois.shape[0]
#     mouse_id_vector.extend([mouse_id] * n_cells)
#     roi_vector.extend(rois)
# mouse_id_vector = np.array(mouse_id_vector)
# roi_vector = np.array(roi_vector)

# cluster_df = pd.DataFrame({
#     'mouse_id': mouse_id_vector,
#     'roi': roi_vector,
#     'cluster_id': clusters
# })

# # Save to CSV
# output_dir = '/mnt/lsens-analysis/Anthony_Renard/data_processed/clustering/'
# output_dir = io.adjust_path_to_host(output_dir)
# csv_path = os.path.join(output_dir, f'cluster_id_day0_learning_nclust_{nclusters}.csv')
# cluster_df.to_csv(csv_path, index=False)



# Plot rastermap.
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, figsize=(12, 10), vmin=0, vmax=0.4)
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'rastermap_on_day0_ncluster_{nclusters}_goodmice.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials, nlearningtrials)
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'cluster_averages_on_day0_ncluster_{nclusters}_goodmice.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)




# Raster plot for each mouse.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
output_dir = io.adjust_path_to_host(output_dir)
pdf_path = os.path.join(output_dir, f'rastermap_per_mouse_onday0_ncluster_{nclusters}.pdf')
plot_rastermap_per_mouse(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                                cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                                nmappingtrials, nlearningtrials, pdf_path)


# Get the rewarded mice (reward_group == 'R+')
mice_subset = [mouse_id for mouse_id, reward_group in zip(raster_dict['mouse_id'], raster_dict['reward_group']) if reward_group == 'R+']
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, vmin=0, vmax=0.4, mice_subset=mice_subset)

ymin= -0.02
ymax= 0.12
plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                      nlearningtrials,  mouse_color_list=mouse_color_list,
                      unique_mouse_ids=unique_mouse_ids, mouse_id_to_color=mouse_id_to_color,
                      mice_subset=mice_subset, ymin=ymin, ymax=ymax)



# Get the rewarded mice (reward_group == 'R+')
mice_subset = [mouse_id for mouse_id, reward_group in zip(raster_dict['mouse_id'], raster_dict['reward_group']) if reward_group == 'R-']
plot_rastermap(sn, mouse_color_list, unique_mouse_ids, mouse_id_to_color,
                   cluster_cmap, cluster_sorted, reward_sorted, reward_palette,
                   nmappingtrials, nlearningtrials, vmin=0, vmax=0.4, mice_subset=mice_subset)

plot_cluster_averages(cluster_sorted, sn, cluster_cmap, nmappingtrials,
                      nlearningtrials,  mouse_color_list=mouse_color_list,
                      unique_mouse_ids=unique_mouse_ids, mouse_id_to_color=mouse_id_to_color,
                      mice_subset=mice_subset, ymin=ymin, ymax=ymax)


# Learning dimension for each cluster.
# ------------------------------------

# Define it as differnce vector between pre and post mapping trials
# Project each learning trial onto this vector.

def compute_learning_dimension(cluster_data, nmappingtrials, nlearningtrials):
    # Compute average activity for day -1 mapping and day +2 mapping.
    pre = cluster_data[:, 0:2*nmappingtrials]
    pre = pre.mean(axis=1)
    map_day1 = cluster_data[:, 3*nmappingtrials+2*nlearningtrials:3*nmappingtrials+3*nlearningtrials]
    map_day2 = cluster_data[:, 4*nmappingtrials+3*nlearningtrials:]
    post = np.concatenate((map_day1, map_day2), axis=1).mean(axis=1)

    # Compute the learning dimension as the difference vector.
    learning_dimension = post - pre
    
    return learning_dimension

# Project learning trial from day 0 for each cluster onto learning dimension.
learning_dimension_projections = {}
for clust in np.unique(cluster_sorted):
    idx = np.where(cluster_sorted == clust)[0]
    if len(idx) == 0:
        continue
    cluster_data = sn[idx]
    learning_dim = compute_learning_dimension(cluster_data, nmappingtrials, nlearningtrials)
    # trials_to_project =  cluster_data[:, nmappingtrials*2: nmappingtrials*2 + nlearningtrials]
    trials_to_project =  cluster_data[:, :]
    # Compute cosine similarity between each trial and the learning dimension
    # Project each trial onto the learning dimension (dot product, normalized by norm of learning_dim)
    learning_dim_norm = np.linalg.norm(learning_dim)
    if learning_dim_norm == 0:
        projection = np.zeros(trials_to_project.shape[1])
    else:
        projection = np.dot(trials_to_project.T, learning_dim) / learning_dim_norm
    # projection = cosine_similarity(trials_to_project.T, learning_dim.reshape(1, -1)).flatten()
    # Smooth the projection along trials
    projection_smooth = gaussian_filter1d(projection, sigma=1.5)
    # Store the smoothed projection for the cluster.
    learning_dimension_projections[clust] = projection_smooth

# Plot the learning dimension projections for each cluster in separate subplots.
n_clusters = len(learning_dimension_projections)
fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)
if n_clusters == 1:
    axes = [axes]  # Ensure axes is always iterable

for ax, (clust, projection) in zip(axes, learning_dimension_projections.items()):
    ax.plot(projection, label=f'Cluster {clust}', color=cluster_cmap(clust))
    ax.set_ylabel('Learning dimension projection')
    ax.set_title(f'Learning dimension projection for Cluster {clust}')
    # Add vertical lines to separate mapping and learning trials
    edges = [nmappingtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials]
    for x in np.cumsum(edges):
        ax.axvline(x, color='grey', linestyle='-')
    # Set x-ticks at the end of each mapping or learning session
    tick_positions = np.cumsum(edges)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.legend(loc='upper right', fontsize='small')

axes[-1].set_xlabel('Trial')
plt.tight_layout()

# Save the figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/clustering/'
output_dir = io.adjust_path_to_host(output_dir)
pdf_path = os.path.join(output_dir, f'cluster_projection_ncluster_{nclusters}.svg')
plt.savefig(pdf_path, format='svg', dpi=300)
# Same plot across mice.

learning_dimension_projections = {}
unique_mice = np.unique(mouse_color_list, axis=0)  # Actually, mouse_color_list is colors, not IDs. Use unique_mouse_ids.

for clust in np.unique(cluster_sorted):
    idx = np.where(cluster_sorted == clust)[0]
    if len(idx) == 0:
        continue

    # For each mouse in this cluster, compute projection
    projections_per_mouse = []
    mouse_id_vector = []
    for i, mouse_id in enumerate(raster_dict['mouse_id']):
        n_cells = raster_dict['activity'][i].shape[0]
        mouse_id_vector.extend([mouse_id]*n_cells)
    mouse_id_vector = np.array(mouse_id_vector)
    mouse_ids_in_cluster = mouse_id_vector[isort][idx]
    
    for mouse_id in np.unique(mouse_ids_in_cluster):
        mouse_idx = idx[mouse_ids_in_cluster == mouse_id]
        if len(mouse_idx) == 0:
            continue
        cluster_data_mouse = sn[mouse_idx]
        if cluster_data_mouse.shape[0] < 2:
            continue  # Need at least 2 cells to compute mean
        learning_dim = compute_learning_dimension(cluster_data_mouse, nmappingtrials, nlearningtrials)
        trials_to_project = cluster_data_mouse[:, :]
        projection = cosine_similarity(trials_to_project.T, learning_dim.reshape(1, -1)).flatten()
        projection_smooth = gaussian_filter1d(projection, sigma=1)
        projections_per_mouse.append(projection_smooth)
    if len(projections_per_mouse) > 0:
        projections_per_mouse = np.stack(projections_per_mouse, axis=0)
        mean_projection = projections_per_mouse.mean(axis=0)
        sem_projection = projections_per_mouse.std(axis=0) / np.sqrt(projections_per_mouse.shape[0])
        learning_dimension_projections[clust] = (mean_projection, sem_projection)

# Plot the learning dimension projections for each cluster (mean ± SEM across mice).
fig, ax = plt.subplots(figsize=(10, 5))
for clust, (mean_projection, sem_projection) in learning_dimension_projections.items():
    ax.plot(mean_projection, label=f'Cluster {clust}', color=cluster_cmap(clust))
    ax.fill_between(np.arange(len(mean_projection)), mean_projection - sem_projection, mean_projection + sem_projection,
                    color=cluster_cmap(clust), alpha=0.2)
ax.set_xlabel('Trial')
ax.set_ylabel('Learning dimension projection')
ax.set_title('Learning dimension projection for each cluster (mean ± SEM across mice)')
edges =  [nmappingtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials, nlearningtrials, nmappingtrials]
for x in np.cumsum(edges):
    ax.axvline(x, color='grey', linestyle='-')
tick_positions = np.cumsum(edges)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_positions)



# # Save the figure.
# save_path = os.path.join(io.processed_dir, 'rasters', 'learning_dimension_projection.pdf')
# with PdfPages(save_path) as pdf:
#     plt.savefig(pdf, format='pdf', bbox_inches='tight')
#     plt.close() 


