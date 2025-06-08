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

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.behavior import compute_performance, plot_single_session
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu


# # #############################################################################
# # Comparing xarray datasets with previous tensors.
# # #############################################################################

# io.processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# mouse_id = 'AR180'
# session_id = 'AR180_20241217_160355'

# arr, mdata = imaging_utils.load_session_2p_imaging(mouse_id,
#                                                     session_id,
#                                                     io.processed_dir
#                                                     )
# # arr = imaging_utils.substract_baseline(arr, 3, ())
# arr = imaging_utils.extract_trials(arr, mdata, 'UM', n_trials=None)
# arr.shape

# # Load the xarray dataset.
# file_name = 'tensor_xarray_mapping_data.nc'
# xarray = imaging_utils.load_mouse_xarray(mouse_id, io.processed_dir, file_name)

# d = xarray.sel(trial=xarray['day'] == 2)


# #############################################################################
# 1. PSTH's.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (-0.5, 4)  
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']


_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
# mice = [m for m in mice if m not in ['AR163']]
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()
# print(mice_count)
# print(mice_count.groupby('reward_group').count().reset_index())


# Load the data.
# --------------

psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = imaging_utils.substract_baseline(xarr, 2, baseline_win)
    
    # Keep days of interest.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    
    xarr.name = 'psth'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    psth.append(xarr)
psth = pd.concat(psth)


# Grand average psth's for all cells and projection neurons.
# ##########################################################

# GF305 has baseline artefact on day -1 at auditory trials.
# data = data.loc[~data.mouse_id.isin(['GF305'])]

# mice_AR = [m for m in mice if m.startswith('AR')]
# mice_GF = [m for m in mice if m.startswith('GF') or m.startswith('MI')]
# data = data.loc[data.mouse_id.isin(mice_AR)]
# len(mice_GF)

variance = 'mice'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data = filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type',])['psth'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()

# Convert data to percent dF/F0.
data['psth'] = data['psth'] * 100

# Plot for all cells.
fig, axes = plt.subplots(1, len(days), figsize=(18, 5), sharey=True)
for j, day in enumerate(days):
    d = data.loc[data['day'] == day]
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                 hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[j], legend=False)
    axes[j].axvline(0, color='#FF9600', linestyle='--')
    axes[j].set_title(f'Day {day} - All Cells')
    axes[j].set_ylabel('DF/F0 (%)')
plt.ylim(-1, 12)
# Adjust spacing between subplots to prevent title overlap
plt.tight_layout()
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_across_days_all_cells_{variance}_long.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

# Plot for each cell type.
fig, axes = plt.subplots(2, len(days), figsize=(18, 10), sharey=True)
for i, cell_type in enumerate(['wS2', 'wM1']):
    for j, day in enumerate(days):
        d = data[(data['cell_type'] == cell_type) & (data['day'] == day)]
        sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                     hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[i, j], legend=False)
        axes[i, j].axvline(0, color='#FF9600', linestyle='--')
        axes[i, j].set_title(f'{cell_type} - Day {day}')
        axes[i, j].set_ylabel('DF/F0 (%)')
plt.ylim(-1, 16)
# Adjust spacing between subplots to prevent title overlap
plt.tight_layout()
sns.despine()

# Save figure for projection types.
svg_file = f'psth_across_days_projection_types_{variance}_long.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Individual mice PSTH's.
# -----------------------

output_dir = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = f'psth_individual_mice_baseline.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse_id in mice:
        print(f"\rProcessing {mouse_id} ({mice.index(mouse_id) + 1}/{len(mice)})", end="")
        # Plot.
        data = psth.loc[psth['day'].isin([-2, -1, 0, 1, 2])
                      & (psth['mouse_id'] == mouse_id)]
        fig, axes = plt.subplots(3, len(days), figsize=(15, 10), sharey=True)
        # Plot for all cells.
        for j, day in enumerate(days):
            d = data.loc[data['day'] == day]
            sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                        hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0, j], legend=False)
            axes[0, j].axvline(0, color='#FF9600', linestyle='--')
            axes[0, j].set_title('All Cells')
            axes[0, j].set_ylabel('DF/F0 (%)')
            axes[0, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}'))

        # Plot for each cell type.
        for i, cell_type in enumerate(['wS2', 'wM1']):
            for j, day in enumerate(days):
                d = data[(data['cell_type'] == cell_type) & (data['day'] == day)]
                sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[i + 1, j], legend=False)
                axes[i + 1, j].axvline(0, color='#FF9600', linestyle='--')
                axes[i + 1, j].set_title(f'{cell_type} - Day {day}')
                axes[i + 1, j].set_ylabel('DF/F0 (%)')
                axes[i + 1, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}'))
        # Adjust spacing between subplots to prevent title overlap
        plt.tight_layout()
        sns.despine()
        plt.suptitle(mouse_id)
        pdf.savefig(dpi=300)
        plt.close()


# #############################################################################
# Quantify the average response amplitude on baseline trials across days.
# #############################################################################

sampling_rate = 30
win_sec = (0, 0.300)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

# Load the data.
# --------------

avg_resp = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = imaging_utils.substract_baseline(xarr, 2, baseline_win)
    
    # Keep days of interest.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Average of time points.
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1])).mean(dim='time')
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    # Convert to dataframe.
    xarr.name = 'average_response'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    avg_resp.append(xarr)
avg_resp = pd.concat(avg_resp)
# Convert to percent dF/F0.
avg_resp['average_response'] = avg_resp['average_response'] * 100


# Grand average response.
# -----------------------

variance = 'mice'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data_plot = filter_data_by_cell_count(avg_resp, min_cells)
    data_plot = data_plot.groupby(['mouse_id', 'day', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
else:
    data_plot = avg_resp.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()

# Plot average response across days for all cells.
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.pointplot(data=data_plot, x='day', y='average_response', hue='reward_group',
                palette=reward_palette, hue_order=['R-', 'R+'], ax=ax, estimator='mean', legend=False)
ax.set_title('All Cells')
ax.set_ylabel('Average response (dF/F0)')
ax.set_ylim(0, 8)
ax.set_xlabel('Days')
sns.despine()

# Save figure for all cells.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'amplitude_response_across_days_all_cells_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

# Plot average response across days for projections.
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True)

# wS2 cells.
sns.pointplot(data=data_plot[data_plot.cell_type == 'wS2'], x='day', y='average_response', hue='reward_group',
                palette=reward_palette, hue_order=['R-', 'R+'], ax=axes[0], estimator='mean', legend=False)
axes[0].set_title('wS2 Cells')
axes[0].set_ylabel('Average response (dF/F0)')
axes[0].set_ylim(0, 10)
# wM1 cells.
sns.pointplot(data=data_plot[data_plot.cell_type == 'wM1'], x='day', y='average_response', hue='reward_group',
                palette=reward_palette, hue_order=['R-', 'R+'], ax=axes[1], estimator='mean', legend=False)
axes[1].set_title('wM1 Cells')
axes[1].set_ylabel('Average response (dF/F0)')
axes[1].set_xlabel('Days')
sns.despine()

# Save figure for projections.
svg_file = f'amplitude_response_across_days_projections_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

# Stats.
# Compare R+ and R- groups each day for all cells and projection neurons.
results = []
for day in days:
    # All cells
    data_stats = data_plot.loc[data_plot['day'] == day]
    r_plus = data_stats[data_stats.reward_group == 'R+']['average_response']
    r_minus = data_stats[data_stats.reward_group == 'R-']['average_response']

    stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
    results.append({
        'day': day,
        'cell_type': 'all',
        'stat': stat,
        'p_value': p_value
    })
    # Projection neurons (wS2 and wM1)
    for cell_type in ['wS2', 'wM1']:
        data_stats_proj = data_stats[data_stats.cell_type == cell_type]
        r_plus_proj = data_stats_proj[data_stats_proj.reward_group == 'R+']['average_response']
        r_minus_proj = data_stats_proj[data_stats_proj.reward_group == 'R-']['average_response']

        stat_proj, p_value_proj = mannwhitneyu(r_plus_proj, r_minus_proj, alternative='two-sided')
        results.append({
            'day': day,
            'cell_type': cell_type,
            'stat': stat_proj,
            'p_value': p_value_proj
        })
stats = pd.DataFrame(results)
stats['p_value'] = stats['p_value'].apply(lambda x: f'{x:.3}')
print(stats)

# Save data.
data_plot.to_csv(os.path.join(output_dir, f'amplitude_response_across_days_across_{variance}.csv'), index=False)
# Save stats.
stats.to_csv(os.path.join(output_dir, f'amplitude_response_across_days_across_{variance}_stats.csv'), index=False)

# Same plot comparing projection types inside each reward group across cells.
# --------------------------------------------------------------------------

variance = "cells"  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data_plot = filter_data_by_cell_count(avg_resp, min_cells)
    data_plot = data_plot.groupby(['mouse_id', 'day', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
else:
    data_plot = avg_resp.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()
# data_plot = data_plot.loc[~data_plot.mouse_id.isin(['AR131'])]

# Plot average response across days comparing projection types within each reward group.
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True)

# R+ group.
sns.pointplot(data=data_plot[data_plot.reward_group == 'R+'], x='day', y='average_response', hue='cell_type',
              palette=s2_m1_palette, hue_order=['wM1', 'wS2'], ax=axes[0], estimator='mean', legend=False)
axes[0].set_title('R+ Group')
axes[0].set_ylabel('Average response (dF/F0)')
axes[0].set_ylim(0, 10)

# R- group.
sns.pointplot(data=data_plot[data_plot.reward_group == 'R-'], x='day', y='average_response', hue='cell_type',
              palette=s2_m1_palette, hue_order=['wM1', 'wS2'], ax=axes[1], estimator='mean', legend=False)
axes[1].set_title('R- Group')
axes[1].set_ylabel('Average response (dF/F0)')
axes[1].set_xlabel('Days')
sns.despine()

# Stats.
# Compare wS2 and wM1 within each reward group and day.
results = []
for reward_group in ['R+', 'R-']:
    for day in days:
        data_stats = data_plot.loc[(data_plot['day'] == day) & (data_plot['reward_group'] == reward_group)]
        wS2 = data_stats[data_stats.cell_type == 'wS2']['average_response']
        wM1 = data_stats[data_stats.cell_type == 'wM1']['average_response']

        stat, p_value = mannwhitneyu(wS2, wM1, alternative='two-sided')
        results.append({
            'reward_group': reward_group,
            'day': day,
            'stat': stat,
            'p_value': p_value
        })
stats_projection = pd.DataFrame(results)
stats_projection['p_value'] = stats_projection['p_value'].apply(lambda x: f'{x:.3}')
print(stats_projection)

# Save figure.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'average_response_projection_comparison_across_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save stats.
stats_projection.to_csv(os.path.join(output_dir, f'average_response_projection_comparison_across_{variance}_stats.csv'), index=False)
# Save data.
data_plot.to_csv(os.path.join(output_dir, f'average_response_projection_comparison_across_{variance}.csv'), index=False)


# Quantifying response before and after learning inside reward groups.
# ####################################################################

sampling_rate = 30
win_sec_amp = (0, 0.300)  
win_sec_psth = (-0.5, 1.5)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

# Load the data.
# --------------

avg_resp = []
psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = imaging_utils.substract_baseline(xarr, 2, baseline_win)
    
    # Average response data.
    # Keep days of interest.
    avg = xarr.sel(trial=xarr['day'].isin(days))
    # Average of time points.
    avg = avg.sel(time=slice(win_sec_amp[0], win_sec_amp[1])).mean(dim='time')
    # # Average trials per days.
    # avg = avg.groupby('day').mean(dim='trial')
    # Convert to dataframe.
    avg.name = 'average_response'
    avg = avg.to_dataframe().reset_index()
    avg['mouse_id'] = mouse_id
    avg['reward_group'] = reward_group
    avg_resp.append(avg)
    
    # PSTH data.
    # Keep days of interest.
    p = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    p = p.sel(time=slice(win_sec_psth[0], win_sec_psth[1]))
    # Average trials per days.
    p = p.groupby('day').mean(dim='trial')
    
    p.name = 'psth'
    p = p.to_dataframe().reset_index()
    p['mouse_id'] = mouse_id
    p['reward_group'] = reward_group
    psth.append(p)
avg_resp = pd.concat(avg_resp)
psth = pd.concat(psth)
# Convert to percent dF/F0.
avg_resp['average_response'] = avg_resp['average_response'] * 100
psth['psth'] = psth['psth'] * 100

# Add a new column 'learning_period' to group days into 'pre' and 'post'
avg_resp['learning_period'] = avg_resp['day'].apply(lambda x: 'pre' if x in [-2,-1] else 'post')
psth['learning_period'] = psth['day'].apply(lambda x: 'pre' if x in [-2,-1] else 'post')


# Pre and post learning responses.
# --------------------------------

variance = 'mice'  # 'mice' or 'cells'
days_selected = [-2,-1, 1,2]
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)

# Select days of interest.
data_plot_avg = avg_resp[avg_resp['day'].isin(days_selected)]
data_plot_psth = psth[psth['day'].isin(days_selected)]

if variance == "mice":
    # Just filter by cell count for the projection types.
    min_cells = 3
    data_plot_avg = filter_data_by_cell_count(data_plot_avg, min_cells)
    data_plot_psth = filter_data_by_cell_count(data_plot_psth, min_cells)
    # Average for all cells and projection types independently.
    data_plot_avg_all = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group'])['average_response'].agg('mean').reset_index()
    data_plot_avg_proj = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
    data_plot_psth_all = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time'])['psth'].agg('mean').reset_index()
    data_plot_psth_proj = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type'])['psth'].agg('mean').reset_index()
else:
    data_plot_avg_all = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()
    data_plot_psth_all = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
    data_plot_avg_proj = data_plot_avg_all
    data_plot_psth_proj = data_plot_psth_all
 
# Create the figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=False, sharey=True)

# Top-left: PSTH for rewarded mice
rewarded_data = data_plot_psth_all[(data_plot_psth_all['reward_group'] == 'R+')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 0], legend=False )
axes[0, 0].set_title('PSTH (Rewarded Mice)')
axes[0, 0].set_ylabel('DF/F0 (%)')
axes[0, 0].axvline(0, color='orange', linestyle='--')

# Bottom-left: PSTH for non-rewarded mice
nonrewarded_data = data_plot_psth_all[(data_plot_psth_all['reward_group'] == 'R-')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 0], legend=False )
axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 0].set_ylabel('DF/F0 (%)')
axes[1, 0].axvline(0, color='orange', linestyle='--')

# Top-right: Response amplitude for rewarded mice
rewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R+']
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 1])
axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 1].set_ylabel('Average Response (dF/F0)')

# Bottom-right: Response amplitude for non-rewarded mice
nonrewarded_avg = data_plot_avg_all[data_plot_avg_all['reward_group'] == 'R-']
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 1])
axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 1].set_ylabel('Average Response (dF/F0)')

sns.despine()

# Save figure.
svg_file = f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_allcells.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=False, sharey=True)

# S2 PSTH
rewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R+') & (data_plot_psth_proj['cell_type'] == 'wS2')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 0], legend=False)
axes[0, 0].set_title('PSTH (Rewarded Mice)')
axes[0, 0].set_ylabel('DF/F0 (%)')
axes[0, 0].axvline(0, color='orange', linestyle='--')

nonrewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R-') & (data_plot_psth_proj['cell_type'] == 'wS2')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 0], legend=False)
axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 0].set_ylabel('DF/F0 (%)')
axes[1, 0].axvline(0, color='orange', linestyle='--')

# S2 Response amplitude
rewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R+') & (data_plot_avg_proj['cell_type'] == 'wS2')]
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 1])
axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 1].set_ylabel('Average Response (dF/F0)')

nonrewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R-') & (data_plot_avg_proj['cell_type'] == 'wS2')]
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 1])
axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 1].set_ylabel('Average Response (dF/F0)')

# M1 PSTH
rewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R+') & (data_plot_psth_proj['cell_type'] == 'wM1')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 2], legend=False)
axes[0, 2].set_title('PSTH (Rewarded Mice)')
axes[0, 2].set_ylabel('DF/F0 (%)')
axes[0, 2].axvline(0, color='orange', linestyle='--')

nonrewarded_data = data_plot_psth_proj[(data_plot_psth_proj['reward_group'] == 'R-') & (data_plot_psth_proj['cell_type'] == 'wM1')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period',hue_order=['pre', 'post'],
    palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 2], legend=False)
axes[1, 2].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 2].set_ylabel('DF/F0 (%)')
axes[1, 2].axvline(0, color='orange', linestyle='--')

# M1 Response amplitude
rewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R+') & (data_plot_avg_proj['cell_type'] == 'wM1')]
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#1b9e77', ax=axes[0, 3])
axes[0, 3].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 3].set_ylabel('Average Response (dF/F0)')

nonrewarded_avg = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == 'R-') & (data_plot_avg_proj['cell_type'] == 'wM1')]
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', order=['pre','post'], color='#c959affe', ax=axes[1, 3])
axes[1, 3].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 3].set_ylabel('Average Response (dF/F0)')

sns.despine()

# Save figure.
svg_file = f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_proj.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Perform stats on response amplitude
results = []
for reward_group in ['R+', 'R-']:
    for cell_type in ['all', 'wS2', 'wM1']:
        if cell_type == 'all':
            data_stats = data_plot_avg_all[data_plot_avg_all['reward_group'] == reward_group]
        else:
            data_stats = data_plot_avg_proj[(data_plot_avg_proj['reward_group'] == reward_group) & (data_plot_avg_proj['cell_type'] == cell_type)]
        pre = data_stats[data_stats['learning_period'] == 'pre']['average_response']
        post = data_stats[data_stats['learning_period'] == 'post']['average_response']
        stat, p_value = wilcoxon(pre, post)
        results.append({
            'reward_group': reward_group,
            'cell_type': cell_type,
            'stat': stat,
            'p_value': p_value
        })
stats_df = pd.DataFrame(results)
print(stats_df)

# Save the figure and stats

stats_df.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_stats.csv'), index=False)
data_plot_avg_all.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_data_allcells.csv'), index=False)
data_plot_avg_proj.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_data_proj.csv'), index=False)








# #############################################################################
# Correlation matrices during mapping across days.
# #############################################################################

# Parameters.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
average_inside_days = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)
# excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
# mice = [m for m in mice if m not in excluded_mice]


mice = ['AR179']
# Load data.
# ----------

vectors_rew = []
vectors_nonrew = []
for mouse in mice:
    print(mouse)
    processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, processed_dir, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    
    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))

    # Check that each day has at least n_map_trials mapping trial
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0,:,0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue
    
    # Select first n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(0, n_map_trials)))
    print(d.shape)
    
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    if rew_gp == 'R-':
        vectors_nonrew.append(d)
    elif rew_gp == 'R+':
        vectors_rew.append(d)
vectors_rew = xr.concat(vectors_rew, dim='cell')
vectors_nonrew = xr.concat(vectors_nonrew, dim='cell')


# Compute correlation matrices.
# ------------------------------

if average_inside_days:
    data_rew = vectors_rew.groupby('day').mean(dim='trial')
    data_nonrew = vectors_nonrew.groupby('day').mean(dim='trial')
else:
    data_rew = vectors_rew
    data_nonrew = vectors_nonrew

cm = np.corrcoef(data_rew.values.T)
cm_nodiag = cm.copy()
np.fill_diagonal(cm_nodiag, np.nan)
vmax = np.nanpercentile(cm_nodiag, 98)
vmin = np.nanpercentile(cm_nodiag, 6)

plt.imshow(cm, cmap='viridis', vmax=vmax, vmin=vmin)
edges = np.cumsum([n_map_trials for _ in range(5)])
for i in edges[:-1] - 0.5:
    plt.axvline(x=i, color='black', linestyle='-', lw=1.5)
    plt.axhline(y=i, color='black', linestyle='-', lw=1.5)
plt.xticks(edges - 0.5, edges)
plt.yticks(edges - 0.5, edges)
cbar = plt.colorbar()
cbar.set_label('Correlation')
plt.tight_layout()

# Save the correlation matrix as an SVG file
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/examples'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'AR179_example.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


cm = np.corrcoef(data_nonrew.values.T)
cm_nodiag = cm.copy()
np.fill_diagonal(cm_nodiag, np.nan)
vmax = np.nanpercentile(cm_nodiag, 98)
vmin = np.nanpercentile(cm_nodiag, 6)

plt.imshow(cm, cmap='viridis', vmax=vmax, vmin=vmin)
edges = np.cumsum([n_map_trials for _ in range(5)])
for i in edges[:-1] - 0.5:
    plt.axvline(x=i, color='black', linestyle='-', lw=1.5)
    plt.axhline(y=i, color='black', linestyle='-', lw=1.5)
plt.xticks(edges - 0.5, edges)
plt.yticks(edges - 0.5, edges)
cbar = plt.colorbar()
cbar.set_label('Correlation')
plt.tight_layout()

# Save the correlation matrix as an SVG file
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/examples'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'AR179_example.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# #############################################################################
# Population vectors for individual mice.
# #############################################################################

# Parameters.
sampling_rate = 30
win = (0, 0.180)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
average_inside_days = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)



output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = 'pop_vectors_individual_mice_180ms.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse in mice:
        print(mouse)

        # Vectors during learning.
        processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        win = (0, 0.300)
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse)

        file_name = 'tensor_xarray_mapping_data.nc'
        xarr_learning = imaging_utils.load_mouse_xarray(mouse, processed_dir, file_name)
        xarr_learning = xarr_learning - np.nanmean(xarr_learning.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)

        # Select days.
        xarr_learning = xarr_learning.sel(trial=xarr_learning['day'].isin([-2,-1,0,1,2]))
        # xarr_learning = xarr_learning.sel(trial=xarr_learning['whisker_stim']==1)
        xarr_learning = xarr_learning.sel(time=slice(win[0], win[1])).mean(dim='time')

        vectors_mapping = xarr_learning.values
        # Count number of whisker trials for each of the 5 days
        whisker_trial_counts = []
        for day in [-2, -1, 0, 1, 2]:
            count = int(np.sum(xarr_learning['day'].values == day))
            whisker_trial_counts.append(count)

        edges = np.cumsum(whisker_trial_counts)


        for i in edges[:-1] - 0.5:
            plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
        plt.xticks(edges - 0.5, edges)

        # Vectors during learning.
        processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        win = (0, 0.300)
        file_name = 'tensor_xarray_learning_data.nc'
        xarr_mapping = imaging_utils.load_mouse_xarray(mouse, processed_dir, file_name)
        xarr_mapping = xarr_mapping - np.nanmean(xarr_mapping.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)

        # Select whisker stim.
        xarr_mapping = xarr_mapping.sel(trial=xarr_mapping['whisker_stim']==1)
        # Select days.
        xarr_mapping = xarr_mapping.sel(trial=xarr_mapping['day'].isin([0]))
        # xarr_mapping = xarr_mapping.sel(trial=xarr_mapping['whisker_stim']==1)
        xarr_mapping = xarr_mapping.sel(time=slice(win[0], win[1])).mean(dim='time')
        vectors_learning = xarr_mapping.values
        
        
        # Concatenate mapping and learning vectors, placing learning vectors after day -1 and before day 0
        # mapping order: [-2, -1, 0, 1, 2]
        # learning order: [0]
        # So, order: mapping(-2), mapping(-1), learning(0), mapping(0), mapping(1), mapping(2)
        vectors_concat = np.concatenate([
            vectors_mapping[:, xarr_learning['day'].values == -2],
            vectors_mapping[:, xarr_learning['day'].values == -1],
            vectors_learning,
            vectors_mapping[:, xarr_learning['day'].values == 0],
            vectors_mapping[:, xarr_learning['day'].values == 1],
            vectors_mapping[:, xarr_learning['day'].values == 2]
        ], axis=1)
        
        vmax = np.percentile(vectors_mapping, 98)
        vmin = np.percentile(vectors_mapping, 2)
        
        f = plt.figure(figsize=(10, 6))
        im = plt.imshow(vectors_concat, cmap='viridis', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im)
        cbar.set_label('Activity')

        whisker_trial_counts = whisker_trial_counts[:2] + [vectors_learning.shape[1]] + whisker_trial_counts[2:]
        edges = np.cumsum(whisker_trial_counts)
        for i in edges[:-1] - 0.5:
            plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
        plt.xticks(edges - 0.5, edges)
        plt.xlabel('Trials')
        plt.ylabel('Cells')
        plt.title(f'{mouse} - {reward_group}')
        plt.tight_layout()
        # Save the figure to the PDF
        pdf.savefig(dpi=300)
        print(f"Saved figure for mouse {mouse} to PDF.")


# #########################################
# Correlation matrices average across mice.
# #########################################

# Similar to the previous section, but compute a correlation matrix for
# each mouse and then average across mice.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
zscore = False
projection_type = 'wS2'  # 'wS2', 'wM1' or None
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data.
vectors_rew = []
vectors_nonrew = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]


for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))    
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    # Z-score
    if zscore:
        d = (d - d.mean(dim='trial')) / d.std(dim='trial')

    if rew_gp == 'R-':
        vectors_nonrew.append(d)
    elif rew_gp == 'R+':
        vectors_rew.append(d)

# # Compute correlation matrices for each mouse and average across mice.
# def compute_average_correlation(vectors):
#     correlation_matrices = []
#     for vector in vectors:
#         cm = np.corrcoef(vector.values.T)
#         np.fill_diagonal(cm, np.nan)  # Exclude diagonal
#         correlation_matrices.append(cm)
#     return np.nanmean(correlation_matrices, axis=0)

# avg_corr_rew = compute_average_correlation(vectors_rew)
# avg_corr_nonrew = compute_average_correlation(vectors_nonrew)
# # Plot average correlation matrix for rewarded group using sns heatmap.
# plt.figure(figsize=(6, 5))
# sns.heatmap(avg_corr_rew, annot=False, cmap='viridis', cbar_kws={'label': 'Correlation'})
# edges = np.cumsum([n_map_trials for _ in range(len(days))])
# for edge in edges[:-1]:
#     plt.axvline(x=edge - 0.5, color='white', linestyle='-', linewidth=0.8)
#     plt.axhline(y=edge - 0.5, color='white', linestyle='-', linewidth=0.8)
# plt.xticks(edges - n_map_trials / 2, days)
# plt.yticks(edges - n_map_trials / 2, days)
# plt.title('Average Correlation Matrix (Rewarded Group)')
# plt.xlabel('Day')
# plt.ylabel('Day')
# plt.show()

# # Plot average correlation matrix for non-rewarded group using sns heatmap.
# plt.figure(figsize=(6, 5))
# sns.heatmap(avg_corr_nonrew, annot=False, cmap='viridis', cbar_kws={'label': 'Correlation'})
# for edge in edges[:-1]:
#     plt.axvline(x=edge - 0.5, color='white', linestyle='-', linewidth=0.8)
#     plt.axhline(y=edge - 0.5, color='white', linestyle='-', linewidth=0.8)
# plt.xticks(edges - n_map_trials / 2, days)
# plt.yticks(edges - n_map_trials / 2, days)
# plt.title('Average Correlation Matrix (Non-Rewarded Group)')
# plt.xlabel('Day')
# plt.ylabel('Day')
# plt.show()

# Compute average correlation for each pair of days for each mouse and return a 5x5 matrix averaged across mice.
def compute_daywise_average_correlation(vectors, days):
    daywise_correlation_matrices = []
    correlation_change_indices = []
    normalized_indices = []
    for vector in vectors:
        day_corr_matrix = np.zeros((len(days), len(days)))
        for i, day1 in enumerate(days):
            for j, day2 in enumerate(days):
                # Select data for the two days.
                day1_data = vector.sel(trial=vector['day'] == day1)
                day2_data = vector.sel(trial=vector['day'] == day2)

                # Compute correlation between days.
                corr = np.corrcoef(day1_data.values.T, day2_data.values.T)

                # If day1 is the same as day2, exclude the diagonal.
                if day1 == day2:
                    np.fill_diagonal(corr, np.nan)

                # Compute average correlation between the two days.
                avg_corr = np.nanmean(corr)
                day_corr_matrix[i, j] = avg_corr

        daywise_correlation_matrices.append(day_corr_matrix)

        # Compute correlation change index and normalized index
        days_pre = [0, 1]  # Indices for pretraining days
        days_post = [2, 3, 4]  # Indices for posttraining days

        corr_within_pre = day_corr_matrix[np.ix_(days_pre, days_pre)]
        avg_corr_within_pre = np.nanmean(corr_within_pre)

        corr_within_post = day_corr_matrix[np.ix_(days_post, days_post)]
        avg_corr_within_post = np.nanmean(corr_within_post)

        corr_between = day_corr_matrix[np.ix_(days_pre, days_post)]
        avg_corr_between = np.nanmean(corr_between)

        index = (avg_corr_within_pre + avg_corr_within_post) / 2 - avg_corr_between
        normalized_index = (avg_corr_within_pre + avg_corr_within_post - 2 * avg_corr_between) / (
            avg_corr_within_pre + avg_corr_within_post + 2 * avg_corr_between
        )

        correlation_change_indices.append(index)
        normalized_indices.append(normalized_index)

    # Average across mice.
    avg_correlation_matrix = np.nanmean(daywise_correlation_matrices, axis=0)

    return avg_correlation_matrix, correlation_change_indices, normalized_indices

# Compute daywise average correlation for rewarded and non-rewarded groups
daywise_avg_corr_rew, cci_rew, nci_rew = compute_daywise_average_correlation(vectors_rew, days)
daywise_avg_corr_nonrew, cci_nonrew, nci_nonrew = compute_daywise_average_correlation(vectors_nonrew, days)

# vmax and vmin for consistent color scaling across both matrices
# vmax = np.nanmax(daywise_avg_corr_rew)
# vmin = np.nanmin(daywise_avg_corr_nonrew)
vmax = 0.44
vmin = 0.1

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot daywise average correlation matrix for rewarded group
sns.heatmap(daywise_avg_corr_rew, annot=True, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[0])
axes[0].set_title('Daywise Average Correlation (Rewarded Group)')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Day')

# Plot daywise average correlation matrix for non-rewarded group
sns.heatmap(daywise_avg_corr_nonrew, annot=True, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[1])
axes[1].set_title('Daywise Average Correlation (Non-Rewarded Group)')
axes[1].set_xlabel('Day')
plt.tight_layout()


# Save plot.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/mapping'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'daywise_average_correlation_ctype_{projection_type}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Combine CCI (Correlation Change Index) for rewarded and non-rewarded groups into a single DataFrame
cci_df = pd.DataFrame({
    'reward_group': ['R+'] * len(cci_rew) + ['R-'] * len(cci_nonrew),
    'cci': cci_rew + cci_nonrew
})

# Plot the CCI for each mouse, grouped by reward group
plt.figure(figsize=(3, 5))
sns.stripplot(data=cci_df, x='reward_group', y='cci', hue='reward_group', palette=reward_palette[::-1],
              order=['R+', 'R-'], alpha=0.5, jitter=True)
sns.pointplot(data=cci_df, x='reward_group', y='cci', hue='reward_group', palette=reward_palette[::-1],
              order=['R+', 'R-'], estimator='mean')
plt.title('Correlation Change Index')
plt.ylabel('CCI')
plt.xlabel('Reward Group')
plt.ylim(0, 0.1)
sns.despine()

# Save plot.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/mapping'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'daywise_average_correlation_cci_ctype_{projection_type}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)

# Perform statistical comparison between the two groups (R+ and R-) for CCI

# Mann-Whitney U test for CCI
r_plus_cci = cci_df[cci_df['reward_group'] == 'R+']['cci']
r_minus_cci = cci_df[cci_df['reward_group'] == 'R-']['cci']
# Remove NaN values from the data
r_plus_cci = r_plus_cci.dropna()
r_minus_cci = r_minus_cci.dropna()
stat, p_value = mannwhitneyu(r_plus_cci, r_minus_cci, alternative='two-sided')

# Save stats to a CSV file
stats_output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/mapping'
stats_output_dir = io.adjust_path_to_host(stats_output_dir)
cci_stats_file = f'cci_stats_ctype_{projection_type}.csv'
pd.DataFrame({'stat': [stat], 'p_value': [p_value]}).to_csv(os.path.join(stats_output_dir, cci_stats_file), index=False)

# Save cci data.
cci_df.to_csv(os.path.join(stats_output_dir, 'cci_data_ctype_{projection_type}.csv'), index=False)



# Illustrate pop vectors of AR127.
# --------------------------------

# Vectors during learning.
file_name = 'tensor_xarray_mapping_data.nc'
folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
mouse = 'AR180'

xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)

# Select days.
xarray = xarray.sel(trial=xarray['day'].isin([-2, -1, 0, 1, 2]))

xarray = xarray.sel(time=slice(0, 0.300)).mean(dim='time')


# Plot
vectors_rew = xarray.values
vmax = np.percentile(vectors_rew, 98)
vmin = np.percentile(vectors_rew, 3)
edges = np.cumsum([50 for _ in range(5)])
f = plt.figure(figsize=(10, 6))
im = plt.imshow(vectors_rew, cmap='viridis', vmin=vmin, vmax=vmax)

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Activity')

for i in edges[:-1] - 0.5:
    plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
plt.xticks(edges - 0.5, edges)


file_name = 'tensor_xarray_learning_data.nc'
folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
mouse = 'AR180'
xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
# Select days
xarray = xarray.sel(trial=xarray['day'].isin([0]))
xarray = xarray.sel(time=slice(0, 0.300)).mean(dim='time')
xarray = xarray.sel(trial=xarray['trial_type'] == 'whisker_trial')

# Plot
vectors_rew = xarray.values
# vmax = np.percentile(vectors_rew, 98)
# vmin = np.percentile(vectors_rew, 2)

# edges = np.cumsum([50 for _ in range(5)])

f = plt.figure(figsize=(2, 6))
im = plt.imshow(vectors_rew, cmap='viridis', vmin=vmin, vmax=vmax)
# # Add colorbar
# cbar = plt.colorbar(im)
# cbar.set_label('Activity')

# for i in edges[:-1] - 0.5:
#     plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
# plt.xticks(edges - 0.5, edges)

# Save the figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/examples'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'AR180_pop_vectors_learning_day0.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)






# ###################################################
# Projection of whisker trials on learning dimension.
# ###################################################

# Compute a learning dimension as the difference vector between pre and post training vectors.
# Project whisker trials during learning on the learning dimension. 
sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = True
zscore = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data for mapping trials and whisker trials
results = []

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Select cells.
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue
    
    # Select pre, post, and day 0 mapping trials
    pre = xarray.sel(trial=xarray['day'].isin([-1,]))
    pre = pre.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    post = xarray.sel(trial=xarray['day'].isin([2]))
    post = post.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Select whisker trials for Day 0
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    xarray = xarray.sel(time=slice(win[0], 0.180)).mean(dim='time')
    
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))

    day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
    day0 = day0[:, :145]  # Select all trials

    learning_dim = post.mean(dim='trial') - pre.mean(dim='trial')

    # Project mapping trials onto the learning dimension
    pre_mapping_proj = np.dot(pre.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)
    post_mapping_proj = np.dot(post.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)
    # Project whisker trials onto the learning dimension
    day0_mapping_proj = np.dot(day0.values.T, learning_dim.values) / np.linalg.norm(learning_dim.values)
    
    pre_index = np.arange(pre_mapping_proj.shape[0])
    post_index = np.arange(post_mapping_proj.shape[0])
    day0_index = np.arange(day0_mapping_proj.shape[0])
    
    # Store the results in a common dataframe
    for proj, period, index in zip(
        [pre_mapping_proj, post_mapping_proj, day0_mapping_proj],
        ['pre', 'post', 'day0'],
        [pre_index, post_index, day0_index]
    ):
        results.append(pd.DataFrame({
            'mouse_id': mouse,
            'reward_group': rew_gp,
            'period': period,
            'projection': proj,
            'trial_index': index,
        }))
        
# Combine all results into a single dataframe
results_df = pd.concat(results, ignore_index=True)

# Add a column for block index (group trials into blocks of 10)
results_df['block_index'] = results_df['trial_index'] // 10

# Compute the mean projection for each block
block_results = results_df.groupby(['mouse_id', 'reward_group', 'period', 'block_index'])['projection'].mean().reset_index()

# Plot projections across mice, averaged in blocks of 10
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)

# Pre mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'pre'], x='block_index', y='projection', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0])
axes[0].set_title('Pre Mapping Trials')
axes[0].set_xlabel('Block Index (10 Trials per Block)')
axes[0].set_ylabel('Projection on Learning Dimension')

# Day 0 mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'day0'], x='block_index', y='projection', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1])
axes[1].set_title('Day 0 Mapping Trials')
axes[1].set_xlabel('Block Index (10 Trials per Block)')

# Post mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'post'], x='block_index', y='projection', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2])
axes[2].set_title('Post Mapping Trials')
axes[2].set_xlabel('Block Index (10 Trials per Block)')

sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/learning_dim'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'learning_dim_projection.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df.to_csv(os.path.join(output_dir, 'learning_dim_projection.csv'), index=False)




# ###################################################
# Correlation of whisker trials with post-learning vector.
# ###################################################

# Compute a post-learning vector as the average response of post-training trials.
# Compute the correlation of whisker trials during learning with the post-learning vector.
sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = True
zscore = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data for mapping trials and whisker trials
results = []

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Select cells.
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue
    
    # Select pre and post mapping trials
    pre = xarray.sel(trial=xarray['day'].isin([-1]))
    pre = pre.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    post = xarray.sel(trial=xarray['day'].isin([2]))
    post = post.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))

    # Compute the post-learning vector
    post_learning_vector = post.mean(dim='trial')

    # Compute correlation of pre and post trials with the post-learning vector
    pre_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in pre.values.T])
    post_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in post.values.T])

    # Select whisker trials for Day 0
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    xarray = xarray.sel(time=slice(win[0], 0.180)).mean(dim='time')
    
    if select_responsive_cells or select_lmi:
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))

    # day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
    # day0 = day0[:, :145]  # Select all trials
    if rew_gp == 'R-':
        day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial'))
        day0 = day0[:, :145]  # Select all trials
    elif rew_gp == 'R+':
        day0 = xarray.sel(trial=(xarray['day'] == 0) & (xarray['trial_type'] == 'whisker_trial') & (xarray['outcome_w'] == 1))
        day0 = day0[:, :145]

    # Compute correlation of whisker trials with the post-learning vector
    day0_corr = np.array([np.corrcoef(trial, post_learning_vector.values)[0, 1] for trial in day0.values.T])
    
    pre_index = np.arange(pre_corr.shape[0])
    post_index = np.arange(post_corr.shape[0])
    day0_index = np.arange(day0_corr.shape[0])
    
    # Store the results in a common dataframe
    for corr, period, index in zip(
        [pre_corr, post_corr, day0_corr],
        ['pre', 'post', 'day0'],
        [pre_index, post_index, day0_index]
    ):
        results.append(pd.DataFrame({
            'mouse_id': mouse,
            'reward_group': rew_gp,
            'period': period,
            'correlation': corr,
            'trial_index': index,
        }))
        
# Combine all results into a single dataframe
results_df = pd.concat(results, ignore_index=True)

# Add a column for block index (group trials into blocks of 10)
results_df['block_index'] = results_df['trial_index'] // 7

# Compute the mean correlation for each block
block_results = results_df.groupby(['mouse_id', 'reward_group', 'period', 'block_index'])['correlation'].mean().reset_index()

# Plot correlations across mice, averaged in blocks of 10
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)

# Pre mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'pre'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0])
axes[0].set_title('Pre Mapping Trials')
axes[0].set_xlabel('Block Index (10 Trials per Block)')
axes[0].set_ylabel('Correlation with Post-Learning Vector')

# Day 0 whisker trials
sns.pointplot(data=block_results[block_results['period'] == 'day0'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1])
axes[1].set_title('Day 0 Whisker Trials')
axes[1].set_xlabel('Block Index (10 Trials per Block)')

# Post mapping trials
sns.pointplot(data=block_results[block_results['period'] == 'post'], x='block_index', y='correlation', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2])
axes[2].set_title('Post Mapping Trials')
axes[2].set_xlabel('Block Index (10 Trials per Block)')

sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/learning_dim'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'correlation_with_post_learning_vector_whhitR+.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df.to_csv(os.path.join(output_dir, 'correlation_with_post_learning_vector_whhitR+.csv'), index=False)



# ###########################################################
# Correlation matrices including mapping and learning trials.
# ###########################################################

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
n_learning_trials = 50
substract_baseline = True
select_responsive_cells = False
select_lmi = True
zscore = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)
mice = ['AR127']
# Load data.
vectors_map_rew = []
vectors_map_nonrew = []
vectors_learning_rew = []
vectors_learning_nonrew = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]


for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray_map = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray_map = xarray_map - np.nanmean(xarray_map.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    file_name = 'tensor_xarray_learning_data.nc'
    xarray_learning = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray_learning = xarray_learning - np.nanmean(xarray_learning.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)

    # Select days.
    xarray_map = xarray_map.sel(trial=xarray_map['day'].isin(days))
    xarray_learning = xarray_learning.sel(trial=xarray_learning['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        xarray_map = xarray_map.sel(cell=xarray_map['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))
        xarray_learning = xarray_learning.sel(cell=xarray_learning['roi'].isin(selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']))

    # Select last n_map_trials mapping trials for each day.
    d_map = xarray_map.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))    
    d_map = d_map.sel(time=slice(win[0], win[1])).mean(dim='time')
    # Z-score
    if zscore:
        d_map = (d_map - d_map.mean(dim='trial')) / d_map.std(dim='trial')
    if rew_gp == 'R-':
        vectors_map_nonrew.append(d_map)
    elif rew_gp == 'R+':
        vectors_map_rew.append(d_map)
    
    # Select learning trials for each day.
    d_learn = xarray_learning.sel(trial=xarray_learning['trial_type'] == 'whisker_trial')
    d_learn = d_learn.groupby('day').apply(lambda x: x.isel(trial=slice(0, n_learning_trials)))    
    d_learn = d_learn.sel(time=slice(win[0], win[1])).mean(dim='time')
    # Z-score
    if zscore:
        d_learn = (d_learn - d_learn.mean(dim='trial')) / d_learn.std(dim='trial')
    if rew_gp == 'R-':
        vectors_learning_nonrew.append(d_learn)
    elif rew_gp == 'R+':
        vectors_learning_rew.append(d_learn)

# Compute average correlation for each pair of days for each mouse and return an 8x8 matrix averaged across mice.
def compute_daywise_average_correlation(map_vectors, learning_vectors):
    daywise_correlation_matrices = []
    for map_vector, learning_vector in zip(map_vectors, learning_vectors):
        # Initialize an 8x8 matrix for mapping and learning days
        day_corr_matrix = np.zeros((8, 8))
        
        # Combine mapping and learning vectors into a single list
        combined_vectors = [
            map_vector.sel(trial=map_vector['day'] == -2),
            map_vector.sel(trial=map_vector['day'] == -1),
            learning_vector.sel(trial=learning_vector['day'] == 0),
            map_vector.sel(trial=map_vector['day'] == 0),
            learning_vector.sel(trial=learning_vector['day'] == 1),
            map_vector.sel(trial=map_vector['day'] == 1),
            learning_vector.sel(trial=learning_vector['day'] == 2),
            map_vector.sel(trial=map_vector['day'] == 2),
        ]
        
        # Compute correlations between all pairs of days
        for i, vect_1 in enumerate(combined_vectors):
            for j, vect_2 in enumerate(combined_vectors):
                # Compute correlation between the two sets of trials
                corr = np.corrcoef(vect_1.values.T, vect_2.values.T)
                
                # If comparing the same day, exclude the diagonal
                if i == j:
                    np.fill_diagonal(corr, np.nan)
                
                # Compute average correlation between the two days
                avg_corr = np.nanmean(corr)
                day_corr_matrix[i, j] = avg_corr
        
        daywise_correlation_matrices.append(day_corr_matrix)
    
    # Average across mice
    avg_correlation_matrix = np.nanmean(daywise_correlation_matrices, axis=0)
    
    return avg_correlation_matrix

# Compute average correlation for each pair of days for each mouse and return an 8x8 matrix averaged across mice.
def compute_trialwise_average_correlation(map_vectors, learning_vectors):
    trialwise_correlation_matrices = []
    for map_vector, learning_vector in zip(map_vectors, learning_vectors):
        
        # Check that each day has at least n_map_trials mapping trials
        n_trials_map = map_vector[0, :].groupby('day').count(dim='trial').values
        if np.any(n_trials_map < n_map_trials):
            print(f'Not enough mapping trials for this mouse.')
            continue

        # Check that each day has at least n_learning_trials learning trials
        n_trials_learning = learning_vector[0, :].groupby('day').count(dim='trial').values
        if np.any(n_trials_learning < n_learning_trials):
            print(f'Not enough learning trials for this mouse.')
            continue
        
        # Combine mapping and learning vectors into a single list
        combined_vectors = [
            map_vector.sel(trial=map_vector['day'] == -2),
            map_vector.sel(trial=map_vector['day'] == -1),
            learning_vector.sel(trial=learning_vector['day'] == 0),
            map_vector.sel(trial=map_vector['day'] == 0),
            learning_vector.sel(trial=learning_vector['day'] == 1),
            map_vector.sel(trial=map_vector['day'] == 1),
            learning_vector.sel(trial=learning_vector['day'] == 2),
            map_vector.sel(trial=map_vector['day'] == 2),
        ]
        for i, vect in enumerate(combined_vectors):
            print(vect.shape)
        # Compute correlations between all pairs of trials
        
        combined_vectors = np.concatenate([vect.values.T for vect in combined_vectors], axis=0)
        # Compute correlation between the two sets of trials
        corr = np.corrcoef(combined_vectors)
        trialwise_correlation_matrices.append(corr)
        
    
    # Average across mice
    avg_correlation_matrix = np.nanmean(trialwise_correlation_matrices, axis=0)
    
    return avg_correlation_matrix



# Compute daywise average correlation for rewarded and non-rewarded groups
daywise_avg_corr_rew = compute_daywise_average_correlation(vectors_map_rew, vectors_learning_rew)
daywise_avg_corr_nonrew = compute_daywise_average_correlation(vectors_map_nonrew, vectors_learning_nonrew)

# Compute daywise average correlation for rewarded and non-rewarded groups
trialwise_avg_corr_rew = compute_trialwise_average_correlation(vectors_map_rew, vectors_learning_rew)
trialwise_avg_corr_nonrew = compute_trialwise_average_correlation(vectors_map_nonrew, vectors_learning_nonrew)

# vmax and vmin for consistent color scaling across both matrices
vmax = np.nanpercentile(daywise_avg_corr_rew, 98)
vmin = np.nanpercentile(daywise_avg_corr_nonrew, 3)
vmax = 0.8
vmin = 0.15

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot daywise average correlation matrix for rewarded group
sns.heatmap(daywise_avg_corr_rew, annot=False, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[0])
axes[0].set_title('Daywise Average Correlation (Rewarded Group)')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Day')

# Plot daywise average correlation matrix for non-rewarded group
sns.heatmap(daywise_avg_corr_nonrew, annot=False, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[1])
axes[1].set_title('Daywise Average Correlation (Non-Rewarded Group)')
axes[1].set_xlabel('Day')

# Adjust layout
plt.tight_layout()
# plt.show()

# Save plot.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/during_learning'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'average_correlation_mapping_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# vmax and vmin for consistent color scaling across both matrices
vmax = np.nanpercentile(trialwise_avg_corr_rew, 99.5)
vmin = np.nanpercentile(trialwise_avg_corr_nonrew, 4)
vmax = 1
vmin = 0.15

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot trialwise average correlation matrix for rewarded group
sns.heatmap(trialwise_avg_corr_rew, annot=False, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[0])
axes[0].set_title('trialwise Average Correlation (Rewarded Group)')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Day')

# Plot trialwise average correlation matrix for non-rewarded group
sns.heatmap(trialwise_avg_corr_nonrew, annot=False, fmt=".2f", cmap='viridis', xticklabels=days, yticklabels=days, 
            cbar_kws={'label': 'Correlation'}, vmax=vmax, vmin=vmin, ax=axes[1])
axes[1].set_title('trialwise Average Correlation (Non-Rewarded Group)')
axes[1].set_xlabel('Day')

# Adjust layout
plt.tight_layout()
# plt.show()

# Save plot.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/during_learning'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'average_correlation_mapping_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



