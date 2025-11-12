import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging 
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *



# #############################################################################
# 1. PSTH's of auditory responses during learning.
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
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Define LMI cells (lmi_p <= 0.025 or >= 0.975)
lmi_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]
# Define Non-LMI cells (lmi_p > 0.025 and < 0.975)
non_lmi_cells = lmi_df.loc[(lmi_df['lmi_p'] > 0.025) & (lmi_df['lmi_p'] < 0.975)]
# Load and preprocess data for all mice
lmi_psth = []
non_lmi_psth = []
lmi_ws2_psth = []
non_lmi_ws2_psth = []
lmi_wm1_psth = []
non_lmi_wm1_psth = []

# Remove mouse GF305 for now (due to data issues)
mice = [m for m in mice if m != 'GF305']

for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)
    xarr = xarr.sel(trial=xarr['auditory_stim'] == 1)
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    # Trial averaging for each cell.
    xarr = xarr.groupby('day').mean(dim='trial')
    xarr.name = 'psth'

    # LMI cells for this mouse
    lmi_cells_for_mouse = lmi_cells.loc[lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_lmi = xarr.sel(cell=xarr['roi'].isin(lmi_cells_for_mouse))
    df_lmi = xarr_lmi.to_dataframe().reset_index()
    df_lmi['mouse_id'] = mouse_id
    df_lmi['reward_group'] = reward_group
    lmi_psth.append(df_lmi)

    # Non-LMI cells for this mouse
    non_lmi_cells_for_mouse = non_lmi_cells.loc[non_lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_non_lmi = xarr.sel(cell=xarr['roi'].isin(non_lmi_cells_for_mouse))
    df_non_lmi = xarr_non_lmi.to_dataframe().reset_index()
    df_non_lmi['mouse_id'] = mouse_id
    df_non_lmi['reward_group'] = reward_group
    non_lmi_psth.append(df_non_lmi)

    # LMI wS2 cells
    lmi_ws2_cells = lmi_cells.loc[(lmi_cells['mouse_id'] == mouse_id) & (lmi_cells['cell_type'] == 'wS2')]['roi']
    xarr_lmi_ws2 = xarr.sel(cell=xarr['roi'].isin(lmi_ws2_cells))
    df_lmi_ws2 = xarr_lmi_ws2.to_dataframe().reset_index()
    df_lmi_ws2['mouse_id'] = mouse_id
    df_lmi_ws2['reward_group'] = reward_group
    lmi_ws2_psth.append(df_lmi_ws2)

    # Non-LMI wS2 cells
    non_lmi_ws2_cells = non_lmi_cells.loc[(non_lmi_cells['mouse_id'] == mouse_id) & (non_lmi_cells['cell_type'] == 'wS2')]['roi']
    xarr_non_lmi_ws2 = xarr.sel(cell=xarr['roi'].isin(non_lmi_ws2_cells))
    df_non_lmi_ws2 = xarr_non_lmi_ws2.to_dataframe().reset_index()
    df_non_lmi_ws2['mouse_id'] = mouse_id
    df_non_lmi_ws2['reward_group'] = reward_group
    non_lmi_ws2_psth.append(df_non_lmi_ws2)

    # LMI wM1 cells
    lmi_wm1_cells = lmi_cells.loc[(lmi_cells['mouse_id'] == mouse_id) & (lmi_cells['cell_type'] == 'wM1')]['roi']
    xarr_lmi_wm1 = xarr.sel(cell=xarr['roi'].isin(lmi_wm1_cells))
    df_lmi_wm1 = xarr_lmi_wm1.to_dataframe().reset_index()
    df_lmi_wm1['mouse_id'] = mouse_id
    df_lmi_wm1['reward_group'] = reward_group
    lmi_wm1_psth.append(df_lmi_wm1)

    # Non-LMI wM1 cells
    non_lmi_wm1_cells = non_lmi_cells.loc[(non_lmi_cells['mouse_id'] == mouse_id) & (non_lmi_cells['cell_type'] == 'wM1')]['roi']
    xarr_non_lmi_wm1 = xarr.sel(cell=xarr['roi'].isin(non_lmi_wm1_cells))
    df_non_lmi_wm1 = xarr_non_lmi_wm1.to_dataframe().reset_index()
    df_non_lmi_wm1['mouse_id'] = mouse_id
    df_non_lmi_wm1['reward_group'] = reward_group
    non_lmi_wm1_psth.append(df_non_lmi_wm1)

lmi_psth = pd.concat(lmi_psth)
non_lmi_psth = pd.concat(non_lmi_psth)
lmi_ws2_psth = pd.concat(lmi_ws2_psth)
non_lmi_ws2_psth = pd.concat(non_lmi_ws2_psth)
lmi_wm1_psth = pd.concat(lmi_wm1_psth)
non_lmi_wm1_psth = pd.concat(non_lmi_wm1_psth)

# Convert to percent dF/F0
lmi_psth['psth'] = lmi_psth['psth'] * 100
non_lmi_psth['psth'] = non_lmi_psth['psth'] * 100
lmi_ws2_psth['psth'] = lmi_ws2_psth['psth'] * 100
non_lmi_ws2_psth['psth'] = non_lmi_ws2_psth['psth'] * 100
lmi_wm1_psth['psth'] = lmi_wm1_psth['psth'] * 100
non_lmi_wm1_psth['psth'] = non_lmi_wm1_psth['psth'] * 100

# Stats over mice.
min_cells = 3
lmi_psth_avg = utils_imaging.filter_data_by_cell_count(lmi_psth, min_cells)
lmi_psth_avg = lmi_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()
non_lmi_psth_avg = utils_imaging.filter_data_by_cell_count(non_lmi_psth, min_cells)
non_lmi_psth_avg = non_lmi_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()

lmi_ws2_psth_avg = utils_imaging.filter_data_by_cell_count(lmi_ws2_psth, min_cells)
lmi_ws2_psth_avg = lmi_ws2_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()
non_lmi_ws2_psth_avg = utils_imaging.filter_data_by_cell_count(non_lmi_ws2_psth, min_cells)
non_lmi_ws2_psth_avg = non_lmi_ws2_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()

lmi_wm1_psth_avg = utils_imaging.filter_data_by_cell_count(lmi_wm1_psth, min_cells)
lmi_wm1_psth_avg = lmi_wm1_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()
non_lmi_wm1_psth_avg = utils_imaging.filter_data_by_cell_count(non_lmi_wm1_psth, min_cells)
non_lmi_wm1_psth_avg = non_lmi_wm1_psth_avg.groupby(['mouse_id', 'day', 'reward_group', 'time',])['psth'].agg('mean').reset_index()

# Stats over cells.
lmi_psth_avg_cells = lmi_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_psth_avg_cells = non_lmi_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()
lmi_ws2_psth_avg_cells = lmi_ws2_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_ws2_psth_avg_cells = non_lmi_ws2_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()
lmi_wm1_psth_avg_cells = lmi_wm1_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_wm1_psth_avg_cells = non_lmi_wm1_psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi'])['psth'].agg('mean').reset_index()

# Helper function to plot PSTH as a single figure with 6 rows
def plot_psth_multi(
    non_lmi_psth, lmi_psth,
    non_lmi_ws2_psth, lmi_ws2_psth,
    non_lmi_wm1_psth, lmi_wm1_psth,
    days, reward_palette, stim_palette, output_dir, fig_name
):
    fig, axes = plt.subplots(6, len(days), figsize=(18, 25), sharey=False)
    row_titles = [
        'Non-LMI Cells',
        'LMI Cells',
        'Non-LMI wS2 Cells',
        'LMI wS2 Cells',
        'Non-LMI wM1 Cells',
        'LMI wM1 Cells'
    ]
    data_rows = [
        non_lmi_psth,
        lmi_psth,
        non_lmi_ws2_psth,
        lmi_ws2_psth,
        non_lmi_wm1_psth,
        lmi_wm1_psth
    ]
    ylims = [(-1, 12), (-1, 12), (-1, 16), (-1, 16), (-1, 16), (-1, 16)]
    for i, (row_title, data, ylim) in enumerate(zip(row_titles, data_rows, ylims)):
        for j, day in enumerate(days):
            d = data.loc[data['day'] == day]
            ax = axes[i, j]
            if not d.empty:
                sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                             hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'{row_title} - Day {day}')
            ax.set_ylabel('DF/F0 (%)')
            ax.set_ylim(ylim)
    plt.tight_layout()
    sns.despine()
    output_dir = io.adjust_path_to_host(output_dir)
    plt.savefig(os.path.join(output_dir, fig_name), format='svg', dpi=300)
    plt.close(fig)

# Usage
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
plot_psth_multi(
    non_lmi_psth_avg, lmi_psth_avg,
    non_lmi_ws2_psth_avg, lmi_ws2_psth_avg,
    non_lmi_wm1_psth_avg, lmi_wm1_psth_avg,
    days, reward_palette, stim_palette, output_dir, fig_name='auditory_psth_during_learning.svg'
)


# PSTH per mouse and per cell type.
# ---------------------------------
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
output_pdf = os.path.join(output_dir, 'lmi_auditory_psth_per_mouse.pdf')
with PdfPages(output_pdf) as pdf:
    for mouse_id in mice:
        fig, axes = plt.subplots(6, 5, figsize=(18, 25), sharey=False)
        fig.suptitle(f'PSTH Plots for Mouse {mouse_id}', fontsize=16)

        # Non-LMI cells 
        d = non_lmi_psth_avg_cells[non_lmi_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[0, j]
            day_data = d[d['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'Non-LMI Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')

        # LMI cells
        d = lmi_psth_avg_cells[lmi_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[1, j]
            day_data = d[d['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'LMI Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')

        # Non-LMI wS2 cells
        d_ws2 = non_lmi_ws2_psth_avg_cells[non_lmi_ws2_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[2, j]
            day_data = d_ws2[d_ws2['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'Non-LMI wS2 Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')

        # LMI wS2 cells
        d_lmi_ws2 = lmi_ws2_psth_avg_cells[lmi_ws2_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[3, j]
            day_data = d_lmi_ws2[d_lmi_ws2['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'LMI wS2 Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')

        # Non-LMI wM1 cells
        d_wm1 = non_lmi_wm1_psth_avg_cells[non_lmi_wm1_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[4, j]
            day_data = d_wm1[d_wm1['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'Non-LMI wM1 Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')

        # LMI wM1 cells
        d_lmi_wm1 = lmi_wm1_psth_avg_cells[lmi_wm1_psth_avg_cells['mouse_id'] == mouse_id]
        for j, day in enumerate(days):
            ax = axes[5, j]
            day_data = d_lmi_wm1[d_lmi_wm1['day'] == day]
            sns.lineplot(data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                         hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False, n_boot=100)
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'LMI wM1 Cells - Day {day}')
            ax.set_ylabel('DF/F0 (%)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        sns.despine()
        pdf.savefig(fig)
        plt.close(fig)


# #############################################################################
# Lick-aligned PSTH's.
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
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Define LMI cells (lmi_p <= 0.025 or >= 0.975)
lmi_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]
# Define Non-LMI cells (lmi_p > 0.025 and < 0.975)
non_lmi_cells = lmi_df.loc[(lmi_df['lmi_p'] > 0.025) & (lmi_df['lmi_p'] < 0.975)]

# Load and preprocess data for all mice
non_lmi_lick = []
lmi_lick = []
non_lmi_ws2_lick = []
lmi_ws2_lick = []
non_lmi_wm1_lick = []
lmi_wm1_lick = []

# Remove mouse GF305 for now (due to data issues)
mice = [m for m in mice if m != 'GF305']

for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    file_name = 'lick_aligned_xarray.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    # Select false alarm trials
    xarr = xarr.sel(trial=xarr['no_stim'] == 1)
    xarr = xarr.sel(trial=xarr['lick_flag'] == 1)
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    xarr = xarr.groupby('day').mean(dim='trial')
    xarr.name = 'psth'

    # Non-LMI cells for this mouse
    non_lmi_cells_for_mouse = non_lmi_cells.loc[non_lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_non_lmi = xarr.sel(cell=xarr['roi'].isin(non_lmi_cells_for_mouse))
    df_non_lmi = xarr_non_lmi.to_dataframe().reset_index()
    df_non_lmi['mouse_id'] = mouse_id
    df_non_lmi['reward_group'] = reward_group
    non_lmi_lick.append(df_non_lmi)

    # LMI cells for this mouse
    lmi_cells_for_mouse = lmi_cells.loc[lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_lmi = xarr.sel(cell=xarr['roi'].isin(lmi_cells_for_mouse))
    df_lmi = xarr_lmi.to_dataframe().reset_index()
    df_lmi['mouse_id'] = mouse_id
    df_lmi['reward_group'] = reward_group
    lmi_lick.append(df_lmi)

    # LMI wS2 cells
    lmi_ws2_cells = lmi_cells.loc[(lmi_cells['mouse_id'] == mouse_id) & (lmi_cells['cell_type'] == 'wS2')]['roi']
    xarr_lmi_ws2 = xarr.sel(cell=xarr['roi'].isin(lmi_ws2_cells))
    df_lmi_ws2 = xarr_lmi_ws2.to_dataframe().reset_index()
    df_lmi_ws2['mouse_id'] = mouse_id
    df_lmi_ws2['reward_group'] = reward_group
    lmi_ws2_lick.append(df_lmi_ws2)

    # Non-LMI wS2 cells
    non_lmi_ws2_cells = non_lmi_cells.loc[(non_lmi_cells['mouse_id'] == mouse_id) & (non_lmi_cells['cell_type'] == 'wS2')]['roi']
    xarr_non_lmi_ws2 = xarr.sel(cell=xarr['roi'].isin(non_lmi_ws2_cells))
    df_non_lmi_ws2 = xarr_non_lmi_ws2.to_dataframe().reset_index()
    df_non_lmi_ws2['mouse_id'] = mouse_id
    df_non_lmi_ws2['reward_group'] = reward_group
    non_lmi_ws2_lick.append(df_non_lmi_ws2)

    # LMI wM1 cells
    lmi_wm1_cells = lmi_cells.loc[(lmi_cells['mouse_id'] == mouse_id) & (lmi_cells['cell_type'] == 'wM1')]['roi']
    xarr_lmi_wm1 = xarr.sel(cell=xarr['roi'].isin(lmi_wm1_cells))
    df_lmi_wm1 = xarr_lmi_wm1.to_dataframe().reset_index()
    df_lmi_wm1['mouse_id'] = mouse_id
    df_lmi_wm1['reward_group'] = reward_group
    lmi_wm1_lick.append(df_lmi_wm1)

    # Non-LMI wM1 cells
    non_lmi_wm1_cells = non_lmi_cells.loc[(non_lmi_cells['mouse_id'] == mouse_id) & (non_lmi_cells['cell_type'] == 'wM1')]['roi']
    xarr_non_lmi_wm1 = xarr.sel(cell=xarr['roi'].isin(non_lmi_wm1_cells))
    df_non_lmi_wm1 = xarr_non_lmi_wm1.to_dataframe().reset_index()
    df_non_lmi_wm1['mouse_id'] = mouse_id
    df_non_lmi_wm1['reward_group'] = reward_group
    non_lmi_wm1_lick.append(df_non_lmi_wm1)

non_lmi_lick = pd.concat(non_lmi_lick)
lmi_lick = pd.concat(lmi_lick)
non_lmi_ws2_lick = pd.concat(non_lmi_ws2_lick)
lmi_ws2_lick = pd.concat(lmi_ws2_lick)
non_lmi_wm1_lick = pd.concat(non_lmi_wm1_lick)
lmi_wm1_lick = pd.concat(lmi_wm1_lick)

# Group by cell for PSTH
non_lmi_lick = non_lmi_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
lmi_lick = lmi_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_ws2_lick = non_lmi_ws2_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
lmi_ws2_lick = lmi_ws2_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_wm1_lick = non_lmi_wm1_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
lmi_wm1_lick = lmi_wm1_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()

# Convert to percent dF/F0
non_lmi_lick['psth'] = non_lmi_lick['psth'] * 100
lmi_lick['psth'] = lmi_lick['psth'] * 100
non_lmi_ws2_lick['psth'] = non_lmi_ws2_lick['psth'] * 100
lmi_ws2_lick['psth'] = lmi_ws2_lick['psth'] * 100
non_lmi_wm1_lick['psth'] = non_lmi_wm1_lick['psth'] * 100
lmi_wm1_lick['psth'] = lmi_wm1_lick['psth'] * 100

# Generate a single figure with 6 rows for lick-aligned PSTH's
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
plot_psth_multi(
    non_lmi_lick,
    lmi_lick,
    non_lmi_ws2_lick,
    lmi_ws2_lick,
    non_lmi_wm1_lick,
    lmi_wm1_lick,
    days,
    reward_palette,
    stim_palette,
    output_dir,
    fig_name='lick_aligned_psth_during_learning.svg'
)




# Checking GF305 day -1.
# ######################

# Plot average auditory response per cell for GF305 using raw data

sampling_rate = 30
win_sec = (-0.5, 4)  
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']

mouse_id = 'GF305'
file_name = 'tensor_xarray_learning_data.nc'
folder = os.path.join(io.processed_dir, 'mice')
xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

# Select auditory trials and day -1
xarr = xarr.sel(trial=xarr['auditory_stim'] == 1)
xarr = xarr.sel(trial=xarr['day'] == -1)
xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

# Average over trials for each cell
xarr_avg = xarr.mean(dim='trial')

# Plot each cell's response, shifted in y
plt.figure(figsize=(10, 60))
offset = 0
offset_step = 0.15 * np.nanmax(xarr_avg.values)
for i, roi in enumerate(xarr_avg['roi'].values):
    trace = xarr_avg.sel(cell=i).values
    plt.plot(xarr_avg['time'].values, trace + offset, label=f'Cell {roi}')
    offset += offset_step
plt.xlabel('Time (s)')
plt.ylabel('DF/F0 (shifted)')
plt.title(f'GF305 Day -1 Auditory Response (each cell shifted)')
plt.tight_layout()
sns.despine()
plt.show()



trace = xarr_avg.sel(cell=12).values
plt.plot(xarr_avg['time'].values, trace)




xarr = xarr / 100
# Save xarr
xarr.to_netcdf(os.path.join(folder, f'tensor_xarray_learning_data.nc'))
