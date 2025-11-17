import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging 
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# #############################################################################
# 1. PSTH's of LMI responses across days.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (-0.5, 6)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']
top_n = 20

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Add reward_group column to lmi_df
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)
# Select top_n positive and negative cells for each reward group independently
top_positive_lmi_cells = []
top_negative_lmi_cells = []
for group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == group]
    group_sorted = group_df.sort_values('lmi', ascending=False)
    top_positive_lmi_cells.append(group_sorted.head(top_n))
    top_negative_lmi_cells.append(group_sorted.tail(top_n))
top_positive_lmi_cells = pd.concat(top_positive_lmi_cells)
top_negative_lmi_cells = pd.concat(top_negative_lmi_cells)

# Load and preprocess data for all mice

# Refactored: Load each mouse's xarray once, select trial types/outcomes, and plot PSTHs for each requested type
mice = [m for m in mice if m != 'GF305']

# Prepare cell lists for each mouse (only positive and negative LMI cells)
cell_groups = {}
for mouse_id in mice:
    cell_groups[mouse_id] = {
        'positive': top_positive_lmi_cells.loc[top_positive_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist(),
        'negative': top_negative_lmi_cells.loc[top_negative_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist()
    }

# Trial type definitions
trial_types = {
    'auditory': lambda xarr: xarr.sel(trial=xarr['auditory_stim'] == 1),
    'whisker': lambda xarr: xarr.sel(trial=xarr['whisker_stim'] == 1),
    'whisker_hit': lambda xarr: xarr.sel(trial=(xarr['whisker_stim'] == 1) & (xarr['lick_flag'] == 1)),
    'whisker_miss': lambda xarr: xarr.sel(trial=(xarr['whisker_stim'] == 1) & (xarr['lick_flag'] == 0)),
    'no_stim': lambda xarr: xarr.sel(trial=xarr['no_stim'] == 1),
    'no_stim_hit': lambda xarr: xarr.sel(trial=(xarr['no_stim'] == 1) & (xarr['lick_flag'] == 1)),
    'no_stim_miss': lambda xarr: xarr.sel(trial=(xarr['no_stim'] == 1) & (xarr['lick_flag'] == 0)),
}

# For each mouse, load xarray once and store
xarrays = {}
for mouse_id in mice:
    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))
    xarrays[mouse_id] = xarr

# For each trial type, compute PSTH for each cell group
results = {tt: {'positive': [], 'negative': []} for tt in trial_types}
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    xarr = xarrays[mouse_id]
    for tt, tt_func in trial_types.items():
        xarr_tt = tt_func(xarr)
        # Average trials per day
        xarr_tt = xarr_tt.groupby('day').mean(dim='trial')
        xarr_tt.name = 'psth'
        for group in ['positive', 'negative']:
            cell_list = cell_groups[mouse_id][group]
            if not cell_list:
                continue
            xarr_group = xarr_tt.sel(cell=xarr_tt['roi'].isin(cell_list))
            # Convert to DataFrame for plotting
            df = xarr_group.to_dataframe().reset_index()
            df['mouse_id'] = mouse_id
            df['reward_group'] = reward_group
            df['psth'] = df['psth'] * 100
            results[tt][group].append(df)

# Concatenate results for each trial type and cell group
for tt in results:
    for group in results[tt]:
        if results[tt][group]:
            results[tt][group] = pd.concat(results[tt][group])
        else:
            results[tt][group] = pd.DataFrame()

# Plotting
plot_order = [
    ('auditory', 'Auditory Trials'),
    ('whisker', 'Whisker Trials'),
    ('whisker_hit', 'Whisker Hit'),
    ('whisker_miss', 'Whisker Miss'),
    ('no_stim', 'No Stim'),
    ('no_stim_hit', 'No Stim Hit (False Alarm)'),
    ('no_stim_miss', 'No Stim Miss (Correct Rejection)'),
]
row_titles = ['Top 20 Positive LMI Cells', 'Top 20 Negative LMI Cells']
cell_groups_plot = ['positive', 'negative']

for tt, tt_label in plot_order:
    fig, axes = plt.subplots(2, len(days), figsize=(18, 8), sharey=False)
    for i, group in enumerate(cell_groups_plot):
        data = results[tt][group]
        for j, day in enumerate(days):
            d = data.loc[data['day'] == day] if not data.empty else pd.DataFrame()
            ax = axes[i, j]
            if not d.empty:
                sns.lineplot(
                    data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                    hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False
                )
            ax.axvline(0, color=stim_palette[0], linestyle='-')
            ax.set_title(f'{row_titles[i]} - Day {day}')
            ax.set_ylabel('DF/F0 (%)')
    plt.suptitle(f'{tt_label} PSTH During Learning', fontsize=16)
    plt.tight_layout()
    sns.despine()
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
    output_dir = io.adjust_path_to_host(output_dir)
    plt.savefig(os.path.join(output_dir, f'{tt}_psth_during_learning.svg'), format='svg', dpi=300)
    # plt.close()



# ==================================================
# PSTH per mouse.
# ==================================================

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
output_dir = io.adjust_path_to_host(output_dir)
output_pdf = os.path.join(output_dir, 'lmi_auditory_psth_per_mouse.pdf')
with PdfPages(output_pdf) as pdf:
    for mouse_id in mice:
        fig, axes = plt.subplots(3, len(days), figsize=(18, 12), sharey=False)
        fig.suptitle(f'Auditory PSTH for Mouse {mouse_id}', fontsize=16)
        for i, group in enumerate(['positive', 'negative', 'non_lmi']):
            # Use previously extracted data
            if group == 'non_lmi':
                data = lmi_df.loc[(lmi_df['lmi_p'] > 0.025) & (lmi_df['lmi_p'] < 0.975)]
                cell_list = data[data['mouse_id'] == mouse_id]['roi'].tolist()
                d_mouse = results['auditory']['positive'][results['auditory']['positive']['roi'].isin(cell_list)]
                # If no data, fallback to empty
                if d_mouse.empty:
                    d_mouse = pd.DataFrame()
            else:
                data = results['auditory'][group]
                d_mouse = data[data['mouse_id'] == mouse_id] if not data.empty else pd.DataFrame()
            for j, day in enumerate(days):
                ax = axes[i, j]
                day_data = d_mouse[d_mouse['day'] == day] if not d_mouse.empty else pd.DataFrame()
                if not day_data.empty:
                    sns.lineplot(
                        data=day_data, x='time', y='psth', errorbar='ci', hue='reward_group',
                        hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False
                    )
                ax.axvline(0, color=stim_palette[0], linestyle='-')
                ax.set_title(f'{group.capitalize()} LMI - Day {day}')
                ax.set_ylabel('DF/F0 (%)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        sns.despine()
        pdf.savefig(fig)
        plt.close(fig)


# ==================================================
# Lick-aligned false alarm PSTH's across days.
# ==================================================

# Parameters.
# -----------

sampling_rate = 30
win_sec = (-0.5, 5)  
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
# Add reward_group column to lmi_df
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)
# Select top_n positive and negative cells for each reward group independently
top_positive_lmi_cells = []
top_negative_lmi_cells = []
for group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == group]
    group_sorted = group_df.sort_values('lmi', ascending=False)
    top_positive_lmi_cells.append(group_sorted.head(top_n))
    top_negative_lmi_cells.append(group_sorted.tail(top_n))
top_positive_lmi_cells = pd.concat(top_positive_lmi_cells)
top_negative_lmi_cells = pd.concat(top_negative_lmi_cells)

# Define Non-LMI cells (lmi_p > 0.025 and < 0.975)
non_lmi_cells = lmi_df.loc[(lmi_df['lmi_p'] > 0.025) & (lmi_df['lmi_p'] < 0.975)]

# Load and preprocess data for all mice
positive_lmi_lick = []
negative_lmi_lick = []
non_lmi_lick = []

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

    # Positive LMI cells for this mouse
    positive_lmi_cells_for_mouse = top_positive_lmi_cells.loc[top_positive_lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_positive_lmi = xarr.sel(cell=xarr['roi'].isin(positive_lmi_cells_for_mouse))
    df_positive_lmi = xarr_positive_lmi.to_dataframe().reset_index()
    df_positive_lmi['mouse_id'] = mouse_id
    df_positive_lmi['reward_group'] = reward_group
    positive_lmi_lick.append(df_positive_lmi)

    # Negative LMI cells for this mouse
    negative_lmi_cells_for_mouse = top_negative_lmi_cells.loc[top_negative_lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_negative_lmi = xarr.sel(cell=xarr['roi'].isin(negative_lmi_cells_for_mouse))
    df_negative_lmi = xarr_negative_lmi.to_dataframe().reset_index()
    df_negative_lmi['mouse_id'] = mouse_id
    df_negative_lmi['reward_group'] = reward_group
    negative_lmi_lick.append(df_negative_lmi)

    # Non-LMI cells for this mouse
    non_lmi_cells_for_mouse = non_lmi_cells.loc[non_lmi_cells['mouse_id'] == mouse_id]['roi']
    xarr_non_lmi = xarr.sel(cell=xarr['roi'].isin(non_lmi_cells_for_mouse))
    df_non_lmi = xarr_non_lmi.to_dataframe().reset_index()
    df_non_lmi['mouse_id'] = mouse_id
    df_non_lmi['reward_group'] = reward_group
    non_lmi_lick.append(df_non_lmi)

positive_lmi_lick = pd.concat(positive_lmi_lick)
negative_lmi_lick = pd.concat(negative_lmi_lick)
non_lmi_lick = pd.concat(non_lmi_lick)

# Group by cell for PSTH
positive_lmi_lick = positive_lmi_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
negative_lmi_lick = negative_lmi_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
non_lmi_lick = non_lmi_lick.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()

# Convert to percent dF/F0
positive_lmi_lick['psth'] = positive_lmi_lick['psth'] * 100
negative_lmi_lick['psth'] = negative_lmi_lick['psth'] * 100
non_lmi_lick['psth'] = non_lmi_lick['psth'] * 100

# Generate a single figure with 3 rows for lick-aligned PSTH's (no helper function)
fig, axes = plt.subplots(3, len(days), figsize=(18, 12), sharey=True)
row_titles = [
    'Top 20 Positive LMI Cells (Lick-Aligned)',
    'Top 20 Negative LMI Cells (Lick-Aligned)',
    'Non-LMI Cells (Lick-Aligned)'
]
data_rows = [
    positive_lmi_lick,
    negative_lmi_lick,
    non_lmi_lick
]

for i, (row_title, data) in enumerate(zip(row_titles, data_rows)):
    for j, day in enumerate(days):
        d = data.loc[data['day'] == day]
        ax = axes[i, j]
        if not d.empty:
            sns.lineplot(
                data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=ax, legend=False
            )
        ax.axvline(0, color=stim_palette[0], linestyle='-')
        ax.set_title(f'{row_title} - Day {day}')
        ax.set_ylabel('DF/F0 (%)')

plt.tight_layout()
sns.despine()
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'lick_aligned_psth_during_learning.svg'), format='svg', dpi=300)
# plt.close(fig)



# ======================================================================
# Single trial responses for first N whisker hits during day 0 learning.
# ======================================================================

# Parameters
n_hits = 100  # Number of first whisker hits to plot
trials_per_row = 10  # Number of trials per row in the figure
block_size = 10  # Number of trials to average per block
sampling_rate = 30
win_sec = (-0.5, 6)
top_n = 50

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Add reward_group column to lmi_df
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)
# Select top_n positive and negative cells for each reward group independently
top_positive_lmi_cells = []
top_negative_lmi_cells = []
for group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == group]
    group_sorted = group_df.sort_values('lmi', ascending=False)
    top_positive_lmi_cells.append(group_sorted.head(top_n))
    top_negative_lmi_cells.append(group_sorted.tail(top_n))
top_positive_lmi_cells = pd.concat(top_positive_lmi_cells)
top_negative_lmi_cells = pd.concat(top_negative_lmi_cells)

mice = [m for m in mice if m != 'GF305']

# Prepare cell lists for each mouse
cell_groups = {}
for mouse_id in mice:
    cell_groups[mouse_id] = {
        'positive': top_positive_lmi_cells.loc[top_positive_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist(),
        'negative': top_negative_lmi_cells.loc[top_negative_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist()
    }

# Store data as numpy arrays in nested dictionaries
# Structure: hit_block_data[group][block_num] = {'traces': list of 1D arrays, 'reward_groups': list of str, 'time': 1D array}
hit_block_data = {
    'positive': {},
    'negative': {}
}

for mouse_id in mice:
    print(f"Processing mouse {mouse_id}...")

    # Load imaging data for day 0
    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)

    # Select day 0 only
    xarr = xarr.sel(trial=xarr['day'] == 0)
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

    # Get whisker trials
    whisker_trials = xarr.sel(trial=xarr['whisker_stim'] == 1)

    if len(whisker_trials.trial) == 0:
        print(f"  No whisker trials for {mouse_id}, skipping")
        continue

    # Find all whisker hits (trials where lick_flag == 1)
    lick_flags = whisker_trials['lick_flag'].values
    hit_indices = np.where(lick_flags == 1)[0]

    if len(hit_indices) == 0:
        print(f"  No whisker hits for {mouse_id}, skipping")
        continue

    # Take first n_hits (or all if fewer than n_hits)
    n_hits_available = min(n_hits, len(hit_indices))
    hit_indices_to_use = hit_indices[:n_hits_available]

    print(f"  Using first {n_hits_available} whisker hits")

    # Extract hit trials
    hit_trials = whisker_trials.isel(trial=hit_indices_to_use)

    # Get reward group
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Get time array (same for all trials)
    time_vals = hit_trials['time'].values

    # Calculate number of blocks
    n_blocks = int(np.ceil(n_hits_available / block_size))

    # Process each cell group
    for group in ['positive', 'negative']:
        cell_list = cell_groups[mouse_id][group]
        if not cell_list:
            continue

        # Select cells for this group
        xarr_group = hit_trials.sel(cell=hit_trials['roi'].isin(cell_list))

        # Process each block
        for block_num in range(n_blocks):
            # Get trial indices for this block
            start_hit = block_num * block_size
            end_hit = min(start_hit + block_size, n_hits_available)

            # Get data for this block of trials
            block_trials = xarr_group.isel(trial=slice(start_hit, end_hit))

            # Step 1: Average across trials for each cell
            block_cell_avg = block_trials.mean(dim='trial')

            # Step 2: Average across cells
            block_avg = block_cell_avg.mean(dim='cell').values * 100  # Convert to percent

            # Store block data
            block_idx = block_num + 1  # 1-indexed
            if block_idx not in hit_block_data[group]:
                hit_block_data[group][block_idx] = {
                    'traces': [],
                    'reward_groups': [],
                    'time': time_vals
                }
            hit_block_data[group][block_idx]['traces'].append(block_avg)
            hit_block_data[group][block_idx]['reward_groups'].append(reward_group)

# Plot block-averaged responses for each cell group separately
group_names = ['positive', 'negative']
group_titles = [f'Top {top_n} Positive LMI Cells', f'Top {top_n} Negative LMI Cells']

for group, group_title in zip(group_names, group_titles):
    if not hit_block_data[group]:
        print(f"No data for {group} group, skipping plot")
        continue

    max_block = max(hit_block_data[group].keys())
    n_rows = int(np.ceil(max_block / trials_per_row))
    n_cols = trials_per_row

    print(f"Plotting {max_block} blocks for {group} group in {n_rows} rows x {n_cols} cols")

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                            sharey=True, sharex=True)

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each block
    for block_num in range(1, max_block + 1):
        row_idx = (block_num - 1) // trials_per_row
        col_idx = (block_num - 1) % trials_per_row
        ax = axes[row_idx, col_idx]

        if block_num not in hit_block_data[group]:
            ax.axis('off')
            continue

        block_info = hit_block_data[group][block_num]
        time_vals = block_info['time']
        traces = np.array(block_info['traces'])  # (n_mice, n_timepoints)
        reward_groups = np.array(block_info['reward_groups'])

        # Plot separately for each reward group
        reward_colors = {'R-': reward_palette[0], 'R+': reward_palette[1]}
        for rg, color in zip(['R-', 'R+'], [reward_colors['R-'], reward_colors['R+']]):
            mask = reward_groups == rg
            if not mask.any():
                continue

            rg_traces = traces[mask]
            mean_trace = rg_traces.mean(axis=0)

            # Compute 95% confidence interval
            if len(rg_traces) > 1:
                sem = rg_traces.std(axis=0) / np.sqrt(len(rg_traces))
                ci = 1.96 * sem
                ax.fill_between(time_vals, mean_trace - ci, mean_trace + ci,
                               color=color, alpha=0.2)

            ax.plot(time_vals, mean_trace, color=color, linewidth=1.5)

        if group == 'positive':
            ax.set_ylim(-50, 200)
        else:
            ax.set_ylim(-50, 100)
        # Mark stimulus onset
        ax.axvline(0, color=stim_palette[0], linestyle='-', alpha=0.5)

        # Title showing block and hit range
        start_hit = (block_num - 1) * block_size + 1
        end_hit = min(block_num * block_size, n_hits)
        if block_size == 1:
            ax.set_title(f'Hit #{block_num}', fontsize=9)
        else:
            if start_hit == end_hit:
                ax.set_title(f'Block {block_num}\n(Hit {start_hit})', fontsize=9)
            else:
                ax.set_title(f'Block {block_num}\n(Hits {start_hit}-{end_hit})', fontsize=9)
        ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
        ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')

    # Hide unused subplots
    for block_num in range(max_block + 1, n_rows * n_cols + 1):
        row_idx = (block_num - 1) // trials_per_row
        col_idx = (block_num - 1) % trials_per_row
        axes[row_idx, col_idx].axis('off')

    if block_size == 1:
        title_str = f'{group_title} - First {max_block} Whisker Hits (Day 0)'
    else:
        total_hits = min(max_block * block_size, n_hits)
        title_str = f'{group_title} - First {total_hits} Whisker Hits (Day 0) - {max_block} Blocks of {block_size}'
    plt.suptitle(title_str, fontsize=14, y=0.995)
    plt.tight_layout()
    sns.despine()
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
    output_dir = io.adjust_path_to_host(output_dir)
    if block_size == 1:
        filename = f'first_{max_block}_whisker_hits_top_{top_n}_{group}.svg'
    else:
        filename = f'first_{total_hits}_whisker_hits_top_{top_n}_{group}_block{block_size}.svg'
    plt.savefig(os.path.join(output_dir, filename), format='svg', dpi=300)
    # plt.close(fig)

print("Single trial hit analysis complete!")




# ===========================================================
# Single trial responses for first N false alarms (lick-aligned) during day 0 learning.
# ===========================================================

# Parameters
n_false_alarms = 100  # Number of first false alarms to plot
trials_per_row = 10  # Number of trials per row in the figure
block_size = 5  # Number of trials to average per block
sampling_rate = 30
win_sec = (-0.5, 5)  # Lick-aligned window
top_n = 50

# Store data as numpy arrays in nested dictionaries
# Structure: false_alarm_block_data[group][block_num] = {'traces': list of 1D arrays, 'reward_groups': list of str, 'time': 1D array}
false_alarm_block_data = {
    'positive': {},
    'negative': {}
}

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
# Add reward_group column to lmi_df
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)
# Select top_n positive and negative cells for each reward group independently
top_positive_lmi_cells = []
top_negative_lmi_cells = []
for group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == group]
    group_sorted = group_df.sort_values('lmi', ascending=False)
    top_positive_lmi_cells.append(group_sorted.head(top_n))
    top_negative_lmi_cells.append(group_sorted.tail(top_n))
top_positive_lmi_cells = pd.concat(top_positive_lmi_cells)
top_negative_lmi_cells = pd.concat(top_negative_lmi_cells)

# Prepare cell lists for each mouse (only positive and negative LMI cells)
cell_groups = {}
for mouse_id in mice:
    cell_groups[mouse_id] = {
        'positive': top_positive_lmi_cells.loc[top_positive_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist(),
        'negative': top_negative_lmi_cells.loc[top_negative_lmi_cells['mouse_id'] == mouse_id]['roi'].tolist()
    }

# Remove mouse GF305 for now (due to data issues)
mice = [m for m in mice if m != 'GF305']

for mouse_id in mice:
    print(f"Processing mouse {mouse_id} for false alarms...")

    # Load lick-aligned imaging data for day 0
    file_name = 'lick_aligned_xarray.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

    # Select day 0 only
    xarr = xarr.sel(trial=xarr['day'] == 0)
    xarr = xarr.sel(time=slice(win_sec[0], win_sec[1]))

    # Get false alarm trials (no_stim == 1 and lick_flag == 1)
    false_alarm_trials = xarr.sel(trial=(xarr['no_stim'] == 1) & (xarr['lick_flag'] == 1))

    if len(false_alarm_trials.trial) == 0:
        print(f"  No false alarm trials for {mouse_id}, skipping")
        continue

    # Get total number of false alarms available
    n_trials_total = len(false_alarm_trials.trial)
    # Take up to n_false_alarms
    n_trials_to_use = min(n_false_alarms, n_trials_total)
    false_alarm_trials = false_alarm_trials.isel(trial=slice(0, n_trials_to_use))

    print(f"  Mouse has {n_trials_total} false alarm trials, using first {n_trials_to_use}")

    # Get reward group
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Get time array (same for all trials)
    time_vals = false_alarm_trials['time'].values

    # Calculate number of blocks
    n_blocks = int(np.ceil(n_trials_to_use / block_size))

    # Process each cell group
    for group in ['positive', 'negative']:
        cell_list = cell_groups[mouse_id][group]
        if not cell_list:
            continue

        # Select cells for this group
        xarr_group = false_alarm_trials.sel(cell=false_alarm_trials['roi'].isin(cell_list))

        # Process each block
        for block_num in range(n_blocks):
            # Get trial indices for this block
            start_trial = block_num * block_size
            end_trial = min(start_trial + block_size, n_trials_to_use)

            # Get data for this block of trials
            block_trials = xarr_group.isel(trial=slice(start_trial, end_trial))

            # Step 1: Average across trials for each cell
            block_cell_avg = block_trials.mean(dim='trial')

            # Step 2: Average across cells
            block_avg = block_cell_avg.mean(dim='cell').values * 100  # Convert to percent

            # Store block data
            block_idx = block_num + 1  # 1-indexed
            if block_idx not in false_alarm_block_data[group]:
                false_alarm_block_data[group][block_idx] = {
                    'traces': [],
                    'reward_groups': [],
                    'time': time_vals
                }
            false_alarm_block_data[group][block_idx]['traces'].append(block_avg)
            false_alarm_block_data[group][block_idx]['reward_groups'].append(reward_group)

# Plot block-averaged responses for each cell group separately
group_names = ['positive', 'negative']
group_titles = [f'Top {top_n} Positive LMI Cells', f'Top {top_n} Negative LMI Cells']

for group, group_title in zip(group_names, group_titles):
    if not false_alarm_block_data[group]:
        print(f"No false alarm data for {group} group, skipping plot")
        continue

    max_block = max(false_alarm_block_data[group].keys())
    n_rows = int(np.ceil(max_block / trials_per_row))
    n_cols = trials_per_row

    print(f"Plotting {max_block} blocks for {group} group in {n_rows} rows x {n_cols} cols")

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                            sharey=True, sharex=True)

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each block
    for block_num in range(1, max_block + 1):
        row_idx = (block_num - 1) // trials_per_row
        col_idx = (block_num - 1) % trials_per_row
        ax = axes[row_idx, col_idx]

        if block_num not in false_alarm_block_data[group]:
            ax.axis('off')
            continue

        block_info = false_alarm_block_data[group][block_num]
        time_vals = block_info['time']
        traces = np.array(block_info['traces'])  # (n_mice, n_timepoints)
        reward_groups = np.array(block_info['reward_groups'])

        # Plot separately for each reward group
        reward_colors = {'R-': reward_palette[0], 'R+': reward_palette[1]}
        for rg, color in zip(['R-', 'R+'], [reward_colors['R-'], reward_colors['R+']]):
            mask = reward_groups == rg
            if not mask.any():
                continue

            rg_traces = traces[mask]
            mean_trace = rg_traces.mean(axis=0)

            # Compute 95% confidence interval
            if len(rg_traces) > 1:
                sem = rg_traces.std(axis=0) / np.sqrt(len(rg_traces))
                ci = 1.96 * sem
                ax.fill_between(time_vals, mean_trace - ci, mean_trace + ci,
                               color=color, alpha=0.2)

            ax.plot(time_vals, mean_trace, color=color, linewidth=1.5)

        if group == 'positive':
            ax.set_ylim(-100, 200)
        else:
            ax.set_ylim(-50, 150)

        # Mark lick time (aligned to 0)
        ax.axvline(0, color='red', linestyle='-', alpha=0.5, linewidth=1.5)

        # Title showing block and trial range
        start_trial = (block_num - 1) * block_size + 1
        end_trial = min(block_num * block_size, n_false_alarms)
        if block_size == 1:
            ax.set_title(f'FA #{block_num}', fontsize=9)
        else:
            if start_trial == end_trial:
                ax.set_title(f'Block {block_num}\n(FA {start_trial})', fontsize=9)
            else:
                ax.set_title(f'Block {block_num}\n(FAs {start_trial}-{end_trial})', fontsize=9)
        ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
        ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')

    # Hide unused subplots
    for block_num in range(max_block + 1, n_rows * n_cols + 1):
        row_idx = (block_num - 1) // trials_per_row
        col_idx = (block_num - 1) % trials_per_row
        axes[row_idx, col_idx].axis('off')

    if block_size == 1:
        title_str = f'{group_title} - First {max_block} False Alarms (Lick-Aligned, Day 0)'
    else:
        total_trials = min(max_block * block_size, n_false_alarms)
        title_str = f'{group_title} - First {total_trials} False Alarms (Lick-Aligned, Day 0) - {max_block} Blocks of {block_size}'
    plt.suptitle(title_str, fontsize=14, y=0.995)
    plt.tight_layout()
    sns.despine()

    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
    output_dir = io.adjust_path_to_host(output_dir)
    if block_size == 1:
        filename = f'first_{max_block}_false_alarms_lick_aligned_top_{top_n}_{group}.svg'
    else:
        filename = f'first_{total_trials}_false_alarms_lick_aligned_top_{top_n}_{group}_block{block_size}.svg'
    plt.savefig(os.path.join(output_dir, filename), format='svg', dpi=300)
    # plt.close(fig)

print("False alarm analysis complete!")



# Plot distribution of number of false alarms (licks on no-stim trials) during day 0 learning across mice

false_alarm_counts = []
for mouse_id in mice:
    file_name = 'lick_aligned_xarray.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
    # Select day 0 only
    xarr_day0 = xarr.sel(trial=xarr['day'] == 0)
    # False alarm trials: no_stim == 1 and lick_flag == 1
    fa_trials = xarr_day0.sel(trial=(xarr_day0['no_stim'] == 1) & (xarr_day0['lick_flag'] == 1))
    fa_count = len(fa_trials.trial)
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    false_alarm_counts.append({'mouse_id': mouse_id, 'false_alarms': fa_count, 'reward_group': reward_group})

fa_df = pd.DataFrame(false_alarm_counts)

plt.figure(figsize=(8, 5))
sns.barplot(data=fa_df, x='mouse_id', y='false_alarms', hue='reward_group', hue_order=['R-', 'R+'], palette=reward_palette)
plt.ylabel('Number of False Alarms (Day 0)')
plt.xlabel('Mouse ID')
plt.title('Distribution of False Alarms (Day 0 Learning) Across Mice by Reward Group')
plt.tight_layout()
sns.despine()
plt.show()


# ######################








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



# ===========================================================
# Per-cell trial-by-trial responses for top LMI cells
# ===========================================================

# Parameters
n_cells_per_group = 10  # Number of top positive/negative cells per reward group
n_trials_max = 300  # Maximum number of trials to plot per cell
trials_per_row = 10  # Number of trials per row in grid
block_size = 1  # Number of trials to average per block
sampling_rate = 30
win_sec_stim = (-0.5, 6)  # Time window for stimulus-aligned data
win_sec_lick = (-0.5, 5)  # Time window for lick-aligned data

# Output PDF
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
output_dir = io.adjust_path_to_host(output_dir)
output_pdf = os.path.join(output_dir, f'per_cell_trial_responses_block_{block_size}.pdf')

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

# Load LMI data
lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)

# Select top n_cells_per_group positive and negative cells per reward group
selected_cells = []
for reward_group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == reward_group]
    group_sorted = group_df.sort_values('lmi', ascending=False)

    # Top positive cells
    top_pos = group_sorted.head(n_cells_per_group).copy()
    top_pos['lmi_group'] = 'positive'
    selected_cells.append(top_pos)

    # Top negative cells
    top_neg = group_sorted.tail(n_cells_per_group).copy()
    top_neg['lmi_group'] = 'negative'
    selected_cells.append(top_neg)

selected_cells_df = pd.concat(selected_cells, ignore_index=True)

mice = [m for m in mice if m != 'GF305']

print(f"\nGenerating per-cell trial-by-trial analysis for {len(selected_cells_df)} cells...")
print(f"Selected cells per reward group: {n_cells_per_group} positive, {n_cells_per_group} negative")

with PdfPages(output_pdf) as pdf:
    for idx, cell_row in selected_cells_df.iterrows():
        mouse_id = cell_row['mouse_id']
        roi = cell_row['roi']
        lmi_group = cell_row['lmi_group']
        reward_group = cell_row['reward_group']
        lmi_value = cell_row['lmi']

        print(f"\nProcessing cell {idx+1}/{len(selected_cells_df)}: {mouse_id} ROI {roi} ({lmi_group} LMI, {reward_group})")

        # Get color for this cell based on reward group
        cell_color = reward_palette[0] if reward_group == 'R-' else reward_palette[1]

        # Load stimulus-aligned data
        file_name = 'tensor_xarray_learning_data.nc'
        folder = os.path.join(io.processed_dir, 'mice')
        xarr_stim = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)
        xarr_stim = xarr_stim.sel(trial=xarr_stim['day'] == 0)
        xarr_stim = xarr_stim.sel(time=slice(win_sec_stim[0], win_sec_stim[1]))

        # Select this specific cell
        cell_idx = np.where(xarr_stim['roi'].values == roi)[0]
        if len(cell_idx) == 0:
            print(f"  Cell ROI {roi} not found in data, skipping")
            continue
        xarr_cell = xarr_stim.isel(cell=cell_idx[0])

        # Load lick-aligned data for false alarms
        file_name_lick = 'lick_aligned_xarray.nc'
        xarr_lick = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_lick, substracted=False)
        xarr_lick = xarr_lick.sel(trial=xarr_lick['day'] == 0)
        xarr_lick = xarr_lick.sel(time=slice(win_sec_lick[0], win_sec_lick[1]))
        cell_idx_lick = np.where(xarr_lick['roi'].values == roi)[0]
        if len(cell_idx_lick) == 0:
            print(f"  Cell ROI {roi} not found in lick-aligned data")
            xarr_cell_lick = None
        else:
            xarr_cell_lick = xarr_lick.isel(cell=cell_idx_lick[0])

        # ===== 1. Whisker Hits =====
        whisker_hits = xarr_cell.sel(trial=(xarr_cell['whisker_stim'] == 1) & (xarr_cell['lick_flag'] == 1))
        n_trials = min(n_trials_max, len(whisker_hits.trial))

        if n_trials > 0:
            whisker_hits = whisker_hits.isel(trial=slice(0, n_trials))
            n_blocks = int(np.ceil(n_trials / block_size))

            # Compute block averages
            block_data = []
            for block_num in range(n_blocks):
                start_trial = block_num * block_size
                end_trial = min(start_trial + block_size, n_trials)
                block_trials = whisker_hits.isel(trial=slice(start_trial, end_trial))
                block_avg = block_trials.mean(dim='trial').values * 100
                block_data.append(block_avg)

            # Plot
            n_rows = int(np.ceil(n_blocks / trials_per_row))
            n_cols = trials_per_row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                    sharey=True, sharex=True)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            time_vals = whisker_hits['time'].values
            for block_num in range(n_blocks):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                ax = axes[row_idx, col_idx]

                ax.plot(time_vals, block_data[block_num], color=cell_color, linewidth=1.5)
                ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)

                start_trial = block_num * block_size + 1
                end_trial = min((block_num + 1) * block_size, n_trials)
                if block_size == 1:
                    ax.set_title(f'Trial {block_num + 1}', fontsize=8)
                else:
                    if start_trial == end_trial:
                        ax.set_title(f'Block {block_num + 1}\n(Trial {start_trial})', fontsize=8)
                    else:
                        ax.set_title(f'Block {block_num + 1}\n(Trials {start_trial}-{end_trial})', fontsize=8)
                ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')
                ax.tick_params(labelsize=7)

            # Hide unused subplots
            for block_num in range(n_blocks, n_rows * n_cols):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                axes[row_idx, col_idx].axis('off')

            plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\nWhisker Hits (Day 0) - {n_blocks} Blocks',
                        fontsize=12, y=0.998)
            plt.tight_layout()
            sns.despine()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Whisker hits: {n_trials} trials, {n_blocks} blocks")
        else:
            print(f"  No whisker hits")

        # ===== 2. Whisker Misses =====
        whisker_misses = xarr_cell.sel(trial=(xarr_cell['whisker_stim'] == 1) & (xarr_cell['lick_flag'] == 0))
        n_trials = min(n_trials_max, len(whisker_misses.trial))

        if n_trials > 0:
            whisker_misses = whisker_misses.isel(trial=slice(0, n_trials))
            n_blocks = int(np.ceil(n_trials / block_size))

            # Compute block averages
            block_data = []
            for block_num in range(n_blocks):
                start_trial = block_num * block_size
                end_trial = min(start_trial + block_size, n_trials)
                block_trials = whisker_misses.isel(trial=slice(start_trial, end_trial))
                block_avg = block_trials.mean(dim='trial').values * 100
                block_data.append(block_avg)

            # Plot
            n_rows = int(np.ceil(n_blocks / trials_per_row))
            n_cols = trials_per_row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                    sharey=True, sharex=True)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            time_vals = whisker_misses['time'].values
            for block_num in range(n_blocks):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                ax = axes[row_idx, col_idx]

                ax.plot(time_vals, block_data[block_num], color=cell_color, linestyle='--', linewidth=1.5)
                ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)

                start_trial = block_num * block_size + 1
                end_trial = min((block_num + 1) * block_size, n_trials)
                if block_size == 1:
                    ax.set_title(f'Trial {block_num + 1}', fontsize=8)
                else:
                    if start_trial == end_trial:
                        ax.set_title(f'Block {block_num + 1}\n(Trial {start_trial})', fontsize=8)
                    else:
                        ax.set_title(f'Block {block_num + 1}\n(Trials {start_trial}-{end_trial})', fontsize=8)
                ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')
                ax.tick_params(labelsize=7)

            # Hide unused subplots
            for block_num in range(n_blocks, n_rows * n_cols):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                axes[row_idx, col_idx].axis('off')

            plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\nWhisker Misses (Day 0) - {n_blocks} Blocks',
                        fontsize=12, y=0.998)
            plt.tight_layout()
            sns.despine()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Whisker misses: {n_trials} trials, {n_blocks} blocks")
        else:
            print(f"  No whisker misses")

        # ===== 3. Auditory Hits =====
        auditory_hits = xarr_cell.sel(trial=(xarr_cell['auditory_stim'] == 1) & (xarr_cell['lick_flag'] == 1))
        n_trials = min(n_trials_max, len(auditory_hits.trial))

        if n_trials > 0:
            auditory_hits = auditory_hits.isel(trial=slice(0, n_trials))
            n_blocks = int(np.ceil(n_trials / block_size))

            # Compute block averages
            block_data = []
            for block_num in range(n_blocks):
                start_trial = block_num * block_size
                end_trial = min(start_trial + block_size, n_trials)
                block_trials = auditory_hits.isel(trial=slice(start_trial, end_trial))
                block_avg = block_trials.mean(dim='trial').values * 100
                block_data.append(block_avg)

            # Plot
            n_rows = int(np.ceil(n_blocks / trials_per_row))
            n_cols = trials_per_row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                    sharey=True, sharex=True)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            time_vals = auditory_hits['time'].values
            for block_num in range(n_blocks):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                ax = axes[row_idx, col_idx]

                ax.plot(time_vals, block_data[block_num], color=cell_color, linewidth=1.5)
                ax.axvline(0, color=stim_palette[0], linestyle='-',  linewidth=1.5)

                start_trial = block_num * block_size + 1
                end_trial = min((block_num + 1) * block_size, n_trials)
                if block_size == 1:
                    ax.set_title(f'Trial {block_num + 1}', fontsize=8)
                else:
                    if start_trial == end_trial:
                        ax.set_title(f'Block {block_num + 1}\n(Trial {start_trial})', fontsize=8)
                    else:
                        ax.set_title(f'Block {block_num + 1}\n(Trials {start_trial}-{end_trial})', fontsize=8)
                ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')
                ax.tick_params(labelsize=7)

            # Hide unused subplots
            for block_num in range(n_blocks, n_rows * n_cols):
                row_idx = block_num // trials_per_row
                col_idx = block_num % trials_per_row
                axes[row_idx, col_idx].axis('off')

            plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\nAuditory Hits (Day 0) - {n_blocks} Blocks',
                        fontsize=12, y=0.998)
            plt.tight_layout()
            sns.despine()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Auditory hits: {n_trials} trials, {n_blocks} blocks")
        else:
            print(f"  No auditory hits")

        # ===== 4. False Alarms (Lick-aligned) =====
        if xarr_cell_lick is not None:
            false_alarms = xarr_cell_lick.sel(trial=(xarr_cell_lick['no_stim'] == 1) & (xarr_cell_lick['lick_flag'] == 1))
            n_trials = min(n_trials_max, len(false_alarms.trial))

            if n_trials > 0:
                false_alarms = false_alarms.isel(trial=slice(0, n_trials))
                n_blocks = int(np.ceil(n_trials / block_size))

                # Compute block averages
                block_data = []
                for block_num in range(n_blocks):
                    start_trial = block_num * block_size
                    end_trial = min(start_trial + block_size, n_trials)
                    block_trials = false_alarms.isel(trial=slice(start_trial, end_trial))
                    block_avg = block_trials.mean(dim='trial').values * 100
                    block_data.append(block_avg)

                # Plot
                n_rows = int(np.ceil(n_blocks / trials_per_row))
                n_cols = trials_per_row
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                        sharey=True, sharex=True)
                if n_rows == 1:
                    axes = axes.reshape(1, -1)

                time_vals = false_alarms['time'].values
                for block_num in range(n_blocks):
                    row_idx = block_num // trials_per_row
                    col_idx = block_num % trials_per_row
                    ax = axes[row_idx, col_idx]

                    ax.plot(time_vals, block_data[block_num], color=cell_color, linewidth=1.5)
                    ax.axvline(0, color=stim_palette[2], linestyle='-', linewidth=1.5)
                    start_trial = block_num * block_size + 1
                    end_trial = min((block_num + 1) * block_size, n_trials)
                    if block_size == 1:
                        ax.set_title(f'Trial {block_num + 1}', fontsize=8)
                    else:
                        if start_trial == end_trial:
                            ax.set_title(f'Block {block_num + 1}\n(Trial {start_trial})', fontsize=8)
                        else:
                            ax.set_title(f'Block {block_num + 1}\n(Trials {start_trial}-{end_trial})', fontsize=8)
                    ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                    ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')
                    ax.tick_params(labelsize=7)

                # Hide unused subplots
                for block_num in range(n_blocks, n_rows * n_cols):
                    row_idx = block_num // trials_per_row
                    col_idx = block_num % trials_per_row
                    axes[row_idx, col_idx].axis('off')

                plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\nFalse Alarms - Lick Aligned (Day 0) - {n_blocks} Blocks',
                            fontsize=12, y=0.998)
                plt.tight_layout()
                sns.despine()
                pdf.savefig(fig)
                plt.close(fig)
                print(f"  False alarms: {n_trials} trials, {n_blocks} blocks")
            else:
                print(f"  No false alarms")

print(f"\nPer-cell analysis complete! PDF saved to: {output_pdf}")
