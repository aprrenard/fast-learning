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
from scipy.stats import wilcoxon


# #############################################################################
# Gradual learning during Day 0.
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
substract_baseline = True
average_inside_days = True
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

processed_folder = io.solve_common_paths('processed_data')  

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)


# excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
# mice = [m for m in mice if m not in excluded_mice]


# Load LMI dataframe.
# -------------------

lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))


# Load data.
#  ----------

dfs = []
for mouse in mice:
    print(mouse)
    processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, processed_dir, file_name)
    if substract_baseline:
        xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    
    # Select day 0. 
    xarray = xarray.sel(trial=xarray['day'] == 0)
    # Select whisker trials.
    xarray = xarray.sel(trial=xarray['whisker_stim']==1)
    # Average time bin.
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    # Select positive LMI cells.
    lmi_pos = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p>=0.975), 'roi']
    xa_pos = xarray.sel(cell=xarray['roi'].isin(lmi_pos))
    print(xa_pos.shape)
    
    # Select negative LMI cells.
    lmi_neg = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p<=0.025), 'roi']
    xa_neg = xarray.sel(cell=xarray['roi'].isin(lmi_neg))
    print(xa_neg.shape)
    
    xa_pos.name = 'activity'
    xa_neg.name = 'activity'
    df_pos = xa_pos.to_dataframe().reset_index()
    df_pos['lmi'] = 'positive'
    df_neg = xa_neg.to_dataframe().reset_index()
    df_neg['lmi'] = 'negative'
    df = pd.concat([df_pos, df_neg])
    df['mouse_id'] = mouse
    df['reward_group'] = rew_gp
    dfs.append(df)
    # Close the xarray dataset.
    xarray.close()
dfs = pd.concat(dfs)

# Plot.

data = dfs.groupby(['mouse_id', 'lmi', 'reward_group', 'trial_w', 'cell_type'])['activity'].agg('mean').reset_index()

# sns.relplot(data=data.loc[data.trial<100], x='trial', y='activity', col='lmi', row='reward_group', hue='cell_type', kind='line', palette=cell_types_palette, height=3, aspect=0.8)
sns.relplot(data=data.loc[data.trial_w<100], x='trial_w', y='activity',
            col='lmi', row='reward_group', kind='line',
            palette=reward_palette, height=3, aspect=0.8,
            col_order=['positive', 'negative'],
            row_order=['R+', 'R-'],)






# #############################################################################
# Gradual learning during Day 0 realigned to "learning trial".
# #############################################################################

# keep plot before meeting with the most gradual four mice (from GF)

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
substract_baseline = True
average_inside_days = True
realigned_to_learning = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

processed_folder = io.solve_common_paths('processed_data')  

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)


# excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
# mice = [m for m in mice if m not in excluded_mice]
# mice = ['GF305', 'GF306', 'GF317', 'GF323', 'GF318', 'GF313']
# mice = ['GF305', 'GF306', 'GF317', ]

# Load LMI dataframe.
# -------------------

lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

learning_trials = {'GF305':138, 'GF306': 200, 'GF317': 96, 'GF323': 211, 'GF318': 208, 'GF313': 141}


# Load data.
#  ----------

dfs = []
for mouse in mice:
    print(mouse)
    processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, processed_dir, file_name)
    if substract_baseline:
        xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    
    # Select day 0. 
    xarray = xarray.sel(trial=xarray['day'] == 0)
    # Select whisker trials.
    xarray = xarray.sel(trial=xarray['whisker_stim']==1)
    
    if realigned_to_learning:
        eureka = learning_trials[mouse]
        eureka_w = xarray.sel(trial=xarray['trial_id']==eureka).trial_w.values
        # Select trials around the learning trial.
        xarray = xarray.sel(trial=xarray['trial_w']>=eureka_w-60)
        xarray = xarray.sel(trial=xarray['trial_w']<=eureka_w+20)
        xarray.coords['trial_w'] = xarray['trial_w'] - eureka_w
    
    # Average time bin.
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    # Select positive LMI cells.
    lmi_pos = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p>=0), 'roi']
    xa_pos = xarray.sel(cell=xarray['roi'].isin(lmi_pos))
    print(xa_pos.shape)
    
    # Select negative LMI cells.
    lmi_neg = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p<=0.025), 'roi']
    xa_neg = xarray.sel(cell=xarray['roi'].isin(lmi_neg))
    print(xa_neg.shape)
    
    xa_pos.name = 'activity'
    xa_neg.name = 'activity'
    df_pos = xa_pos.to_dataframe().reset_index()
    df_pos['lmi'] = 'positive'
    df_neg = xa_neg.to_dataframe().reset_index()
    df_neg['lmi'] = 'negative'
    df = pd.concat([df_pos, df_neg])
    df['mouse_id'] = mouse
    df['reward_group'] = rew_gp
    dfs.append(df)
    # Close the xarray dataset.
    xarray.close()
dfs = pd.concat(dfs)

# Plot.

data = dfs.groupby(['mouse_id', 'lmi', 'reward_group', 'trial_w', 'cell_type'])[['activity', 'outcome_w']].agg('mean').reset_index()

# Smooth activity and outcome with a rolling window.
rolling_window = 10  # Define the size of the rolling window.
data['activity_smoothed'] = data.groupby(['mouse_id', 'lmi', 'reward_group', 'cell_type'])['activity'].transform(lambda x: x.rolling(rolling_window, center=True).mean())
data['outcome_w_smoothed'] = data.groupby(['mouse_id', 'reward_group'])['outcome_w'].transform(lambda x: x.rolling(rolling_window, center=True).mean())

plt.figure(dpi=300,)
sns.lineplot(data=data, x='trial_w', y='hr_w',
            palette=reward_palette)
sns.despine()
plt.figure(dpi=300, figsize=(15, 5))
sns.relplot(data=data, x='trial_w', y='activity',
            col='lmi', row='reward_group', kind='line', palette=reward_palette,
            height=3, aspect=0.8, col_order=['positive', 'negative'])
# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/sensory_plasticity/gradual_learning'
svg_file = 'gradual_potentiation.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# sns.relplot(data=data, x='trial_w', y='activity', col='lmi', row='reward_group', hue='cell_type', kind='line', palette=cell_types_palette, height=3, aspect=0.8)

# Same but smoothed
plt.figure(dpi=300,)
sns.lineplot(data=data, x='trial_w', y='outcome_w_smoothed',
            palette=reward_palette)
sns.despine()
plt.figure(dpi=300, figsize=(15, 5))
sns.relplot(data=data, x='trial_w', y='activity_smoothed',
            col='lmi', row='reward_group', kind='line', palette=reward_palette,
            height=3, aspect=0.8, col_order=['positive', 'negative'])
# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/sensory_plasticity/gradual_learning'
svg_file = 'gradual_potentiation_smoothed.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# plt.plot(data.loc[data.mouse_id=='GF305'].outcome_w)

# d = dfs.loc[dfs.mouse_id=='GF306']
# plt.scatter(x=d.loc[d.lick_flag==0]['trial_w'],y=d.loc[d.lick_flag==0]['outcome_w']-0.15)
# plt.scatter(x=d.loc[d.lick_flag==1]['trial_w'],y=d.loc[d.lick_flag==1]['outcome_w']-1.15)


# # Remove mice that licked to first whisker stim.
# mice_to_keep = dfs.loc[(dfs.trial_w==1.0) & (dfs.outcome_w==0.0), 'mouse_id'].unique()
# # mice_to_keep = mice_to_keep[-6:]
# data = dfs.loc[(dfs.mouse_id.isin(mice_to_keep))]
# data = data.loc[(data.outcome_w==1.0)]
# data = data.groupby(['mouse_id', 'lmi', 'cell', 'reward_group', 'trial'])['activity'].agg('mean').reset_index()
# data = data.loc[(data.reward_group=='R+') & (data.lmi=='positive')]
# data = data.loc[data.trial<100]
# sns.lineplot(data=data, x='trial', y='activity', palette=cell_types_palette)

# dfs.mouse_id.unique()












# #############################################################################
# PSTH during learning.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-0.5, 1.5)
win_bin = (0, 0.180)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [0]
days_str = ['0']

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name, substracted=True)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh = xarr_wh.groupby('day').mean(dim='trial')
    
    # Select auditory trials.
    xarr_aud = xarr.sel(trial=xarr['auditory_stim']==1)
    # Average trials per days.
    xarr_aud = xarr_aud.groupby('day').mean(dim='trial')
    
    # Select no stim trials.
    xarr_ns = xarr.sel(trial=xarr['no_stim']==1)
    # Average trials per days.
    xarr_ns = xarr_ns.groupby('day').mean(dim='trial')
    # 
    xarr_wh.name = 'psth'
    xarr_wh = xarr_wh.to_dataframe().reset_index()
    xarr_wh['mouse_id'] = mouse_id
    xarr_wh['reward_group'] = reward_group
    xarr_wh['stim'] = 'whisker'
    psth.append(xarr_wh)
    
    xarr_aud.name = 'psth'
    xarr_aud = xarr_aud.to_dataframe().reset_index()
    xarr_aud['mouse_id'] = mouse_id
    xarr_aud['reward_group'] = reward_group
    xarr_aud['stim'] = 'auditory'
    psth.append(xarr_aud)
    
    xarr_ns.name = 'psth'
    xarr_ns = xarr_ns.to_dataframe().reset_index()
    xarr_ns['mouse_id'] = mouse_id
    xarr_ns['reward_group'] = reward_group
    xarr_ns['stim'] = 'no_stim'
    psth.append(xarr_ns)

psth = pd.concat(psth)


# PSTH for the three stimuli.
# ###########################

variance = 'mice'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data = imaging_utils.filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim'])['psth'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim'])['psth'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim'])['psth'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim'])['psth'].agg('mean').reset_index()
# Convert data to percent dF/F0.
data['psth'] = data['psth'] * 100


# Plot for all cells.
# -------------------

fig, axes = plt.subplots(3, figsize=(3, 8), sharey=True)

# Whisker stim.
d = data.loc[(data['stim'] == 'whisker')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0], legend=False)
axes[0].axvline(0, color='#FF9600', linestyle='--')

# Auditory stim.
d = data.loc[(data['stim'] == 'auditory')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1], legend=False)
axes[1].axvline(0, color='#1f77b4', linestyle='--')

# No stim.
d = data.loc[(data['stim'] == 'no_stim')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2], legend=False)
axes[2].axvline(0, color='#333333', linestyle='--')

axes[0].set_title('All cells')
axes[0].set_ylabel('Whisker stim')
axes[1].set_ylabel('Auditory stim')
axes[2].set_ylabel('No stim')
plt.ylim(-1, 12)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_three_stim_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Plot for projectors.
# --------------------

cell_types = ['wM1', 'wS2']
fig, axes = plt.subplots(3, 2, figsize=(6, 8), sharey=True)

# Whisker stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'whisker')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'whisker')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0, j], legend=False)
    axes[0, j].axvline(0, color='#FF9600', linestyle='--')

# Auditory stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'auditory')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'auditory')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1, j], legend=False)
    axes[1, j].axvline(0, color='#1f77b4', linestyle='--')

# No stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'no_stim')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'no_stim')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2, j], legend=False)
    axes[2, j].axvline(0, color='#333333', linestyle='--')

axes[0, 0].set_title('wM1 projectors')
axes[0, 1].set_title('wS2 projectors')
axes[0, 0].set_ylabel('Whisker stim')
axes[1, 0].set_ylabel('Auditory stim')
axes[2, 0].set_ylabel('No stim')
plt.ylim(-1, 20)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_three_stim_projectors_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


















# #############################################################################
# PSTH during learning. Early vs last trials.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-0.5, 1.5)
win_bin = (0, 0.180)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [0]
days_str = ['0']
early_trials = range(0, 10)
late_trials = range(20, 200)  # Last trials of the session.
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh_early = xarr_wh.sel(trial=xarr_wh['trial_w'].isin(early_trials))
    xarr_wh_late = xarr_wh.sel(trial=xarr_wh['trial_w'].isin(late_trials))
    xarr_wh_early = xarr_wh_early.groupby('day').mean(dim='trial')
    xarr_wh_late = xarr_wh_late.groupby('day').mean(dim='trial')
    
    xarr_wh_early.name = 'activity'
    xarr_wh_early = xarr_wh_early.to_dataframe().reset_index()
    xarr_wh_early['mouse_id'] = mouse_id
    xarr_wh_early['reward_group'] = reward_group
    xarr_wh_early['stim'] = 'whisker'
    xarr_wh_early['epoch'] = 'early'
    psth.append(xarr_wh_early)
    
    xarr_wh_late.name = 'activity'
    xarr_wh_late = xarr_wh_late.to_dataframe().reset_index()
    xarr_wh_late['mouse_id'] = mouse_id
    xarr_wh_late['reward_group'] = reward_group
    xarr_wh_late['stim'] = 'whisker'
    xarr_wh_late['epoch'] = 'late'
    psth.append(xarr_wh_late)

psth = pd.concat(psth)


variance = 'mice'  # 'mice' or 'cells'
if variance == "mice":
    min_cells = 3
    data = imaging_utils.filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
# Convert data to percent dF/F0.
data['activity'] = data['activity'] * 100


# Plot for psth all cells.
# ------------------------

fig, axes = plt.subplots(2, figsize=(3, 8), sharey=True)

# Whisker stim.
sns.lineplot(data=data.loc[data.reward_group=='R+'], x='time', y='activity',
            errorbar='ci', style='epoch', style_order=['late', 'early'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0])
sns.lineplot(data=data.loc[data.reward_group=='R-'], x='time', y='activity',
            errorbar='ci', style='epoch', style_order=['late', 'early'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1])
axes[0].axvline(0, color='#FF9600', linestyle='--')
axes[1].axvline(0, color='#FF9600', linestyle='--')

axes[0].set_title(f'All cells - early vs late trials {early_trials}-{late_trials}')
axes[0].set_ylabel('R+')
axes[1].set_ylabel('R-')
plt.ylim(-1, 12)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Plot for projectors.
# --------------------

cell_types = ['wM1', 'wS2']

fig, axes = plt.subplots(2,2, figsize=(6, 8), sharey=True)

for j, ct in enumerate(cell_types):
    d = data.loc[(data.reward_group=='R+') & (data.cell_type==ct)]
    sns.lineplot(data=d, x='time', y='activity', errorbar='ci', style='epoch', style_order=['late', 'early'],
                color=reward_palette[1], estimator='mean', legend=False, ax=axes[0, j])
    d = data.loc[(data.reward_group=='R-') & (data.cell_type==ct)]
    sns.lineplot(data=d, x='time', y='activity', errorbar='ci', style='epoch', style_order=['late', 'early'],
                color=reward_palette[0], estimator='mean', legend=False, ax=axes[1, j])
    axes[0, j].axvline(0, color='#FF9600', linestyle='--')
    axes[1, j].axvline(0, color='#FF9600', linestyle='--')

axes[0,0].set_title(f'wM1 - early vs late trials {early_trials}-{late_trials}')
axes[0,1].set_title(f'wS2 - early vs late trials {early_trials}-{late_trials}')
axes[0,0].set_ylabel('R+')
axes[1,0].set_ylabel('R-')
plt.ylim(-2, 20)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_earlyvslate_projectors_on_{variance}_{early_trials}_{late_trials}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Bar plot and stats.
# -------------------
fig, axes = plt.subplots(2, 3, figsize=(9, 8), sharey=True)

# All cells
sns.barplot(
    data=data.loc[data.reward_group == 'R+'],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[1], estimator='mean', ax=axes[0, 0]
)
sns.barplot(
    data=data.loc[data.reward_group == 'R-'],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[0], estimator='mean', ax=axes[1, 0]
)
axes[0, 0].set_title('All cells')
axes[0, 0].set_ylabel('R+')
axes[1, 0].set_ylabel('R-')

# wS2 projectors
sns.barplot(
    data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wS2')],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[1], estimator='mean', ax=axes[0, 1]
)
sns.barplot(
    data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wS2')],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[0], estimator='mean', ax=axes[1, 1]
)
axes[0, 1].set_title('wS2 projectors')

# wM1 projectors
sns.barplot(
    data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wM1')],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[1], estimator='mean', ax=axes[0, 2]
)
sns.barplot(
    data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wM1')],
    x='epoch', y='activity',
    errorbar='ci', order=['early', 'late'],
    color=reward_palette[0], estimator='mean', ax=axes[1, 2]
)
axes[0, 2].set_title('wM1 projectors')

for ax in axes.flat:
    ax.set_xlabel('')

plt.ylim(0, 8)
sns.despine()
plt.tight_layout()

# Save figure for all cells and projectors.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_all_wS2_wM1.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Stats.
# Prepare data for stats and export
barplot_data = data.groupby(['mouse_id', 'reward_group', 'cell_type', 'epoch'])['activity'].mean().reset_index()

stats_results = []

# 1. All cells (ignore cell_type)
for group in barplot_data['reward_group'].unique():
    d = barplot_data[barplot_data['reward_group'] == group]
    # Average across cell types for each mouse and epoch
    d_all = d.groupby(['mouse_id', 'epoch'])['activity'].mean().reset_index()
    early = d_all[d_all['epoch'] == 'early'].set_index('mouse_id')['activity']
    late = d_all[d_all['epoch'] == 'late'].set_index('mouse_id')['activity']
    common = early.index.intersection(late.index)
    if len(common) > 0:
        stat, p = wilcoxon(early.loc[common], late.loc[common])
    else:
        stat, p = np.nan, np.nan
    stats_results.append({'reward_group': group, 'cell_type': 'all', 'wilcoxon_stat': stat, 'p_value': p})

# 2. wS2 and 3. wM1
for cell_type in ['wS2', 'wM1']:
    for group in barplot_data['reward_group'].unique():
        d = barplot_data[(barplot_data['reward_group'] == group) & (barplot_data['cell_type'] == cell_type)]
        early = d[d['epoch'] == 'early'].set_index('mouse_id')['activity']
        late = d[d['epoch'] == 'late'].set_index('mouse_id')['activity']
        common = early.index.intersection(late.index)
        if len(common) > 0:
            stat, p = wilcoxon(early.loc[common], late.loc[common])
        else:
            stat, p = np.nan, np.nan
        stats_results.append({'reward_group': group, 'cell_type': cell_type, 'wilcoxon_stat': stat, 'p_value': p})

stats_df = pd.DataFrame(stats_results)

# Save barplot data and stats to CSV
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/'
output_dir = io.adjust_path_to_host(output_dir)
barplot_data.to_csv(os.path.join(output_dir, f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_data.csv'), index=False)
stats_df.to_csv(os.path.join(output_dir, f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_stats.csv'), index=False)
















# #############################################################################
# PSTH during learning. Hit VS miss trials.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-0.5, 4)
win_bin = (0, 0.180)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [0]
days_str = ['0']
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh_hit = xarr_wh.sel(trial=xarr_wh['outcome_w']==1)
    xarr_wh_miss = xarr_wh.sel(trial=xarr_wh['outcome_w']==0)
    xarr_wh_hit = xarr_wh_hit.groupby('day').mean(dim='trial')
    xarr_wh_miss = xarr_wh_miss.groupby('day').mean(dim='trial')
    
    xarr_wh_hit.name = 'activity'
    xarr_wh_hit = xarr_wh_hit.to_dataframe().reset_index()
    xarr_wh_hit['mouse_id'] = mouse_id
    xarr_wh_hit['reward_group'] = reward_group
    xarr_wh_hit['stim'] = 'whisker'
    xarr_wh_hit['outcome'] = 'hit'
    psth.append(xarr_wh_hit)
    
    xarr_wh_miss.name = 'activity'
    xarr_wh_miss = xarr_wh_miss.to_dataframe().reset_index()
    xarr_wh_miss['mouse_id'] = mouse_id
    xarr_wh_miss['reward_group'] = reward_group
    xarr_wh_miss['stim'] = 'whisker'
    xarr_wh_miss['outcome'] = 'miss'
    psth.append(xarr_wh_miss)
   
    # Select no stim trials.
    xarr_nt = xarr.sel(trial=xarr['no_stim']==1)
    
    # Check if that nouse has FA trials.
    if xarr_nt.sel(trial=xarr_nt['outcome_c']==1).size == 0:
        # If no FA trials, we can skip this mouse.
        continue
    # Average trials per days.
    xarr_nt_hit = xarr_nt.sel(trial=xarr_nt['outcome_c']==1)
    xarr_nt_miss = xarr_nt.sel(trial=xarr_nt['outcome_c']==0)
    xarr_nt_hit = xarr_nt_hit.groupby('day').mean(dim='trial')
    xarr_nt_miss = xarr_nt_miss.groupby('day').mean(dim='trial')
    
    xarr_nt_hit.name = 'activity'
    xarr_nt_hit = xarr_nt_hit.to_dataframe().reset_index()
    xarr_nt_hit['mouse_id'] = mouse_id
    xarr_nt_hit['reward_group'] = reward_group
    xarr_nt_hit['stim'] = 'no_stim'
    xarr_nt_hit['outcome'] = 'hit'
    psth.append(xarr_nt_hit)
    
    xarr_nt_miss.name = 'activity'
    xarr_nt_miss = xarr_nt_miss.to_dataframe().reset_index()
    xarr_nt_miss['mouse_id'] = mouse_id
    xarr_nt_miss['reward_group'] = reward_group
    xarr_nt_miss['stim'] = 'no_stim'
    xarr_nt_miss['outcome'] = 'miss'
    psth.append(xarr_nt_miss)

psth = pd.concat(psth)


variance = 'cell'  # 'mice' or 'cells'
if variance == "mice":
    min_cells = 3
    data = imaging_utils.filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
# Convert data to percent dF/F0.
data['activity'] = data['activity'] * 100


# Plot for psth all cells.
# ------------------------

fig, axes = plt.subplots(2, 2, figsize=(6, 8), sharey=True)

# Whisker stim.
d = data.loc[(data.reward_group=='R+') & (data.stim=='whisker')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0,0])
d = data.loc[(data.reward_group=='R-') & (data.stim=='whisker')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1,0])

d = data.loc[(data.reward_group=='R+') & (data.stim=='no_stim')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0,1])
d = data.loc[(data.reward_group=='R-') & (data.stim=='no_stim')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1,1])

for ax in axes.flat:
    ax.axvline(0, color='#FF9600', linestyle='--')

axes[0,0].set_title('Whisker stim')
axes[0,1].set_title('No stim')
axes[0,0].set_ylabel('R+')
axes[1,0].set_ylabel('R-')
plt.ylim(-1, 12)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_hitvsmiss_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Plot for projectors.
# --------------------
cell_types = ['wM1', 'wS2']
stims = ['whisker', 'no_stim']
stim_titles = {'whisker': 'Whisker stim', 'no_stim': 'No stim'}

fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharey=True)

for j, ct in enumerate(cell_types):
    for k, stim in enumerate(stims):
        # R+
        d = data.loc[
            (data.reward_group == 'R+') &
            (data.cell_type == ct) &
            (data.stim == stim)
        ]
        sns.lineplot(
            data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0, j*2 + k]
        )
        axes[0, j*2 + k].axvline(0, color='#FF9600', linestyle='--')
        axes[0, j*2 + k].set_title(f"{ct} - {stim_titles[stim]}")
        # R-
        d = data.loc[
            (data.reward_group == 'R-') &
            (data.cell_type == ct) &
            (data.stim == stim)
        ]
        sns.lineplot(
            data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1, j*2 + k]
        )
        axes[1, j*2 + k].axvline(0, color='#FF9600', linestyle='--')

axes[0, 0].set_ylabel('R+')
axes[1, 0].set_ylabel('R-')
plt.ylim(-2, 20)
sns.despine()

# Save figure for projector neurons.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_hitvsmiss_projectors_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Bar plot and stats for hit vs miss, for both whisker and no_stim.
# -----------------------------------------------------------------
for stim in ['whisker', 'no_stim']:
    stim_title = 'Whisker stim' if stim == 'whisker' else 'No stim'
    fig, axes = plt.subplots(2, 3, figsize=(9, 8), sharey=True)

    # All cells
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 0]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 0]
    )
    axes[0, 0].set_title('All cells')
    axes[0, 0].set_ylabel('R+')
    axes[1, 0].set_ylabel('R-')

    # wS2 projectors
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wS2') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 1]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wS2') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 1]
    )
    axes[0, 1].set_title('wS2 projectors')

    # wM1 projectors
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wM1') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 2]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wM1') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 2]
    )
    axes[0, 2].set_title('wM1 projectors')

    for ax in axes.flat:
        ax.set_xlabel('')

    plt.suptitle(f'Hit vs Miss: {stim_title}')
    plt.ylim(0, 8)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure for all cells and projectors.
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
    output_dir = io.adjust_path_to_host(output_dir)
    svg_file = f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1.svg'
    plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
    plt.close(fig)

    # Stats.
    # Prepare data for stats and export
    barplot_data = data.loc[data.stim == stim].groupby(['mouse_id', 'reward_group', 'cell_type', 'outcome'])['activity'].mean().reset_index()

    stats_results = []

    # 1. All cells (ignore cell_type)
    for group in barplot_data['reward_group'].unique():
        d = barplot_data[barplot_data['reward_group'] == group]
        # Average across cell types for each mouse and outcome
        d_all = d.groupby(['mouse_id', 'outcome'])['activity'].mean().reset_index()
        hit = d_all[d_all['outcome'] == 'hit'].set_index('mouse_id')['activity']
        miss = d_all[d_all['outcome'] == 'miss'].set_index('mouse_id')['activity']
        common = hit.index.intersection(miss.index)
        if len(common) > 0:
            stat, p = wilcoxon(hit.loc[common], miss.loc[common])
        else:
            stat, p = np.nan, np.nan
        stats_results.append({'stim': stim, 'reward_group': group, 'cell_type': 'all', 'wilcoxon_stat': stat, 'p_value': p})

    # 2. wS2 and 3. wM1
    for cell_type in ['wS2', 'wM1']:
        for group in barplot_data['reward_group'].unique():
            d = barplot_data[(barplot_data['reward_group'] == group) & (barplot_data['cell_type'] == cell_type)]
            hit = d[d['outcome'] == 'hit'].set_index('mouse_id')['activity']
            miss = d[d['outcome'] == 'miss'].set_index('mouse_id')['activity']
            common = hit.index.intersection(miss.index)
            if len(common) > 0:
                stat, p = wilcoxon(hit.loc[common], miss.loc[common])
            else:
                stat, p = np.nan, np.nan
            stats_results.append({'stim': stim, 'reward_group': group, 'cell_type': cell_type, 'wilcoxon_stat': stat, 'p_value': p})

    stats_df = pd.DataFrame(stats_results)

    # Save barplot data and stats to CSV
    barplot_data.to_csv(os.path.join(output_dir, f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1_data.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1_stats.csv'), index=False)








# #############################################################################
# Day 0 modulation index.
# #############################################################################

# Look for cells active during the first whisk trials of the 
# session but stop firing at some point and cells initially silent that became
# active by the end.
# I expect the rewarded group to have less cells that stop firing which are
# consolidated by reward.


sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
substract_baseline = True
n_first = 15
n_last = 15

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

processed_folder = io.solve_common_paths('processed_data')  

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)

# Load day 0 LMI.
lmi_df = os.path.join(io.processed_dir, f'lmi_day0_results.csv')
lmi_df = pd.read_csv(lmi_df)
selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

psth = []
for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = imaging_utils.substract_baseline(xarr, 2, baseline_win)
    
    
    # Keep days of interest.
    xarr = xarr.sel(trial=xarr['day'].isin([0]))
    # Select whisker trials.
    xarr = xarr.sel(trial=xarr['whisker_stim']==1)
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win[0], win[1]))
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    
    xarr.name = 'psth'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    psth.append(xarr)
psth = pd.concat(psth)
    
# Plot proportion of pre vs post modulated cells for both reward groups during
# day 0.

# Compute and plot proportion across mice for both reward groups
lmi_df['reward_group'] = lmi_df['mouse_id'].map(lambda m: io.get_mouse_reward_group_from_db(io.db_path, m, db))
mouse_props = []
for mouse, group in lmi_df.groupby('mouse_id'):
    reward_group = group['reward_group'].iloc[0]
    n_cells = group['roi'].nunique()
    n_pos = group.loc[group['lmi_p'] >= 0.975, 'roi'].nunique()
    n_neg = group.loc[group['lmi_p'] <= 0.025, 'roi'].nunique()
    mouse_props.append({
        'mouse_id': mouse,
        'reward_group': reward_group,
        'prop_positive': n_pos / n_cells if n_cells > 0 else np.nan,
        'prop_negative': n_neg / n_cells if n_cells > 0 else np.nan
    })
mouse_props_df = pd.DataFrame(mouse_props)

plt.figure(figsize=(6,4), dpi=150)
sns.barplot(data=mouse_props_df.melt(id_vars=['mouse_id', 'reward_group'], 
                                     value_vars=['prop_positive', 'prop_negative'],
                                     var_name='modulation', value_name='proportion'),
            x='reward_group', y='proportion', hue='modulation', ci='sd')
sns.stripplot(data=mouse_props_df.melt(id_vars=['mouse_id', 'reward_group'], 
                                       value_vars=['prop_positive', 'prop_negative'],
                                       var_name='modulation', value_name='proportion'),
              x='reward_group', y='proportion', hue='modulation', 
              dodge=True, marker='o', alpha=0.7, linewidth=0.5, edgecolor='k')
plt.title('Proportion of significantly modulated ROIs per mouse')
plt.ylabel('Proportion')
plt.xlabel('Reward group')
plt.legend(title='Modulation', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.tight_layout()
plt.show()


