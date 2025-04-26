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


# #############################################################################
# 2. Gradual learning during Day 0.
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
# 3. Gradual learning during Day 0 realigned to "learning trial".
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



