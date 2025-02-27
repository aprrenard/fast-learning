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

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.behavior import compute_performance, plot_single_session
import warnings

# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
# %matplotlib inline

# Path to the directory containing the processed data.
processed_dir = io.solve_common_paths('processed_data')
nwb_dir = io.solve_common_paths('nwb')
db_path = io.solve_common_paths('db')

# Color palettes.
reward_palette = sns.color_palette(['#d51a1c', '#238443'])
cell_types_palette = sns.color_palette(['#a3a3a3', '#c959affe', '#3351ffff'])
s2_m1_palette = sns.color_palette(['#c959affe', '#3351ffff'])


# #############################################################################
# Count the number of cells and proportion of cell types.
# #############################################################################
_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['GF', 'MI', 'AR'])


# Get roi and cell type for each mouse.
dataset = []
for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(db_path, mouse_id)
    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(processed_dir, 'mice')
    data = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    
    rois = data.coords['roi'].values
    cts = data.coords['cell_type'].values
    df = pd.DataFrame(data={'roi': rois, 'cell_type': cts})
    df['mouse_id'] = mouse_id
    df['reward_group'] = reward_group
    dataset.append(df)
dataset = pd.concat(dataset)

# Counts.
cell_count = dataset.groupby(['mouse_id', 'reward_group', 'cell_type'])['roi'].count().reset_index()
temp = cell_count.groupby(['mouse_id', 'reward_group'])[['roi']].sum().reset_index()
temp['cell_type'] = 'all_cells'
cell_count = pd.concat([cell_count, temp])
cell_count = cell_count.loc[cell_count.cell_type.isin(['all_cells', 'wS2', 'wM1'])]

prop_ct = cell_count.loc[cell_count.cell_type.isin(['wS2', 'wM1'])]
prop_ct = prop_ct.merge(cell_count.loc[cell_count.cell_type == 'all_cells', ['mouse_id', 'reward_group', 'roi']], on=['mouse_id', 'reward_group'], suffixes=('', '_total'))
prop_ct['prop'] = prop_ct['roi'] / prop_ct['roi_total']

# Save figures to PDF.
save_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/counts'
save_dir = io.adjust_path_to_host(save_dir)
pdf_file = 'cell_counts_and_proportions.pdf'

with PdfPages(os.path.join(save_dir, pdf_file)) as pdf:
    fig = sns.catplot(data=cell_count,
                      x='reward_group', y='roi', hue='cell_type', kind='bar',
                      palette=cell_types_palette, hue_order=['all_cells', 'wS2', 'wM1'])
    fig.set_axis_labels('', 'Number of cells')
    pdf.savefig(fig.figure)
    plt.close(fig.figure)

    fig = sns.catplot(data=prop_ct,
                      x='reward_group', y='prop', hue='cell_type', kind='bar',
                      palette=s2_m1_palette, hue_order=['wS2', 'wM1'])
    fig.set_axis_labels('', 'Proportion of cells')
    pdf.savefig(fig.figure)
    plt.close(fig.figure)

# Print the results.
print(cell_count.groupby(['reward_group', 'cell_type'])['roi'].mean().reset_index())
print(prop_ct.groupby(['reward_group', 'cell_type'])['prop'].mean().reset_index())

# Save dataframe.
cell_count.to_csv(os.path.join(save_dir, 'cell_counts.csv'), index=False)
prop_ct.to_csv(os.path.join(save_dir, 'cell_proportions.csv'), index=False)







# #############################################################################
# 1. PSTH's.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (0.8, 3)  
win = (int(win_sec[0] * sampling_rate), int(win_sec[1] * sampling_rate))
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = ['-2', '-1', '0', '+1', '+2']
cell_type = None
variance_across = 'cells'

_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['GF', 'MI', 'AR'])
# mice = [m for m in mice if m not in ['AR163']]
print(mice)
len(mice)


# Load the data.
# --------------

psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(processed_dir, 'mice')
    data = imaging_utils.load_mouse_xarray(mouse_id, folder, file_name)
    data = imaging_utils.substract_baseline(data, 2, baseline_win)
        
    avg_response = data.groupby('day').mean(dim='trial')
    avg_response.name = 'psth'
    avg_response = avg_response.to_dataframe().reset_index()
    avg_response['mouse_id'] = mouse_id
    avg_response['reward_group'] = reward_group
    
    psth.append(avg_response)
psth = pd.concat(psth)


# Grand average psth's for all cells and projection neurons.
# ----------------------------------------------------------

data = psth.loc[(psth.time>=-0.5) & (psth.time<1.5)]
data = data.loc[data['day'].isin([-2, -1, 0, 1, 2])]
data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type'])['psth'].agg('mean').reset_index()
# data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()

# # GF305 has baseline artefact on day -1 at auditory trials.
# data = data.loc[~data.mouse_id.isin(['GF305'])]

# data = data.loc[data['cell_type']=='wM1']
# data = data.loc[~data.mouse_id.isin(['GF305', 'GF306', 'GF307'])]
fig = sns.relplot(data=data, x='time', y='psth', errorbar='ci', col='day',
            kind='line', hue='reward_group',
            hue_order=['R-','R+'], palette=reward_palette,
            height=3, aspect=0.8)
for ax in fig.axes.flatten():
    ax.axvline(0, color='#FF9600', linestyle='--')
    ax.set_title('')    

fig = sns.relplot(data=data, x='time', y='psth', errorbar='se', col='day', row='cell_type',
            kind='line', hue='reward_group',
            hue_order=['R-','R+'], palette=reward_palette, row_order=['wS2', 'wM1',],
            height=3, aspect=0.8)
for ax in fig.axes.flatten():
    ax.axvline(0, color='#FF9600', linestyle='--')
    ax.set_title('')


# Individual mice PSTH's.

output_dir = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/sensory_plasticity/psth'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = f'psth_individual_mice_mapping.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse_id in mice:
        # Plot.
        data = psth.loc[psth['day'].isin([-2, -1, 0, 1, 2])
                      & (psth['mouse_id'] == mouse_id)]

        sns.relplot(data=data, x='time', y='psth', errorbar='se', col='day', row='cell_type',
                    kind='line', hue='reward_group',
                    hue_order=['R-','R+'], palette=reward_palette,)
        plt.suptitle(mouse_id)
        pdf.savefig(dpi=300)
        plt.close()


# #############################################################################
# Proportion of responsive cells across days.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
tests = pd.read_csv(os.path.join(processed_folder, 'response_test_results_win_180ms.csv'))

_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR'])

for mouse in tests.mouse_id.unique():
    tests.loc[tests.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(db_path, mouse)

tests['thr_5%_aud'] = tests['pval_aud'] <= 0.05
tests['thr_5%_wh'] = tests['pval_wh'] <= 0.05
tests['thr_5%_mapping'] = tests['pval_mapping'] <= 0.05
tests['thr_5%_both'] = (tests['pval_aud'] <= 0.05) & (tests['pval_wh'] <= 0.05)


prop = tests.groupby(['mouse_id', 'session_id', 'reward_group', 'day'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_both', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
prop_proj = tests.groupby([ 'mouse_id', 'session_id', 'reward_group', 'day', 'cell_type'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_both', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
# Convert to long format.
prop = prop.melt(id_vars=['mouse_id', 'session_id', 'reward_group', 'day'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')
prop_proj = prop_proj.melt(id_vars=['mouse_id', 'session_id', 'reward_group', 'day', 'cell_type'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')

sns.barplot(data=prop.loc[prop.reward_group=='R-'], x='day', y='prop', hue='test', palette='deep')
plt.ylim(0, 0.4)
sns.barplot(data=prop_proj.loc[(prop_proj.reward_group=='R-') & ((prop_proj.cell_type=='wS2'))], x='day', y='prop', hue='test', palette='deep')
plt.ylim(0, 0.4)

prop.loc[prop.session_id=='AR127_20240225_142858']
tests.loc[tests.session_id=='AR127_20240225_142858']

sns.barplot(data=prop, x='day', y='thr_5%_wh',  palette='deep')

sns.barplot(data=prop.loc[prop.test=='thr_5%_mapping'], hue='reward_group', x='day', y='prop', palette='deep')

sns.barplot(data=prop_proj.loc[(prop_proj.cell_type=='wM1') & (prop_proj.test=='thr_5%_mapping')], hue='reward_group', x='day', y='prop', palette='deep')


# #############################################################################
# Proportion of LMI.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes',)

for mouse in lmi_df.mouse_id.unique():
    lmi_df.loc[lmi_df.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(db_path, mouse)

lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= 0.975
lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= 0.025

lmi_prop = lmi_df.groupby(['mouse_id', 'reward_group'])[['lmi_pos', 'lmi_neg']].apply(lambda x: x.sum() / x.count()).reset_index()
lmi_prop_ct = lmi_df.groupby(['mouse_id', 'reward_group', 'cell_type'])[['lmi_pos', 'lmi_neg']].apply(lambda x: x.sum() / x.count()).reset_index()

fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharey=True)
sns.barplot(data=lmi_prop, x='reward_group', order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[0], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.barplot(data=lmi_prop, x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg',  ax=axes[1], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wS2'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[2], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wS2'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg', ax=axes[3], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wM1'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[4], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wM1'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg', ax=axes[5], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
sns.despine(trim=True)






# # #############################################################################
# # Comparing xarray datasets with previous tensors.
# # #############################################################################

# processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
# mouse_id = 'AR180'
# session_id = 'AR180_20241217_160355'

# arr, mdata = imaging_utils.load_session_2p_imaging(mouse_id,
#                                                     session_id,
#                                                     processed_dir
#                                                     )
# # arr = imaging_utils.substract_baseline(arr, 3, ())
# arr = imaging_utils.extract_trials(arr, mdata, 'UM', n_trials=None)
# arr.shape

# # Load the xarray dataset.
# file_name = 'tensor_xarray_mapping_data.nc'
# xarray = imaging_utils.load_mouse_xarray(mouse_id, processed_dir, file_name)

# d = xarray.sel(trial=xarray['day'] == 2)



# #############################################################################
# Correlation matrices during mapping across days.
# #############################################################################

# Parameters.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = ['-2', '-1', '0', '+1', '+2']
substract_baseline = True
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes',)
print(mice)
excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
mice = [m for m in mice if m not in excluded_mice]


# Load the data.
# --------------
processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
file_name = 'tensor_xarray_mapping_data.nc'
xarray = imaging_utils.load_mouse_xarray('GF306', processed_dir, file_name)
xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
# d = xarray.sel(time=slice(0, 0.4)).mean(dim='time')
d = np.mean(xarray[:,:,30:39], axis=2)
d.shape
cm = np.corrcoef(d.values.T)
plt.imshow(cm, cmap='viridis', vmin=-.19, vmax=0.5)