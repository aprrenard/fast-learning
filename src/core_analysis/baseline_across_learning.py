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
from src.core_analysis.behavior import compute_performance, plot_single_session
from src.utils.utils_plot import *
from statannotations.Annotator import Annotator


def filter_data_by_cell_count(data, min_cells):
    """
    Filters the data to exclude entries where the number of distinct ROIs of a specific type
    for a mouse is below a given threshold.

    Parameters:
    - data (pd.DataFrame): The data to filter, containing columns 'mouse_id', 'cell_type', 'roi', etc.
    - min_cells (int): Minimum number of distinct ROIs required to keep the data.

    Returns:
    - pd.DataFrame: Filtered data.
    """
    # Count distinct ROIs per mouse and cell type
    roi_counts = data.groupby(['mouse_id', 'cell_type'])['roi'].nunique().reset_index()
    roi_counts = roi_counts.rename(columns={'roi': 'roi_count'})

    # Merge ROI counts back into the data
    data = data.merge(roi_counts, on=['mouse_id', 'cell_type'])

    # Filter out entries where the ROI count is below the threshold
    data = data[data['roi_count'] >= min_cells]

    # Drop the auxiliary 'roi_count' column
    data = data.drop(columns=['roi_count'])

    return data


# #############################################################################
# Count the number of cells and proportion of cell types.
# #############################################################################

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['GF', 'MI', 'AR'])

# Get roi and cell type for each mouse.
dataset = []
for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(io.processed_dir, 'mice')
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
pdf_file = f'cell_counts.pdf'  # TODO set correct path

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

# Print counts.
print(cell_count.groupby(['reward_group', 'cell_type'])['roi'].sum().reset_index())
print(cell_count.groupby(['reward_group', 'cell_type'])['roi'].mean().reset_index())
print(prop_ct.groupby(['reward_group', 'cell_type'])['prop'].mean().reset_index())

# Save dataframe.
cell_count.to_csv(os.path.join(save_dir, 'cell_counts.csv'), index=False)
prop_ct.to_csv(os.path.join(save_dir, 'cell_proportions.csv'), index=False)

# Count mice.
print(cell_count[['mouse_id', 'reward_group']].drop_duplicates().groupby('reward_group').count().reset_index())
print(prop_ct[['mouse_id', 'reward_group']].drop_duplicates().groupby('reward_group').count().reset_index())

# Read counts from file.
cell_count = io.adjust_path_to_host('/mnt/lsens-analysis/Anthony_Renard/analysis_output/counts/cell_counts.csv')
cell_count = pd.read_csv(cell_count)
cell_count
mice_AR = [m for m in mice if m.startswith('AR')]
mice_GF = [m for m in mice if m.startswith('GF') or m.startswith('MI')]
cell_count.loc[cell_count.mouse_id.isin(mice_AR)].groupby(['reward_group', 'cell_type'])['roi'].sum()
cell_count.loc[cell_count.mouse_id.isin(mice_GF)].groupby(['reward_group', 'cell_type'])['roi'].sum()


# #############################################################################
# Proportion of responsive cells across days.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
tests = pd.read_csv(os.path.join(processed_folder, 'response_test_results_win_300ms.csv'))

_, _, mice, _ = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

for mouse in tests.mouse_id.unique():
    tests.loc[tests.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

# Select days.
tests = tests.loc[tests.day.isin([-2, -1, 0, 1, 2])]

tests['thr_5%_aud'] = tests['pval_aud'] <= 0.05
tests['thr_5%_wh'] = tests['pval_wh'] <= 0.05
tests['thr_5%_mapping'] = tests['pval_mapping'] <= 0.05
tests['thr_5%_both'] = (tests['pval_aud'] <= 0.05) & (tests['pval_wh'] <= 0.05)

prop = tests.groupby(['mouse_id', 'session_id', 'reward_group', 'day'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_both', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
prop_proj = tests.groupby([ 'mouse_id', 'session_id', 'reward_group', 'day', 'cell_type'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_both', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
# Convert to long format.
prop = prop.melt(id_vars=['mouse_id', 'session_id', 'reward_group', 'day'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')
prop_proj = prop_proj.melt(id_vars=['mouse_id', 'session_id', 'reward_group', 'day', 'cell_type'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')
prop['prop'] = prop['prop'] * 100
prop_proj['prop'] = prop_proj['prop'] * 100


# Cell proportions across days. All cells.
# ----------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
palette = sns.color_palette(['#a6cee3', '#1f78b4'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_aud'],
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[0])
plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_wh'],
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1])
plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_mapping'],
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[2])
plt.ylim(0, 30)
sns.despine()

# Save fig to svg.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/cell_proportions/'
svg_file = f'prop_responsive_cells_across_days.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Cell proportions across days. Projection neurons.
# -------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(12, 4), sharey=True)
palette = sns.color_palette(['#a6cee3', '#1f78b4'])
data = prop_proj.loc[(prop_proj.test=='thr_5%_aud') & (prop_proj.cell_type=='wS2')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[0,0])
plt.ylim(0, 30)
data = prop_proj.loc[(prop_proj.test=='thr_5%_aud') & (prop_proj.cell_type=='wM1')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1,0])
plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
data = prop_proj.loc[(prop_proj.test=='thr_5%_wh') & (prop_proj.cell_type=='wS2')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[0,1])
plt.ylim(0, 30)
data = prop_proj.loc[(prop_proj.test=='thr_5%_wh') & (prop_proj.cell_type=='wM1')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1,1])
plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
data = prop_proj.loc[(prop_proj.test=='thr_5%_mapping') & (prop_proj.cell_type=='wS2')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[0,2])
plt.ylim(0, 30)
data = prop_proj.loc[(prop_proj.test=='thr_5%_mapping') & (prop_proj.cell_type=='wM1')]
sns.barplot(data=data,
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1,2])
plt.ylim(0, 30)
sns.despine()

# Save fig to svg.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/cell_proportions/'
svg_file = f'prop_responsive_cells_projections_across_days.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# #############################################################################
# Proportion of LMI.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

_, _, mice, _ = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)

for mouse in lmi_df.mouse_id.unique():
    lmi_df.loc[lmi_df.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

lmi_df = lmi_df.loc[lmi_df.mouse_id.isin(mice)]

lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= 0.975
lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= 0.025

lmi_prop = lmi_df.groupby(['mouse_id', 'reward_group'])[['lmi_pos', 'lmi_neg']].apply(lambda x: x.sum() / x.count()).reset_index()
lmi_prop_ct = lmi_df.groupby(['mouse_id', 'reward_group', 'cell_type'])[['lmi_pos', 'lmi_neg']].apply(lambda x: x.sum() / x.count()).reset_index()

# Plot.
fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharey=True)
sns.barplot(data=lmi_prop, x='reward_group', order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[0], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[0].set_title('LMI Positive')
sns.barplot(data=lmi_prop, x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg',  ax=axes[1], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[1].set_title('LMI Negative')
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wS2'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[2], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[2].set_title('LMI Positive wS2')
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wS2'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg', ax=axes[3], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[3].set_title('LMI Negative wS2')
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wM1'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_pos', ax=axes[4], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[4].set_title('LMI Positive wM1')
sns.barplot(data=lmi_prop_ct.loc[lmi_prop_ct.cell_type=='wM1'], x='reward_group',  order=['R+', 'R-'], hue='reward_group', y='lmi_neg', ax=axes[5], palette=reward_palette, hue_order=['R-', 'R+'], legend=False)
axes[5].set_title('LMI Negative wM1')
sns.despine(trim=True)

# Stats.
# Perform Mann-Whitney U test for each LMI group.
results = []
groups = ['lmi_pos', 'lmi_neg']
cell_types = [None, 'wS2', 'wM1']

for group in groups:
    for cell_type in cell_types:
        if cell_type:
            data = lmi_prop_ct[lmi_prop_ct.cell_type == cell_type]
        else:
            data = lmi_prop

        r_plus = data[data.reward_group == 'R+'][group]
        r_minus = data[data.reward_group == 'R-'][group]

        stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
        results.append({
            'group': group,
            'cell_type': cell_type if cell_type else 'all',
            'stat': stat,
            'p_value': p_value
        })
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'prop_lmi_mannwhitney_results.csv'), index=False)

# Save figure and data.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/cell_proportions/'
svg_file = f'prop_lmi.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
lmi_prop.to_csv(os.path.join(output_dir, 'prop_lmi.csv'), index=False)
lmi_prop_ct.to_csv(os.path.join(output_dir, 'prop_lmi_ct.csv'), index=False)
 

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
win_sec = (-0.5, 1.5)  
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, +1, +2]
days_str = ['-2', '-1', '0', '+1', '+2']


_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
# mice = [m for m in mice if m not in ['AR163']]
mice_count = db[['subject_id', 'reward_group']].drop_duplicates()
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

variance = 'cells'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data = filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type',])['psth'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()
# data = data.loc[~data.mouse_id.isin(['AR131'])]

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

# Save fig to svg.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'psth_across_days_across_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Individual mice PSTH's.
# -----------------------

output_dir = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
pdf_file = f'psth_individual_mice_baseline.pdf'

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse_id in mice:
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

variance = 'cells'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data_plot = filter_data_by_cell_count(avg_resp, min_cells)
    data_plot = data_plot.groupby(['mouse_id', 'day', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
else:
    data_plot = avg_resp.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()
data_plot = data_plot.loc[~data_plot.mouse_id.isin(['AR131'])]

# Plot average response across days.
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True, sharey=True)

# All cells.
sns.pointplot(data=data_plot, x='day', y='average_response', hue='reward_group',
              palette=reward_palette, hue_order=['R-', 'R+'], ax=axes[0], estimator='mean', legend=False)
axes[0].set_title('All Cells')
axes[0].set_ylabel('Average response (dF/F0)')
axes[0].set_ylim(0, 10)

# wS2 cells.
sns.pointplot(data=data_plot[data_plot.cell_type == 'wS2'], x='day', y='average_response', hue='reward_group',
              palette=reward_palette, hue_order=['R-', 'R+'], ax=axes[1], estimator='mean', legend=False)
axes[1].set_title('wS2 Cells')
axes[1].set_ylabel('Average response (dF/F0)')

# wM1 cells.
sns.pointplot(data=data_plot[data_plot.cell_type == 'wM1'], x='day', y='average_response', hue='reward_group',
              palette=reward_palette, hue_order=['R-', 'R+'], ax=axes[2], estimator='mean', legend=False)
axes[2].set_title('wM1 Cells')
axes[2].set_ylabel('Average response (dF/F0)')
axes[2].set_xlabel('Days')
sns.despine()

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

# Save figure.
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'amplitude_response_across_days_across_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data_plot.to_csv(os.path.join(output_dir, f'amplitude_response_across_days_across_{variance}.csv'), index=False)
# Save stats.
stats.to_csv(os.path.join(output_dir, f'amplitude_response_across_days_across_{variance}_stats.csv'), index=False)


# Same plot comparing projection types inside each reward group across cells.
# --------------------------------------------------------------------------

variance = "mice"  # 'mice' or 'cells'

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

variance = 'cells'  # 'mice' or 'cells'
days_selected = [-2,-1, 0, 1,2]

# Select days of interest.
data_plot_avg = avg_resp[avg_resp['day'].isin(days_selected)]
data_plot_psth = psth[psth['day'].isin(days_selected)]

if variance == "mice":
    min_cells = 3
    data_plot_avg = filter_data_by_cell_count(data_plot_avg, min_cells)
    data_plot_psth = filter_data_by_cell_count(data_plot_psth, min_cells)
    data_plot_avg = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type'])['average_response'].agg('mean').reset_index()
    data_plot_psth = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type'])['psth'].agg('mean').reset_index()
else:
    data_plot_avg = data_plot_avg.groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type', 'roi'])['average_response'].agg('mean').reset_index()
    data_plot_psth = data_plot_psth.groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type', 'roi'])['psth'].agg('mean').reset_index()


# Create the figure with four subplots
fig, axes = plt.subplots(2, 6, figsize=(36, 10), sharex=False, sharey=False)

# Top-left: PSTH for rewarded mice
rewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R+')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 0])
axes[0, 0].set_title('PSTH (Rewarded Mice)')
axes[0, 0].set_ylabel('DF/F0 (%)')
axes[0, 0].axvline(0, color='orange', linestyle='--')

# Bottom-left: PSTH for non-rewarded mice
nonrewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R-')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 0])
axes[1, 0].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 0].set_ylabel('DF/F0 (%)')
axes[1, 0].axvline(0, color='orange', linestyle='--')

# Top-right: Response amplitude for rewarded mice
rewarded_avg = data_plot_avg[data_plot_avg['reward_group'] == 'R+']
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', color='#1b9e77', ax=axes[0, 1])
axes[0, 1].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 1].set_ylabel('Average Response (dF/F0)')

# Bottom-right: Response amplitude for non-rewarded mice
nonrewarded_avg = data_plot_avg[data_plot_avg['reward_group'] == 'R-']
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', color='#c959affe', ax=axes[1, 1])
axes[1, 1].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 1].set_ylabel('Average Response (dF/F0)')

# S2 PSTH
rewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R+') & (data_plot_psth['cell_type'] == 'wS2')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 2])
axes[0, 2].set_title('PSTH (Rewarded Mice)')
axes[0, 2].set_ylabel('DF/F0 (%)')
axes[0, 2].axvline(0, color='orange', linestyle='--')

nonrewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R-') & (data_plot_psth['cell_type'] == 'wS2')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 2])
axes[1, 2].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 2].set_ylabel('DF/F0 (%)')
axes[1, 2].axvline(0, color='orange', linestyle='--')

# S2 Response amplitude
rewarded_avg = data_plot_avg[(data_plot_avg['reward_group'] == 'R+') & (data_plot_avg['cell_type'] == 'wS2')]
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', color='#1b9e77', ax=axes[0, 3])
axes[0, 3].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 3].set_ylabel('Average Response (dF/F0)')

nonrewarded_avg = data_plot_avg[(data_plot_avg['reward_group'] == 'R-') & (data_plot_avg['cell_type'] == 'wS2')]
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', color='#c959affe', ax=axes[1, 3])
axes[1, 3].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 3].set_ylabel('Average Response (dF/F0)')

# M1 PSTH
rewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R+') & (data_plot_psth['cell_type'] == 'wM1')]
sns.lineplot(data=rewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#1b9e77']), ax=axes[0, 4])
axes[0, 4].set_title('PSTH (Rewarded Mice)')
axes[0, 4].set_ylabel('DF/F0 (%)')
axes[0, 4].axvline(0, color='orange', linestyle='--')

nonrewarded_data = data_plot_psth[(data_plot_psth['reward_group'] == 'R-') & (data_plot_psth['cell_type'] == 'wM1')]
sns.lineplot(data=nonrewarded_data, x='time', y='psth', hue='learning_period', palette=sns.color_palette(['#a3a3a3', '#c959affe']), ax=axes[1, 4])
axes[1, 4].set_title('PSTH (Non-Rewarded Mice)')
axes[1, 4].set_ylabel('DF/F0 (%)')
axes[1, 4].axvline(0, color='orange', linestyle='--')

# M1 Response amplitude
rewarded_avg = data_plot_avg[(data_plot_avg['reward_group'] == 'R+') & (data_plot_avg['cell_type'] == 'wM1')]
sns.pointplot(data=rewarded_avg, x='learning_period', y='average_response', color='#1b9e77', ax=axes[0, 5])
axes[0, 5].set_title('Response Amplitude (Rewarded Mice)')
axes[0, 5].set_ylabel('Average Response (dF/F0)')

nonrewarded_avg = data_plot_avg[(data_plot_avg['reward_group'] == 'R-') & (data_plot_avg['cell_type'] == 'wM1')]
sns.pointplot(data=nonrewarded_avg, x='learning_period', y='average_response', color='#c959affe', ax=axes[1, 5])
axes[1, 5].set_title('Response Amplitude (Non-Rewarded Mice)')
axes[1, 5].set_ylabel('Average Response (dF/F0)')


# Adjust layout
# plt.tight_layout()
sns.despine()

# Perform stats on response amplitude
results = []
for reward_group in ['R+', 'R-']:
    for cell_type in ['all', 'wS2', 'wM1']:
        if cell_type == 'all':
            data_stats = data_plot_avg[data_plot_avg['reward_group'] == reward_group]
        else:
            data_stats = data_plot_avg[(data_plot_avg['reward_group'] == reward_group) & (data_plot_avg['cell_type'] == cell_type)]
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
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/psth'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
stats_df.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_stats.csv'), index=False)
data_plot_avg.to_csv(os.path.join(output_dir, f'pre_post_psth_amplitude_{variance}_days_selected_{days_selected}_data.csv'), index=False)



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
n_map_trials = 45
substract_baseline = True
average_inside_days = True
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)
# excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
# mice = [m for m in mice if m not in excluded_mice]


mice = ['AR127']
# Load data.
# ----------

vectors_rew = []
vectors_nonrew = []
for mouse in mice:
    print(mouse)
    io.processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, io.processed_dir, file_name)
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
plt.imshow(cm, cmap='grey_r', vmax=np.nanmax(cm_nodiag))
cm = np.corrcoef(data_nonrew.values.T)
cm_nodiag = cm.copy()
np.fill_diagonal(cm_nodiag, np.nan)
plt.imshow(cm, cmap='grey_r', vmax=np.nanmax(cm_nodiag))


cm = np.corrcoef(data.sel(cell=(data.cell_type=='wS2')).values.T)
plt.imshow(cm, cmap='viridis', vmin=-.19, vmax=0.7)


# Illustrate pop vectors of AR127.

# Vectors during learning.
file_name = 'tensor_xarray_learning_data.nc'
xarray = imaging_utils.load_mouse_xarray(mouse, io.processed_dir, file_name)
xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

# Select days.
xarray = xarray.sel(trial=xarray['day'].isin([2]))
xarray = xarray.sel(trial=xarray['whisker_stim']==1)

xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')


# Plot
vectors_rew = xarray.values
vmax = np.percentile(vectors_rew, 98)
vmin = np.percentile(vectors_rew, 2)
edges = np.cumsum([n_map_trials for _ in range(5)])
f = plt.figure(figsize=(10, 6))
im = plt.imshow(vectors_rew, cmap='viridis', vmin=vmin, vmax=vmax)



for i in edges[:-1] - 0.5:
    plt.axvline(x=i, color='white', linestyle='-', lw=0.5)
plt.xticks(edges - 0.5, edges)









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
    io.processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, io.processed_dir, file_name)
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
    io.processed_dir = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, io.processed_dir, file_name)
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
