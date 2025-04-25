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
tests = pd.read_csv(os.path.join(processed_folder, 'response_test_results_win_180ms.csv'))

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
# plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_wh'],
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1])
# plt.ylim(0, 30)

palette = sns.color_palette(['#d95f02', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_mapping'],
            x='day', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[2])
# plt.ylim(0, 30)
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
# Proportion of responsive cells all days pulled together.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
tests = pd.read_csv(os.path.join(processed_folder, 'response_test_results_alldaystogether_win_180ms.csv'))

_, _, mice, _ = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

for mouse in tests.mouse_id.unique():
    tests.loc[tests.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

tests['thr_5%_aud'] = tests['pval_aud'] <= 0.01
tests['thr_5%_wh'] = tests['pval_wh'] <= 0.01
tests['thr_5%_mapping'] = tests['pval_mapping'] <= 0.01

prop = tests.groupby(['mouse_id', 'reward_group'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
prop_proj = tests.groupby(['mouse_id', 'reward_group', 'cell_type'])[['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping']].apply(lambda x: x.sum() / x.count()).reset_index()
# Convert to long format.
prop = prop.melt(id_vars=['mouse_id', 'reward_group'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')
prop_proj = prop_proj.melt(id_vars=['mouse_id', 'reward_group', 'cell_type'], value_vars=['thr_5%_aud', 'thr_5%_wh', 'thr_5%_mapping'], var_name='test', value_name='prop')
prop['prop'] = prop['prop'] * 100
prop_proj['prop'] = prop_proj['prop'] * 100


# Cell proportions. All cells and projection types.
# -------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

palette = sns.color_palette(['#0ddddd', '#1f77b4'])

sns.barplot(data=prop.loc[prop.test=='thr_5%_aud'],
            x='reward_group', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[0])
# plt.ylim(0, 30)

palette = sns.color_palette(['#c959affe', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_wh'],
            x='reward_group', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[1])
# plt.ylim(0, 30)

palette = sns.color_palette(['#c959affe', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%_mapping'],
            x='reward_group', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes[2])
# plt.ylim(0, 30)
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
# Proportion of responsive cells based on ROC analysis.
# #############################################################################

processed_folder = io.solve_common_paths('processed_data')
tests = pd.read_csv(os.path.join(processed_folder, 'response_test_results_mapping_ROC.csv'))

_, _, mice, _ = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])

for mouse in tests.mouse_id.unique():
    tests.loc[tests.mouse_id==mouse, 'reward_group'] = io.get_mouse_reward_group_from_db(io.db_path, mouse)

tests['thr_5%'] = tests['roc_p'] <= 0.05

prop = tests.groupby(['mouse_id', 'reward_group'])[['thr_5%']].apply(lambda x: x.sum() / x.count()).reset_index()
prop_proj = tests.groupby(['mouse_id', 'reward_group', 'cell_type'])[['thr_5%']].apply(lambda x: x.sum() / x.count()).reset_index()
# Convert to long format.
prop = prop.melt(id_vars=['mouse_id', 'reward_group'], value_vars=['thr_5%'], var_name='test', value_name='prop')
prop_proj = prop_proj.melt(id_vars=['mouse_id', 'reward_group', 'cell_type'], value_vars=['thr_5%'], var_name='test', value_name='prop')
prop['prop'] = prop['prop'] * 100
prop_proj['prop'] = prop_proj['prop'] * 100


# Cell proportions. All cells and projection types.
# -------------------------------------------------

fig, axes = plt.subplots(1, 1, figsize=(12, 4), sharey=True)

palette = sns.color_palette(['#c959affe', '#1b9e77'])
sns.barplot(data=prop.loc[prop.test=='thr_5%'],
            x='reward_group', y='prop', hue='reward_group',
            palette=palette,
            hue_order=['R-', 'R+'],
            estimator=np.mean,
            ax=axes)
# plt.ylim(0, 30)
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
 
