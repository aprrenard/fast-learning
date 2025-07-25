"""This script contains functions to analyze behavior data.
It assumes that a excel sheet contains sessions metadata
and that data is stored on the server with the usual folder structure.
This code will be refactored to read behavioral data from NWB files.
"""

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pymc as pm 
import scipy as sp

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from nwb_wrappers import nwb_reader_functions as nwb_read
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.cm as cm



# nwb_dir = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/NWB'
# session_list = ['MS039_20240305_085825.nwb',
#                 'MS041_20240305_123141.nwb',
#                 'MS065_20240421_150811.nwb',
#                 'MS067_20240422_152536.nwb',
#                 'MS069_20240422_154117.nwb',
#                 'MS128_20240926_111219.nwb',
#                 'MS129_20240926_112052.nwb',
#                 'MS130_20240925_112915.nwb',
#                 'MS131_20240926_120457.nwb',
#                 'MS135_20240925_133312.nwb']

# # session_list = ['MS061_20240421_144950.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS127_20240926_105210.nwb',
# #                 'MS132_20240925_115907.nwb',
# #                 'MS133_20240925_121916.nwb',
# #                 'MS134_20240925_114817.nwb']

# nwb_list = [os.path.join(nwb_dir, nwb) for nwb in session_list]
# reward_group = 'R+'

# table = []
# for nwb, session in zip(nwb_list, session_list):
#     df = nwb_read.get_trial_table(nwb)
#     df = df.reset_index()
#     df.rename(columns={'id': 'trial_id'}, inplace=True)
#     df = compute_performance(df, session, reward_group)
#     table.append(df)
# table = pd.concat(table)
# table['day'] = '0'
# table = table.astype({'mouse_id':str,
#                       'session_id':str,
#                       'day':str})

# table = table.loc[table.block_id<=17]
# sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)
# plt.figure(figsize=(15,6))
# plot_perf_across_blocks(table, reward_group, palette, nmax_trials=300, ax=None)
# sns.despine()


# Create and save data tables.
# ############################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Add mice with inactivation done after D0 learning.
mice_pharma_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['yes'],
                                    pharma_inactivation_type = ['execution'],
                                    )
mice_opto_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['execution'],
                                    )
mice_opto_learning = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['learning'],
                                    )
# Because some opto mice had both execution and learning inactivation.
mice_opto = list(set(mice_opto_execution) - set(mice_opto_learning))
# Combine and deduplicate mouse lists
mice_behavior = list(set(mice_behavior + mice_pharma_execution + mice_opto))
mice_behavior = sorted(mice_behavior)

len(mice_behavior)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',],
    day = ["-2", "-1", '0', '+1', '+2'],
    mouse_id = mice_behavior,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',  'two_p_exclude'],
    day = ["-2", "-1", '0', '+1', '+2'],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_imaging,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# Read behavior results.

particle_test_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'two_p_exclude'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = particle_test_mice,
    )

table_particle_test = make_behavior_table(nwb_list, session_list, db_path, cut_session=False, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
save_path = io.adjust_path_to_host(save_path)
table_particle_test.to_csv(save_path, index=False)


mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )

mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# # plot_single_session(table, 'GF311_19112020_160412')
# # table.loc[(table.reward_group=='R+') & (table.day==0), 'session_id'].unique()
# # sns.despine()    

#     # Plot lick trace for a trial.
#     # ------------------------

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    # day = ["-2", "-1", '0', '+1', '+2'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    subject_id = mice_imaging,
    )


#     bin_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/AR184/Training/AR184_20250319_175559/log_continuous.bin"
#     bin_file = io.adjust_path_to_host(bin_file)
#     # Read binary file using numpy
#     bin_data = np.fromfile(bin_file)
#     ttl = bin_data[2::6]
#     lick_trace = bin_data[1::6]

table = make_behavior_table(nwb_list, session_list, db_path= db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


# ############################################# 
# Day 0 for each mouse.
# #############################################

# table = pd.read_csv(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_muscimol.csv')
# table = table.loc[table.pharma_inactivation_type=='learning']

# session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
#     db_path, nwb_dir, experimenters=['AR'],
#     exclude_cols=['exclude'],
#     pharma_day = ['pre_-1', 'pre_-2', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3'],
#     subject_id = ['AR181', 'AR182', 'AR183', 'AR184',],
#     )

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_all_mice_5days.csv')
table = pd.read_csv(table_file)

session_list = table.loc[table.day==0].session_id.drop_duplicates().to_list()
pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_allmice.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        print(session_id)
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()

# Average performance across days.
# ################################

pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_muscimol.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()


# All five days for a single mouse.
# ################################

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

# mouse_id = 'AR180'
mouse_id = 'GF305'
# for  mouse_id in mice_imaging:
data = table.loc[table.mouse_id==mouse_id]
data  = data.loc[data.day.isin([-2, -1, 0, 1, 2])]
sessions = data.session_id.drop_duplicates().to_list()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': .8,'ytick.major.width': .8,})

data = data.loc[data.trial_id<=180]

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, session in enumerate(sessions):
    ax = axes[i]
    plot_single_session(
        data, session, ax=ax, palette=behavior_palette, 
         do_scatter=False, linewidth=1.5,
    )
    
# Save the figure.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior/exemples'
plt.savefig(os.path.join(output_dir, f'behavior_single_mouse_{mouse_id}.svg'), dpi=300)

sns.plotting_context("paper")


# ################################
# Average performance across days.
# ################################


# table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
# table.to_csv(save_path, index=False)

# Load table.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Remove spurious whisker trials coming mapping session.
table.loc[table.day.isin([-2, -1]), 'outcome_w'] = np.nan
table.loc[table.day.isin([-2, -1]), 'hr_w'] = np.nan

# Average performance.
table = table.groupby(['mouse_id','session_id','reward_group','day'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg(np.mean)
# Convert performance to percentage
table[['outcome_c', 'outcome_a', 'outcome_w']] = table[['outcome_c', 'outcome_a', 'outcome_w']] * 100

# Plot.
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': 1, 'ytick.major.width': 1, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
table['day'] = table['day'].astype(str)  # Convert 'day' column to string for categorical alignment

sns.lineplot(data=table, x='day', y='outcome_c', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[4:6], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_a', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[0:2], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_w', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[2:4], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)

sns.pointplot(data=table, x='day', y='outcome_c', estimator=np.mean, palette=behavior_palette[4:6], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_a', estimator=np.mean, palette=behavior_palette[0:2], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_w', estimator=np.mean, palette=behavior_palette[2:4], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)

plt.xlabel('Training days')
plt.ylabel('Lick probability (%)')
# plt.ylim([-0.2, 1.05])
plt.legend()
sns.despine(trim=True)

# Ensure tick thickness is set for SVG output
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(width=1)

# Save plot.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_across_days_imagingmice.svg'), dpi=300)
# Save the dataframe to CSV
table.to_csv(os.path.join(output_dir, 'performance_across_days_imagingmice_data.csv'), index=False)


# Histogram quantifying D0.
# -------------------------

# Select data of day 0
day_0_data = table[table['day'] == 0]
avg_performance = day_0_data.groupby(['mouse_id', 'reward_group'])['outcome_w'].mean().reset_index()

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
# Plot histogram to compare average performance across mice of the two reward groups
plt.figure(figsize=(4, 6))
sns.barplot(data=avg_performance, x='reward_group', y='outcome_w', palette=behavior_palette[2:4][::-1], width=0.3)
sns.stripplot(data=avg_performance, x='reward_group', y='outcome_w', color='grey', jitter=False, dodge=True)
plt.xlabel('Reward grouop')
plt.ylabel('Lick probability')
plt.ylim([0, 100])
sns.despine(trim=True)

# Test significance with Mann-Whitney U test
def test_significance(data, group_col, value_col, group1, group2):
    group1_data = data[data[group_col] == group1][value_col]
    group2_data = data[data[group_col] == group2][value_col]
    stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    return stat, p_value
# Perform the test
stat, p_value = test_significance(avg_performance, 'reward_group', 'outcome_w', 'R+', 'R-')

# Add stars to the plot to indicate significance
ax = plt.gca()
if p_value < 0.001:
    plt.text(0.5, 0.9, '***', ha='center', va='bottom', color='black', fontsize=12, transform=ax.transAxes)
elif p_value < 0.01:
    plt.text(0.5, 0.9, '**', ha='center', va='bottom', color='black', fontsize=12, transform=ax.transAxes)
elif p_value < 0.05:
    plt.text(0.5, 0.9, '*', ha='center', va='bottom', color='black', fontsize=12, transform=ax.transAxes)

# Save.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_D0_barplot_imagingmice.svg'), dpi=300)
avg_performance.to_csv(os.path.join(output_dir, 'performance_D0_barplot_imagingmice_data.csv'), index=False)
with open(os.path.join(output_dir, 'performance_D0_barplot_imagingmice_stats.csv'), 'w') as f:
    f.write(f'Statistic: {stat}, P-value: {p_value}')


# Plot first whisker hit for both reward group (sanity check).
# ------------------------------------------------------------

# Select data of day 0
day_0_data = table[(table['day'] == 0) & (table.whisker_stim == 1)]
f = lambda x: x.reset_index(drop=True).idxmax()+1
fh = day_0_data.groupby(['mouse_id', 'reward_group'], as_index=False)[['outcome_w']].agg(f)

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
                rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
plt.figure(figsize=(4, 6))
sns.barplot(data=fh, x='reward_group', y='outcome_w', palette=reward_palette[::-1], width=0.3)
sns.stripplot(data=fh, x='reward_group', y='outcome_w', color='grey', jitter=False, dodge=True, alpha=.4)
plt.ylabel('First hit trial')
sns.despine()
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'firsthit_D0.svg'), dpi=300)

# Save the first hit data to CSV
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
fh.to_csv(os.path.join(output_dir, 'firsthit_D0_data.csv'), index=False)

# Mann-Whitney U test for first hit trial between reward groups
stat, p_value = mannwhitneyu(
    fh[fh['reward_group'] == 'R+']['outcome_w'],
    fh[fh['reward_group'] == 'R-']['outcome_w'],
    alternative='two-sided'   
)
# Save stats to file
with open(os.path.join(output_dir, 'firsthit_D0_stats.csv'), 'w') as f:
    f.write(f'Mann-Whitney U Test: Statistic={stat}, P-value={p_value}')
 
# Particle test plot.
# -------------------

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
table_path = io.adjust_path_to_host(table_path)
table_particle_test = pd.read_csv(table_path)

df = table_particle_test.loc[table_particle_test.reward_group=='R+']
df['outcome_w'] = df['outcome_w'] * 100
df = df.groupby(['mouse_id', 'behavior_type'])['outcome_w'].mean().reset_index()

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': .8, 'xtick.minor.width': .8, 'ytick.major.width': .8, 'xtick.minor.width': .8, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
plt.figure(figsize=(4, 8))
sns.barplot(data=df, x='behavior_type', y='outcome_w', order=['whisker_on_1', 'whisker_off', 'whisker_on_2'], color='#1b9e77')
ax = plt.gca()
   
# Draw lines to connect each individual mouse across the three behavior types
for mouse_id in df.mouse_id.unique():
    a = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_on_1'), 'outcome_w'].to_numpy()[0]
    b = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_off'), 'outcome_w'].to_numpy()[0]
    plt.plot(['whisker_on_1', 'whisker_off'], [a,b], alpha=.8, color='grey', marker='', linewidth=1)
    a = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_off'), 'outcome_w'].to_numpy()[0]
    b = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_on_2'), 'outcome_w'].to_numpy()[0]
    plt.plot(['whisker_off', 'whisker_on_2'], [a,b], alpha=.8, color='grey', marker='', linewidth=1)
plt.xticks([0, 1, 2], ['ON', 'OFF', 'ON'])
plt.ylabel('Lick probability (%)')
plt.xlabel('Particle test')
sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'particle_test_imagingmice.svg'), dpi=300)


def test_significance(data, group_col, value_col, group1, group2):
    group1_data = data[data[group_col] == group1][value_col]
    group2_data = data[data[group_col] == group2][value_col]
    stat, p_value = wilcoxon(group1_data, group2_data, alternative='two-sided')
    return stat, p_value

# Perform the test for whisker_on_1 vs whisker_off
stat1, p_value1 = test_significance(df, 'behavior_type', 'outcome_w', 'whisker_on_1', 'whisker_off')
# Perform the test for whisker_off vs whisker_on_2
stat2, p_value2 = test_significance(df, 'behavior_type', 'outcome_w', 'whisker_off', 'whisker_on_2')

# Save the dataframe and stats to CSV files
save_path = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
save_path = io.adjust_path_to_host(save_path)
table_particle_test.to_csv(os.path.join(save_path, 'particle_test_imagningmice_data.csv'), index=False)
with open(os.path.join(save_path, 'particle_test_imagningmice_stats.csv'), 'w') as f:
    f.write(f'whisker_on_1 vs whisker_off: Statistic: {stat1}, P-value: {p_value1}\n')
    f.write(f'whisker_off vs whisker_on_2: Statistic: {stat2}, P-value: {p_value2}')
    # Test significance with Wilcoxon signed-rank test



# #####################################################################
# Performance over blocks during whisker learning sessions across mice.
# #####################################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
# table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv')
table = pd.read_csv(table_file)
table.mouse_id.unique().size


# Performance over blocks.
# ------------------------
# Adjust seaborn context for thicker lines
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})


data = table.copy()    
data['block_id'] = data.block_id + 1  # Start block index at 1 for plot.

# Performance over blocks
fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
plot_perf_across_blocks(data, "R-", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R-", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R-", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R-", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R-", 2, behavior_palette, nmax_trials=240, ax=axes[4])

plot_perf_across_blocks(data, "R+", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R+", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R+", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R+", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R+", 2, behavior_palette, nmax_trials=240, ax=axes[4])


# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_over_blocks_imagingmice.svg')
plt.savefig(output_file, format='svg', dpi=300)


# Performance over trials aligned to first hit.
# ---------------------------------------------

# Remove initial segment of trials where outcome_w == 0 for each session
# also the first whisker hit
def remove_initial_zero_outcome_w(table):
    def remove_initial_zeros(df):
        first_hit_index = df[df['outcome_w'] == 1].index.min()
        if pd.notna(first_hit_index):
            return df.loc[first_hit_index+1:]
        return df
    table = table.groupby('session_id', group_keys=False).apply(remove_initial_zeros)
    return table

block_size = 1
n_trials = 60
df = table.loc[(table.whisker_stim==1) & (table.day==0)]
df_fh = remove_initial_zero_outcome_w(df)
df_fh['trial_w'] = df_fh.groupby('session_id').cumcount()
# Filter to keep only the first n_trials
df_fh = df_fh.loc[df_fh.trial_w <= n_trials]

# Reindex trial_w.
df_fh['trial_w'] = df_fh.groupby('session_id').cumcount()
df_fh = df_fh[['mouse_id', 'reward_group', 'session_id','trial_w','outcome_w']]
df_fh['block_id'] = (df_fh['trial_w']) // block_size + 1
block_avg = df_fh.groupby(['session_id', 'reward_group','block_id'])['outcome_w'].mean().reset_index()


plt.figure(figsize=(8,5))
sns.lineplot(data=block_avg, x='block_id', y='outcome_w', 
                palette=reward_palette[::-1], legend=False, hue='reward_group',
                errorbar='ci', err_style='band')

# Perform Mann-Whitney U test for each block_id
p_values = []
for block_id in block_avg['block_id'].unique():
    group_R_plus = block_avg[(block_avg['block_id'] == block_id) & (block_avg['reward_group'] == 'R+')]['outcome_w']
    group_R_minus = block_avg[(block_avg['block_id'] == block_id) & (block_avg['reward_group'] == 'R-')]['outcome_w']
    stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
    p_values.append((block_id, p_value))

# Account for multiple comparisons using Benjamini-Hochberg (FDR) correction.

# Extract p-values for correction
block_ids, raw_pvals = zip(*p_values)
rejected, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
# Replace p_values with corrected p-values
p_values = list(zip(block_ids, corrected_pvals))
# Plot p-value squares on the graph using a colormap from white (p=0.05) to black (p=0)

# Create a colormap from white (p=0.05) to black (p=0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('pval_cmap', ['black', 'white'])
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

for block_id, p_value in p_values:
    # Clamp p_value to [0, 0.05] for colormap
    color = cmap(norm(min(p_value, 0.05)))
    plt.gca().add_patch(plt.Rectangle((block_id - 0.4, .95), 0.8, 0.03, color=color, edgecolor='none'))

plt.ylim([-0.1, 1])
plt.xlabel(f'Trials (average over blocks of {block_size} trials)')
plt.ylabel('Lick probability')
sns.despine()
# Change x tick labels to block_id * block_size
ax = plt.gca()
ax.set_xticklabels([str(int(label) * block_size) for label in ax.get_xticks()])

# Add number of mice in each reward group to the plot
n_mice = block_avg.groupby('reward_group')['session_id'].nunique()
for i, group in enumerate(['R+', 'R-']):
    count = n_mice.get(group, 0)
    plt.text(
        0.98, 1.15 - i*0.07,  # y offset for each group
        f"{group}: n={count}",
        ha='right', va='top', fontsize=12, color='black', transform=ax.transAxes
    )
    
# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, f'performance_D0_over_trials_imagingmice_blocksize_{block_size}.svg')
plt.savefig(output_file, format='svg', dpi=300)
# Save the block average data to CSV
block_avg.to_csv(os.path.join(output_dir, f'performance_D0_over_trials_imagingmice_blocksize_{block_size}_data.csv'), index=False)
# Save the p-values to CSV
p_values_df = pd.DataFrame(p_values, columns=['block_id', 'p_value'])
p_values_df.to_csv(os.path.join(output_dir, f'performance_D0_over_trials_imagingmice_blocksize_{block_size}_stats.csv'), index=False)


# Same with fitted learning curves.
# ---------------------------------

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

n_trials = 60
df = table.loc[(table.whisker_stim==1) & (table.day==0)]
df_fh = remove_initial_zero_outcome_w(df)
df_fh = df_fh.loc[df_fh.trial_w <= n_trials]
df_fh['trial_w'] = df_fh.groupby('session_id').cumcount() + 1


plt.figure(figsize=(8,5))
sns.lineplot(data=df_fh, x='trial_w', y='learning_curve_w', 
                palette=reward_palette[::-1], legend=False, hue='reward_group',
                errorbar='ci', err_style='band')

# Perform Mann-Whitney U test for each block_id
p_values = []
for trial_w in df_fh['trial_w'].unique():
    group_R_plus = df_fh[(df_fh['trial_w'] == trial_w) & (df_fh['reward_group'] == 'R+')]['learning_curve_w']
    group_R_minus = df_fh[(df_fh['trial_w'] == trial_w) & (df_fh['reward_group'] == 'R-')]['learning_curve_w']
    stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
    p_values.append((trial_w, p_value))

# Account for multiple comparisons using Benjamini-Hochberg (FDR) correction.

# Extract p-values for correction
trials, raw_pvals = zip(*p_values)
rejected, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
# Replace p_values with corrected p-values
p_values = list(zip(trials, corrected_pvals))
# Plot p-value squares on the graph using a colormap from white (p=0.05) to black (p=0)

# Create a colormap from white (p=0.05) to black (p=0)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('pval_cmap', ['black', 'white'])
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

for trial, p_value in p_values:
    # Clamp p_value to [0, 0.05] for colormap
    color = cmap(norm(min(p_value, 0.05)))
    plt.gca().add_patch(plt.Rectangle((trial - 0.4, .95), 0.8, 0.03, color=color, edgecolor='none'))

plt.ylim([-0.1, 1])
plt.xlabel(f'Trials')
plt.ylabel('Lick probability')
sns.despine()
# Change x tick labels to block_id * block_size
ax = plt.gca()
ax.set_xticklabels([str(int(label) * block_size) for label in ax.get_xticks()])

# Add number of mice in each reward group to the plot
n_mice = block_avg.groupby('reward_group')['session_id'].nunique()
for i, group in enumerate(['R+', 'R-']):
    count = n_mice.get(group, 0)
    plt.text(
        0.98, 1.15 - i*0.07,  # y offset for each group
        f"{group}: n={count}",
        ha='right', va='top', fontsize=12, color='black', transform=ax.transAxes
    )
    
# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, f'performance_D0_over_trials_learningcurves.svg')
plt.savefig(output_file, format='svg', dpi=300)
# Save the block average data to CSV
block_avg.to_csv(os.path.join(output_dir, f'performance_D0_over_trials_learningcurves_data.csv'), index=False)
# Save the p-values to CSV
p_values_df = pd.DataFrame(p_values, columns=['block_id', 'p_value'])
p_values_df.to_csv(os.path.join(output_dir, f'performance_D0_over_trials_learningcurves_stats.csv'), index=False)



# ############################################################
# Muscimol inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_muscimol = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    pharma_inactivation_type = ['learning'],
    pharma_day = ["pre_-2", "pre_-1",
            "muscimol_1", "muscimol_2", "muscimol_3",
            "recovery_1", "recovery_2", "recovery_3"],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_muscimol,
    )

# table_muscimol_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
# save_path = io.adjust_path_to_host(save_path)
# table_muscimol_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'pharma_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3']

data = table.groupby(['mouse_id', 'session_id', 'pharma_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['pharma_day'] = pd.Categorical(data['pharma_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'pharma_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', 'M 1', 'M 2', 'M 3', 'R 1', 'R 2', 'R 3'])
    ax.set_xlabel('Muscimol inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'muscimol_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'muscimol_learning_data.csv'), index=False)


# Bar plot and stats on Day 0.
# ----------------------------

# Filter data for Day 0 (muscimol_1) and group by area
day_0_data = data[data['pharma_day'] == 'muscimol_1']


# Plot bar plot for whisker performance (outcome_w) with reduced bar width
plt.figure(figsize=(4, 6))
sns.barplot(data=day_0_data, x='area', y='outcome_w', color=stim_palette[1], width=0.3)
sns.stripplot(data=day_0_data, x='area', y='outcome_w', color='black', jitter=False, dodge=True, alpha=0.6)
plt.xlabel('Inactivation Area')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
sns.despine()

# Perform Mann-Whitney U test to compare wS1 and fpS1 groups
stat, p_value = mannwhitneyu(
    day_0_data[day_0_data['area'] == 'wS1']['outcome_w'],
    day_0_data[day_0_data['area'] == 'fpS1']['outcome_w'],
    alternative='two-sided'
)

# Add significance stars and p-value to the plot
ax = plt.gca()
if p_value < 0.001:
    plt.text(0.5, 0.9, '***', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
elif p_value < 0.01:
    plt.text(0.5, 0.9, '**', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
elif p_value < 0.05:
    plt.text(0.5, 0.9, '*', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
# Add p-value text
plt.text(0.5, 0.85, f'p = {p_value:.3g}', ha='center', va='bottom', color='black', fontsize=12, transform=ax.transAxes)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'muscimol_learning_day0.svg'), format='svg', dpi=300)
day_0_data.to_csv(os.path.join(output_dir, 'muscimol_learning_day0_data.csv'), index=False)
with open(os.path.join(output_dir, 'muscimol_learning_day0_stats.csv'), 'w') as f:
    f.write(f'Mann-Whitney U Test: Statistic={stat}, P-value={p_value}')


# ###################
# Reaction time plot.
# ###################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

experimenters = ['AR', 'GF', 'MI']
mice_imaging = io.select_mice_from_db(db_path, nwb_dir,
                                    experimenters = experimenters,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Compute reaction time.
table['reaction_time'] = table['lick_time'] - table['stim_onset']
table.loc[table.reaction_time>=1.25] = np.nan

table = table.loc[table.mouse_id.isin(mice_groups['gradual_day0'])]

# Plot reaction time histogram together with scatter
# plot of reaction time for whisker trials across
# the three days with trial_w on the x-axis and reaction_time on the y-axis.
# --------------------------------------------------------------------------

binwidth = 0.05  # 50 ms bin width
nwhiskertrials = 60

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
data = table.loc[(table.reward_group=='R+') & (table.auditory_stim==1)] 
for day in [0, 1, 2]:
    day_data = data[data['day'] == day]
    sns.histplot(day_data, x='reaction_time',
                binwidth=binwidth, ax=axes[day], stat='probability', color='steelblue', alpha=0.7)
data = table.loc[(table.reward_group=='R+') & (table.whisker_stim==1)] 
for day in [0, 1, 2]:
    day_data = data[data['day'] == day]
    sns.histplot(day_data, x='reaction_time',
                binwidth=binwidth, ax=axes[day], stat='probability', color='#1b9e77', alpha=0.7)
# Plot reaction time for whisker trials as scatter, color-coded by trial_w (order in session)
for day in [0, 1, 2]:
    day_data = table[(table['reward_group'] == 'R+') & (table['whisker_stim'] == 1) & (table['day'] == day) & (table.trial_w <= nwhiskertrials)] 
    if not day_data.empty:
        cmap = cm.get_cmap('coolwarm', nwhiskertrials)
        norm = Normalize(vmin=day_data['trial_w'].min(), vmax=day_data['trial_w'].max())
        colors = cmap(norm(day_data['trial_w']))
        axes[day].scatter(day_data['reaction_time'], np.random.normal(0.4, 0.03, size=len(day_data)), 
                            c=colors, s=20, alpha=0.7, label='Whisker trials')
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[day], orientation='vertical', pad=0.01)
        cbar.set_label('Whisker trial order (trial_w)')
        axes[day].set_title(f'Day {day}')
# Save figure.
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, f'reaction_time_histogram_{nwhiskertrials}.svg'), format='svg', dpi=300)


# Plot reaction time to whisker trials across the three days with trial_w on the x-axis and reaction_time on the y-axis.
mwhiskertrials = 40
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, day in enumerate([0, 1, 2]):
    day_data = table[(table['reward_group'] == 'R+') & (table['whisker_stim'] == 1)
                        & (table['day'] == day) & (table.trial_w <= nwhiskertrials)
                        & (table.lick_flag == 1)]
    day_data = day_data.assign(trial_whit_id=day_data.groupby('session_id').cumcount() + 1)
    sns.pointplot(
        data=day_data,
        x='trial_whit_id',
        y='reaction_time',
        ax=axes[i],
        dodge=True,
        errorbar='ci',
        markers='o',
        linestyles='-',
    )
    axes[i].set_title(f'Day {day}')
sns.despine()
# Save figure.
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, f'reaction_time_pointplot.svg'), format='svg', dpi=300)




# ############################################################
# Opto inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'opto_exclude'],
                                    opto_inactivation_type = ['learning'],
                                    optogenetic = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'opto_exclude'],
    opto_inactivation_type = ['learning'],
    opto_day = ["pre_-2", "pre_-1",
            "opto", "recovery_1",],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_opto,
    )

table_opto_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
save_path = io.adjust_path_to_host(save_path)
table_opto_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'opto_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']

data = table.groupby(['mouse_id', 'session_id', 'opto_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['opto_day'] = pd.Categorical(data['opto_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'opto_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
                rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', '0', '+1'])
    ax.set_xlabel('Optogenetic inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'opto_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'opto_learning_data.csv'), index=False)


# Bar plot and stats on Day 0.
# ----------------------------

# Filter data for Day 0 (opto) and group by area
day_0_data = data[data['opto_day'] == 'opto']

# Plot bar plot for whisker performance (outcome_w) with reduced bar width
plt.figure(figsize=(4, 6))
sns.barplot(data=day_0_data, x='area', y='outcome_w', color=stim_palette[1], width=0.3, order=['wS1', 'fpS1'])
sns.stripplot(data=day_0_data, x='area', y='outcome_w', color='black', jitter=False, dodge=True, alpha=0.6, order=['wS1', 'fpS1'])
plt.xlabel('Inactivation Area')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
sns.despine()

# Perform Mann-Whitney U test to compare wS1 and fpS1 groups
stat, p_value = mannwhitneyu(
    day_0_data[day_0_data['area'] == 'wS1']['outcome_w'],
    day_0_data[day_0_data['area'] == 'fpS1']['outcome_w'],
    alternative='two-sided'
)

# Add significance stars and p-value to the plot
ax = plt.gca()
if p_value < 0.001:
    plt.text(0.5, 0.9, '***', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
elif p_value < 0.01:
    plt.text(0.5, 0.9, '**', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
elif p_value < 0.05:
    plt.text(0.5, 0.9, '*', ha='center', va='bottom', color='black', fontsize=20, transform=ax.transAxes)
# Add p-value text
plt.text(0.5, 0.85, f'p = {p_value:.3g}', ha='center', va='bottom', color='black', fontsize=12, transform=ax.transAxes)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'opto_learning_day0.svg'), format='svg', dpi=300)
day_0_data.to_csv(os.path.join(output_dir, 'opto_learning_day0_data.csv'), index=False)
with open(os.path.join(output_dir, 'opto_learning_day0_stats.csv'), 'w') as f:
    f.write(f'Mann-Whitney U Test: Statistic={stat}, P-value={p_value}')


# ##############################################
# Fit learning curves and define learning trial.
# ##############################################


# Load behavior results.
# ----------------------

db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

experimenters = ['AR', 'GF', 'MI']
mice_imaging = io.select_mice_from_db(db_path, nwb_dir,
                                    experimenters = experimenters,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Fit learning curves and define learning trial.
table = compute_learning_curves(table)
table = compute_learning_trial(table, n_consecutive_trials=10)

# Save updated table.
save_path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table.to_csv(save_path, index=False)

table = pd.read_csv(save_path)



# # Convert learning curve columns to float type to avoid dtype issues
# learning_curve_cols = [
#     'learning_curve_w', 'learning_curve_w_ci_low', 'learning_curve_w_ci_high',
#     'learning_curve_a', 'learning_curve_a_ci_low', 'learning_curve_a_ci_high',
#     'learning_curve_ns', 'learning_curve_ns_ci_low', 'learning_curve_ns_ci_high'
# ]
# for col in learning_curve_cols:
#     if col in table.columns:
#         table[col] = pd.to_numeric(table[col], errors='coerce')

# Day 0 learning curves pdf
pdf_path = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/behavior/learning_curves_day0_smooth.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)
session_list = table.session_id.unique()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)


with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        
        data = table.loc[table.session_id==session_id]
        if data.day.iloc[0] != 0:
            continue
        reward_group = data.reward_group.values[0]
        if reward_group == 'R-':
            color = reward_palette[0]
        else:
            color = reward_palette[1]
        
        d = data.loc[data.whisker_stim==1].reset_index(drop=True)
        # Smooth the learning curves for plotting

        # Apply Gaussian smoothing (sigma can be adjusted for more/less smoothing)
        sigma = 1

        smoothed_curve_w = gaussian_filter1d(d.learning_curve_w.values.astype(float), sigma=sigma)
        smoothed_ci_low = gaussian_filter1d(d.learning_curve_w_ci_low.values.astype(float), sigma=sigma)
        smoothed_ci_high = gaussian_filter1d(d.learning_curve_w_ci_high.values.astype(float), sigma=sigma)
        smoothed_chance = gaussian_filter1d(d.learning_curve_chance.astype(float), sigma=sigma)
        
        # smoothed_curve_w = d.learning_curve_w.values.astype(float)
        # smoothed_ci_low = d.learning_curve_w_ci_low.values.astype(float)
        # smoothed_ci_high = d.learning_curve_w_ci_high.values.astype(float)
        # smoothed_chance = d.learning_curve_chance.astype(float)
        
        

        # plt.plot(d.stim_onset.values, d.learning_curve_w.values, label='Whisker', color=color)
        plt.plot(d.trial_w, smoothed_curve_w, label='Whisker', color=color)
        plt.fill_between(d.trial_w, smoothed_ci_low, smoothed_ci_high, color=color, alpha=0.2)
        sns.lineplot(data=d, x='trial_w', y='hr_w', color=color,legend=False, linestyle='--')
        # plt.plot(d.stim_onset.values, d.learning_curve_ns.values, label='No_stim', color=stim_palette[2])
        plt.plot(d.trial_w, smoothed_chance, label='No_stim', color=stim_palette[2])
        # plt.fill_between(d.stim_onset.values, d.learning_curve_ns_ci_low.values, d.learning_curve_ns_ci_high.values, color=stim_palette[2], alpha=0.2)
        # sns.lineplot(data=d, x='stim_onset', y='hr_c', color=stim_palette[2],legend=False, linestyle='--')
        # Add vertical line indicating learning trial.
        learning_trial = data.learning_trial.values[0]
        if not pd.isna(learning_trial):
            plt.axvline(x=learning_trial, color='black', linestyle='--', label='Learning trial')
        plt.title(f'Session {session_id} - {reward_group}')
        plt.ylim([0, 1])
        plt.xlabel('Whisker trial')
        plt.ylabel('Lick probability')
        sns.despine()
        
        pdf.savefig()
        plt.close()
