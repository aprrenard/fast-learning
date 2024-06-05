"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


# PSTH's day -1 VS day +1 full population.
# ########################################

read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_GF.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_GF.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

# Substract baseline.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)


excluded = ['GF264', 'GF278', 'GF208', 'GF340']



# Convert psth to pandas.
# #######################

def convert_psth_np_to_pd(traces_rew, traces_non_rew):

    nmice, ndays, ntypes = traces_rew.shape[:3]
    ncells = traces_rew.shape[2] * traces_rew.shape[3]
    df_psth_rew = []
    days = ['D-3','D-2', 'D-1', 'D0', 'D+1', 'D+2']
    for imouse in range(nmice):
        for isession in range(ndays):
            cell_count = 0
            for itype in range(ntypes):
                ncells = np.sum(~np.isnan(traces_rew[imouse, isession, itype, :, 0, 0]))
                for icell in range(ncells):
                    response = np.nanmean(traces_rew[imouse, isession, itype, icell], axis=0)
                    df = pd.DataFrame([], columns = ['time', 'activity', 'roi', 'cell_type', 'session_id', 'mouse_id'])
                    df['time'] = np.arange(traces_rew.shape[5]) / 30
                    df['activity'] = response
                    df['roi'] = cell_count
                    df['cell_type'] = metadata_rew['cell_types'][itype]
                    df['mouse_id'] = metadata_rew['mice'][imouse]
                    df['session_id'] = days[isession]
                    df_psth_rew.append(df)
                    cell_count += 1

    df_psth_rew = pd.concat(df_psth_rew, axis=0, ignore_index=True)
    df_psth_rew['reward_group'] = 'R+'

    nmice, ndays, ntypes = traces_non_rew.shape[:3]
    ncells = traces_non_rew.shape[2] * traces_non_rew.shape[3]
    df_psth_non_rew = []
    days = ['D-3','D-2', 'D-1', 'D0', 'D+1', 'D+2']
    for imouse in range(nmice):
        for isession in range(ndays):
            cell_count = 0
            for itype in range(ntypes):
                ncells = np.sum(~np.isnan(traces_non_rew[imouse, isession, itype, :, 0, 0]))
                for icell in range(ncells):
                    response = np.nanmean(traces_non_rew[imouse, isession, itype, icell], axis=0)
                    df = pd.DataFrame([], columns = ['time', 'activity', 'roi', 'cell_type', 'session_id', 'mouse_id'])
                    df['time'] = np.arange(traces_non_rew.shape[5]) / 30
                    df['activity'] = response
                    df['roi'] = cell_count
                    df['cell_type'] = metadata_non_rew['cell_types'][itype]
                    df['mouse_id'] = metadata_non_rew['mice'][imouse]
                    df['session_id'] = days[isession]
                    df_psth_non_rew.append(df)
                    cell_count += 1

    df_psth_non_rew = pd.concat(df_psth_non_rew, ignore_index=True)
    df_psth_non_rew['reward_group'] = 'R-'
    df_psth = pd.concat((df_psth_rew, df_psth_non_rew), ignore_index=True)

    return df_psth

df_psth = convert_psth_np_to_pd(traces_rew, traces_non_rew)

# Remove sessions with baseline aberrations.
df_psth.loc[(df_psth.mouse_id=='GF308') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='GF208') & (df_psth.session_id=='D-2'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='MI075') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='AR144') & (df_psth.session_id=='D-2'), 'activity'] = np.nan


# Count cells.
# ------------

np.sum(np.nansum(~np.isnan(traces_rew[:,0,:,:,0,0]), axis=(1,2)))
np.sum(np.nansum(~np.isnan(traces_non_rew[:,0,:,:,0,0]), axis=(1,2)))

np.nansum(~np.isnan(traces_rew[:,0,1,:,0,0]), axis=(1))
np.nansum(~np.isnan(traces_non_rew[:,0,1,:,0,0]), axis=(1))

# Count trials per session.
np.nansum(~np.isnan(traces_rew[:,:,0,0,:,0]), axis=(2))
np.nansum(~np.isnan(traces_non_rew[:,:,0,0,:,0]), axis=(2))


# mouse_id = 'AR133'
# far_red_file = f"\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\projection_neurons\\FarRedRois.npy"
# red_file = f"\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\{mouse_id}\\projection_neurons\\RedRois.npy"
# red = np.load(red_file, allow_pickle=True)
# far_red = np.load(far_red_file, allow_pickle=True)
# print(f"mouse {mouse_id} red {red.size} far red {far_red.size}")

# from nwb_wrappers import nwb_reader_functions as nwb_read
# nwb = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\AR133_20240424_130306.nwb"
# rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
# rrs = nwb_read.get_cell_indices_by_cell_type(nwb, rrs_keys)

# # Plot responses across cells.
# day_m1 = np.nanmean(traces_rew, axis=(4))
# day_m1 = np.nanmean(day_m1, axis=(0,2,3))[1]
# day_1 = np.nanmean(traces_rew, axis=(4))
# day_1 = np.nanmean(day_1, axis=(0,2,3))[3]

# day_m1_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_m1_non_rew = np.nanmean(day_m1_non_rew, axis=(0,2,3))[1]
# day_1_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_1_non_rew = np.nanmean(day_1_non_rew, axis=(0,2,3))[3]

# f, axes = plt.subplots(1,2, sharey=True)
# axes[0].plot(day_m1)
# axes[0].plot(day_1)

# axes[1].plot(day_m1_non_rew)
# axes[1].plot(day_1_non_rew)



# # Projection neurons.
# # -------------------

# # Plot responses across cells.
# day_m1_S2 = np.nanmean(traces_rew, axis=(4))
# day_m1_S2 = np.nanmean(day_m1_S2, axis=(0,3))[1,2]
# day_1_S2 = np.nanmean(traces_rew, axis=(4))
# day_1_S2 = np.nanmean(day_1_S2, axis=(0,3))[3,2]

# day_m1_S2_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_m1_S2_non_rew = np.nanmean(day_m1_S2_non_rew, axis=(0,3))[1,2]
# day_1_S2_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_1_S2_non_rew = np.nanmean(day_1_S2_non_rew, axis=(0,3))[3,2]

# # Plot responses across cells.
# day_m1_M1 = np.nanmean(traces_rew, axis=(4))
# day_m1_M1 = np.nanmean(day_m1_M1, axis=(0,3))[1,1]
# day_1_M1 = np.nanmean(traces_rew, axis=(4))
# day_1_M1 = np.nanmean(day_1_M1, axis=(0,3))[3,1]

# day_m1_M1_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_m1_M1_non_rew = np.nanmean(day_m1_M1_non_rew, axis=(0,3))[1,1]
# day_1_M1_non_rew = np.nanmean(traces_non_rew, axis=(4))
# day_1_M1_non_rew = np.nanmean(day_1_M1_non_rew, axis=(0,3))[3,1]

# f, axes = plt.subplots(2,2, sharey=True)

# axes[0,0].plot(day_m1_M1, color='k')
# axes[0,0].plot(day_1_M1, color='b')

# axes[0,1].plot(day_m1_S2, color='k')
# axes[0,1].plot(day_1_S2, color='r')

# axes[1,0].plot(day_m1_M1_non_rew, color='k')
# axes[1,0].plot(day_1_M1_non_rew, color='b')

# axes[1,1].plot(day_m1_S2_non_rew, color='k')
# axes[1,1].plot(day_1_S2_non_rew, color='r')




# # Plot the 5 days for each mouse.
# # ###############################

# sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
#               rc={'font.sans-serif':'Arial'})
# palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])

# # psth_rew = np.nanmean(traces_rew, axis=(4))
# # psth_rew = np.nanmean(psth_rew, axis=(3))

# # psth_non_rew = np.nanmean(traces_non_rew, axis=(4))
# # psth_non_rew = np.nanmean(psth_non_rew, axis=(3))

# # TODO: add ROI in metadata dict. 



# nmice = psth_rew.shape[0]
# for imouse in range(nmice):
#     f, axes = plt.subplots(1,5, sharey=True, figsize=(16,3.2))
#     mouse = metadata_rew['mice'][imouse]
#     f.suptitle(f'{mouse} R+')
#     for iday in range(5):
#         axes[iday].plot(psth_rew[imouse, iday], color=palette[0])
    
# nmice = psth_non_rew.shape[0]
# for imouse in range(nmice):
#     f, axes = plt.subplots(1,5, sharey=True, figsize=(16,3.2))
#     mouse = metadata_non_rew['mice'][imouse]
#     f.suptitle(f'{mouse} R-')
#     for iday in range(5):
#         axes[iday].plot(psth_non_rew[imouse, iday], color=palette[0])



# # Projection neurons for each mouse.
# # ##################################

# sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
#               rc={'font.sans-serif':'Arial'})
# palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])

# # Plot responses across cells.
# psth_S2_rew = np.nanmean(traces_rew, axis=(4))
# psth_S2_rew = np.nanmean(psth_S2_rew, axis=(3))[:,:,2]

# psth_S2_non_rew = np.nanmean(traces_non_rew, axis=(4))
# psth_S2_non_rew = np.nanmean(psth_S2_non_rew, axis=(3))[:,:,2]

# # Plot responses across cells.
# psth_M1_rew = np.nanmean(traces_rew, axis=(4))
# psth_M1_rew = np.nanmean(psth_M1_rew, axis=(3))[:,:,1]

# psth_M1_non_rew = np.nanmean(traces_non_rew, axis=(4))
# psth_M1_non_rew = np.nanmean(psth_M1_non_rew, axis=(3))[:,:,1]

# nmice = psth_rew.shape[0]
# for imouse in range(nmice):
#     f, axes = plt.subplots(1,5, sharey=True, figsize=(18,3.2))
#     mouse = metadata_rew['mice'][imouse]
#     f.suptitle(f'{mouse} R+')
#     for iday in range(5):
#         axes[iday].plot(psth_S2_rew[imouse, iday], color=palette[2])
#         axes[iday].plot(psth_M1_rew[imouse, iday], color=palette[1])

# nmice = psth_non_rew.shape[0]
# for imouse in range(nmice):
#     f, axes = plt.subplots(1,5, sharey=True, figsize=(18,3.2))
#     mouse = metadata_non_rew['mice'][imouse]
#     f.suptitle(f'{mouse} R-')
#     for iday in range(5):
#         axes[iday].plot(psth_S2_non_rew[imouse, iday], color=palette[2])
#         axes[iday].plot(psth_M1_non_rew[imouse, iday], color=palette[1])



sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})
palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])

mice_rew = df_psth.loc[df_psth.reward_group=='R+', 'mouse_id'].unique()
mice_non_rew = df_psth.loc[df_psth.reward_group=='R-', 'mouse_id'].unique()



# psth across days and all cells

# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#238443', '#d51a1c'])
sns.relplot(data=df_psth, x='time', y='activity', errorbar='se', col='session_id',
            kind='line', hue='reward_group',
            hue_order=['R+','R-'], palette=palette)
plt.ylim([-0.005,0.05])

# D-1 VS D+1
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id')

# D-1 VS D0
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D0'])]
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id')

# D-1 VS D+2
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id')

# for projectors
# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#3351ff', '#c959af'])
data = df_psth.loc[ (df_psth.cell_type.isin(['wM1','wS2']))]
fig = sns.relplot(data=data, x='time', y='activity', errorbar='se', row='reward_group', col='session_id',
            kind='line', palette=palette, hue='cell_type')
fig.set_titles(col_template='{col_name}')

# D-1 VS D+1 for projection neurons
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1']) & (df_psth.cell_type.isin(['wM1','wS2']))]
sns.relplot(data=data, x='time', y='activity', errorbar='se', row='reward_group', col='session_id',
            kind='line', palette=palette[1:], hue='cell_type')
plt.ylim([-0.02,0.1])


# same plot for across populations rather than cells

data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'time'], as_index=False).agg({'activity':np.nanmean})
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id', hue_order=['D-1', 'D+1'], col_order=['R+','R-'])



# Rewarded one plot per mouse

data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[:5])]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  errorbar='se', facet_kws={'ylim':(-0.02,0.1)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[5:10])]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  errorbar='se', facet_kws={'ylim':(-0.02,0.06)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[10:15])]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  errorbar='se', facet_kws={'ylim':(-0.02,0.06)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


# Rewarded one plot per mouse projection neurons

data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[:5]) & (df_psth.cell_type != 'na')]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  hue='cell_type', errorbar='se', facet_kws={'ylim':(-0.02,0.2)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[5:10]) & (df_psth.cell_type != 'na')]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  hue='cell_type', errorbar='se', facet_kws={'ylim':(-0.02,0.1)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[df_psth.mouse_id.isin(mice_rew[10:15]) & (df_psth.cell_type != 'na')]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  hue='cell_type', errorbar='se', facet_kws={'ylim':(-0.02,0.1)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


# Non reward once plot per mouse

data = df_psth.loc[(df_psth.mouse_id.isin(mice_non_rew[:5]))]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  errorbar='se', facet_kws={'ylim':(-0.02,0.06)})
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[(df_psth.mouse_id.isin(mice_non_rew[5:10]))]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  errorbar='se', facet_kws={'ylim':(-0.02,0.06)})
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


# Non rewarded projectors

data = df_psth.loc[(df_psth.mouse_id.isin(mice_non_rew[:5])) & (df_psth.cell_type != 'na')]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  hue='cell_type', errorbar='se', facet_kws={'ylim':(-0.02,0.2)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


data = df_psth.loc[(df_psth.mouse_id.isin(mice_non_rew[5:10])) & (df_psth.cell_type != 'na')]
fig = sns.relplot(data=data, x='time', y='activity', row='mouse_id', col='session_id', kind='line',
                  hue='cell_type', errorbar='se', facet_kws={'ylim':(-0.02,0.1)}, legend=None)
fig.set_titles(col_template='{col_name}')
fig.tight_layout()


# Stability of population responses during the pretraining days.
# --------------------------------------------------------------

palette = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc', '#333333']
sns.palplot(palette)
df_max = df_psth.loc[(df_psth.time>1) & (df_psth.time<2)]
df_max = df_max.groupby(['mouse_id', 'session_id', 'reward_group', 'cell_type', 'roi'], as_index=False).agg({'activity':np.nanmax})
df_max = df_max.groupby(['mouse_id', 'session_id', 'reward_group',], as_index=False).agg({'activity':np.nanmean})




plt.figure()
sns.pointplot(data=df_max, x='session_id', y='activity', errorbar='se', hue='reward_group',
              order=['D-2', 'D-1', 'D0', 'D+1', 'D+2'], hue_order=['R+','R-'], palette=palette[2:4])
plt.ylim([0,0.1])
sns.despine()



data = df_max.loc[df_max.mouse_id.isin(['GF310', 'GF317', 'GF311'])]
# data = df_max.loc[df_max.mouse_id.isin(['GF308', 'GF318', 'GF333'])]
plt.figure()
sns.pointplot(data=data, x='session_id', y='activity', errorbar='se', hue='mouse_id',
              order=['D-3','D-2', 'D-1', 'D0', 'D+1', 'D+2'])
plt.ylim([0,0.1])
sns.despine()

# same for the projectors
palette = sns.color_palette(['#3351ff', '#c959af'])
df_max = df_psth.loc[(df_psth.time>1) & (df_psth.time<2)]
df_max = df_max.groupby(['mouse_id', 'session_id', 'reward_group', 'cell_type', 'roi'], as_index=False).agg({'activity':np.nanmax})
# df_max = df_max.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
df_max = df_max.loc[df_max.cell_type.isin(['wM1','wS2'])]

fig, (ax1, ax2) = plt.subplots(2,1)
data = df_max.loc[df_max.reward_group=='R+']
sns.pointplot(data=data, x='session_id', y='activity', errorbar='se', hue='cell_type',
              order=['D-2', 'D-1', 'D0', 'D+1', 'D+2'], hue_order=['wM1', 'wS2'],
              palette=palette, ax=ax1)
data = df_max.loc[df_max.reward_group=='R-']
sns.pointplot(data=data, x='session_id', y='activity', errorbar='se', hue='cell_type',
              order=['D-2', 'D-1', 'D0', 'D+1', 'D+2'], hue_order=['wM1', 'wS2'],
              palette=palette, ax=ax2)
ax1.set_ylim([0,0.1])
ax2.set_ylim([0,0.1])

sns.despine()


# Plot crade

perf = [0.8, 0.61, 0.59, .17, .38, .4]
act = [-0.009, 0.01, -0.01, -0.006, 0.019, 0.005]

fig = plt.figure()
plt.scatter(perf, act)
sns.despine()
ax = plt.gca()
ax.set_xlabel('Performance at D0')
ax.set_ylabel('population average dff D0 - D-1')

# PSTH's with responsive cells.
# #############################

samples_rew = np.nanmax(traces_rew[:,:,:,:,:,30:60], axis=5) - np.nanmax(traces_rew[:,:,:,:,:,0:30], axis=5)
samples_rew = np.concatenate([samples_rew[:,i] for i in range(samples_rew.shape[1])], axis=3)
responsive_mask_rew = np.full((samples_rew.shape[:3]), False)
samples_non_rew = np.nanmax(traces_non_rew[:,:,:,:,:,30:60], axis=5) - np.nanmax(traces_non_rew[:,:,:,:,:,0:30], axis=5)
samples_non_rew = np.concatenate([samples_non_rew[:,i] for i in range(samples_non_rew.shape[1])], axis=3)
responsive_mask_non_rew = np.full((samples_non_rew.shape[:3]), False)

for imouse in range(samples_rew.shape[0]):
    for itype in range(samples_rew.shape[1]):
        for icell in range(samples_rew.shape[2]):
            if not np.all(np.isnan(samples_rew[imouse,itype,icell])):
                _, p = wilcoxon(samples_rew[imouse,itype,icell], nan_policy='omit', alternative='greater')
                if p <= 0.05:
                    responsive_mask_rew[imouse,itype,icell] = True
for imouse in range(samples_non_rew.shape[0]):
    for itype in range(samples_non_rew.shape[1]):
        for icell in range(samples_non_rew.shape[2]):
            if not np.all(np.isnan(samples_non_rew[imouse,itype,icell])):
                _, p = wilcoxon(samples_non_rew[imouse,itype,icell], nan_policy='omit', alternative='greater')
                if p <= 0.05:
                    responsive_mask_non_rew[imouse,itype,icell] = True


# Adding say dim to the mask
responsive_mask_rew = np.repeat(responsive_mask_rew[:,np.newaxis], repeats=traces_rew.shape[1], axis=1)
responsive_mask_non_rew = np.repeat(responsive_mask_non_rew[:,np.newaxis], repeats=traces_non_rew.shape[1], axis=1)
traces_rew[~responsive_mask_rew] = np.nan
traces_non_rew[~responsive_mask_non_rew] = np.nan

df_psth_resp = convert_psth_np_to_pd(traces_rew, traces_non_rew)



# check selected cells.
traces_rew.shape
psth_rew = np.nanmean(traces_rew, axis=4)

psth_flat = psth


# D-1 VS D+1
data = df_psth_resp.loc[df_psth_resp.session_id.isin(['D-1', 'D+1'])]
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id')
