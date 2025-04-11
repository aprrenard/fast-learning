"""This script generates PSTH numpy arrays from lists of NWB files.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from sklearn.metrics import auc, roc_curve
from sklearn.utils import shuffle

# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

PROCESSED_DATA_PATH = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed'
# PROCESSED_DATA_PATH = r'E:\anthony\analysis\data_processed'


def remove_nan_segments_from_axis(array, axis):
    t = [i for i in range(array.ndim) if i != axis]
    mask = ~np.isnan(array).all(axis=tuple(t))
    slices = [slice(None)] * array.ndim  # Create a list of slice(None)
    slices[axis] = mask  # Replace the slice for the specified axis with the mask
    return array[tuple(slices)]


def convert_psth_np_to_pd(data, metadata, reward_group, lmi_rew=None, axis='cell'):

    nmice, ndays, ntypes = data.shape[:3]
    ncells = data.shape[2] * data.shape[3]
    df_list = []
    if ndays == 5:
        days = ['D-2', 'D-1', 'D0', 'D+1', 'D+2']
    elif ndays == 6:
        days = ['D-3', 'D-2', 'D-1', 'D0', 'D+1', 'D+2']
    elif ndays == 3:
        days = ['D0', 'D+1', 'D+2']
    for imouse in range(nmice):
        for isession in range(ndays):
            for itype in range(ntypes):
                print(imouse, isession, itype, end='\r')
                
                
                if axis == 'cell':
                    ncells = np.sum(~np.isnan(data[imouse, isession, itype, :, 0, 0]))
                    counter = 0
                    for icell in range(ncells):
                        if np.all(np.isnan(data[imouse, isession, itype, icell])):
                            continue
                        response = np.nanmean(data[imouse, isession, itype, icell], axis=0)
                        df = pd.DataFrame([], columns = ['time', 'activity', 'roi', 'cell_type', 'session_id', 'mouse_id'])
                        df['time'] = np.arange(data.shape[5]) / 30
                        df['activity'] = response
                        df['roi'] = counter

                        df['cell_type'] = metadata['cell_types'][metadata_rew["sessions"][imouse][isession]][itype]
                        df['mouse_id'] = metadata['mice'][imouse]
                        df['session_id'] = days[isession]
                        if lmi_rew is not None:
                            df['lmi'] = lmi_rew[imouse, isession, itype, icell]
                        
                        df_list.append(df)
                        counter += 1
                        
                elif axis == 'event':
                    nevents = np.sum(~np.isnan(data[imouse, isession, itype, 0, :, 0]))
                    counter = 0
                    for ievent in range(nevents):
                        if np.all(np.isnan(data[imouse, isession, itype, :, ievent])):
                            continue
                        response = np.nanmean(data[imouse, isession, itype, :, ievent], axis=0)
                        df = pd.DataFrame([], columns = ['time', 'activity', 'event', 'cell_type', 'session_id', 'mouse_id'])
                        df['time'] = np.arange(data.shape[5]) / 30
                        df['activity'] = response
                        df['event'] = counter
                        df['cell_type'] = metadata['cell_types'][itype]
                        df['mouse_id'] = metadata['mice'][imouse]
                        df['session_id'] = days[isession]
                        if lmi_rew is not None:
                            df['lmi'] = lmi_rew[imouse, isession, itype, ievent]
                        
                        df_list.append(df)
                        counter += 1

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df['reward_group'] = reward_group

    return df


def compute_lmi(data, nshuffles=10000):
    '''
    Compute ROC analysis and Learning modulation index for each cell in data.
    Data: np array of shape (mouse, day, type, cell, trial).

    Assumes only 5 days (sessions) are given.
    LMI are computed on days D-2, D-1 together VERSUS D+1, D+2 together.
    '''
    nmouse, nsession, ntype, ncell, _ = data.shape

    lmi = np.full((nmouse, ntype, ncell), np.nan)
    lmi_p = np.full((nmouse, ntype, ncell), np.nan)
    for imouse in range(nmouse):
        for itype in range(ntype):
            for icell in range(ncell):
                print(f'{imouse}/{nmouse} mice {icell}/{ncell} cells', end='\r')
                X = [data[imouse,i,itype,icell] for i in [0,1,3,4]]
                X = [x[~np.isnan(x)] for x in X]
                
                # Accounting for nan cells.
                if X[0].size == 0:
                    continue
                
                X_pre = np.r_[X[0], X[1]]
                X_post = np.r_[X[2], X[3]]
                X = np.r_[X_pre, X_post]
                y = np.r_[[0 for _ in range(X_pre.shape[0])], [1 for _ in range(X_post.shape[0])]]
                
                fpr, tpr, _ = roc_curve(y, X)
                roc_auc = auc(fpr, tpr)
                lmi[imouse, itype, icell] = (roc_auc - 0.5) * 2
                
                # Test significativity of LMI values with shuffles.
                roc_auc_shuffle = np.zeros(nshuffles)
                for ishuffle in range(nshuffles):
                    y_shuffle = shuffle(y, random_state=ishuffle)
                    fpr, tpr, _ = roc_curve(y_shuffle, X)
                    roc_auc_shuffle[ishuffle] = auc(fpr, tpr)
                    
                if roc_auc <= np.percentile(roc_auc_shuffle, 2.5):
                    lmi_p[imouse, itype, icell] = -1
                elif roc_auc >= np.percentile(roc_auc_shuffle, 97.5):
                    lmi_p[imouse, itype, icell] = 1
                else:
                    lmi_p[imouse, itype, icell] = 0

    # Extend lmi arrays on session dim.
    lmi = np.concatenate([lmi[:,np.newaxis] for _ in range(nsession)], axis=1)
    lmi_p = np.concatenate([lmi_p[:,np.newaxis] for _ in range(nsession)], axis=1)

    return lmi, lmi_p


# #######################################
# PSTH analysis for the whole population.
# #######################################

# Load numpy datasets.
read_path = os.path.join(PROCESSED_DATA_PATH, 'psth_sensory_map_trials_rewarded.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'psth_sensory_map_trials_rewarded_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

read_path = os.path.join(PROCESSED_DATA_PATH, 'psth_sensory_map_trials_non_rewarded.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'psth_sensory_map_trials_non_rewarded_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

traces_rew = traces_rew.astype(np.float32)
traces_non_rew = traces_non_rew.astype(np.float32)

# Substract baseline.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Remove my two shitty mice.
traces_rew = traces_rew[:-2]

metadata_rew["cell_types"][metadata["sessions"][3][5]]
metadata_rew["mice"]
metadata["mice"][3]



# Convert psth to pandas.
df_psth_rew = convert_psth_np_to_pd(traces_rew, metadata_rew, 'R+')
df_psth = convert_psth_np_to_pd(traces_rew, traces_non_rew)
# Remove sessions with baseline aberrations.
df_psth.loc[(df_psth.mouse_id=='GF308') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='GF208') & (df_psth.session_id=='D-2'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='MI075') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='AR144') & (df_psth.session_id=='D-2'), 'activity'] = np.nan

# traces_rew = np.nanmean(traces_rew, axis=4)
# traces_non_rew = np.nanmean(traces_non_rew, axis=4)

# for icell in range(300):
#     plt.plot(df_psth.loc[(df_psth.mouse_id=='GF208')
#                          & (df_psth.session_id=='D-2')
#                          & (df_psth.roi==icell)
#                          & (df_psth.event==icell), 'activity'] + 0.1 * icell)



# Count cells.
# ------------

np.sum(np.nansum(~np.isnan(traces_rew[:,2,[1],:,0,0]), axis=(1,2)))
np.sum(np.nansum(~np.isnan(traces_non_rew[:,2,[1],:,0,0]), axis=(1,2)))

np.nansum(~np.isnan(traces_rew[:,0,1,:,0,0]), axis=(1))
np.nansum(~np.isnan(traces_non_rew[:,0,1,:,0,0]), axis=(1))

# Count trials per session.
np.nansum(~np.isnan(traces_rew[:,:,0,0,:,0]), axis=(2))
np.nansum(~np.isnan(traces_non_rew[:,:,0,0,:,0]), axis=(2))




# Figure 3.
# ---------

sns.set_theme(context='poster', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})

# PSTH.
data = df_psth.loc[df_psth.session_id.isin(['D-2', 'D-1', 'D0', 'D+1','D+2'])]
# data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'time'], as_index=False).agg({'activity':np.nanmean})
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
palette = sns.color_palette(['#212120', '#238443',])
palette = sns.color_palette(['#212120', '#d51a1c'])
palette = sns.color_palette('viridis')
sns.lineplot(data=data.loc[data.reward_group=='R+'], x='time', y='activity', errorbar='se',
            hue='session_id', ax=ax1, palette=palette, hue_order=['D-2', 'D-1', 'D0', 'D+1','D+2'])
sns.lineplot(data=data.loc[data.reward_group=='R-'], x='time', y='activity', errorbar='se',
            hue='session_id', ax=ax2, palette=palette, hue_order=['D-2', 'D-1', 'D0', 'D+1','D+2'])

palette = sns.color_palette(['#238443', '#d51a1c'])
sns.lineplot(data=data.loc[data.session_id=='D-1'], x='time', y='activity', errorbar='se',
            hue='reward_group', palette=palette, hue_order=['R-', 'R+'])
fig = sns.relplot(data=data.loc[data.session_id=='D-1'], x='time', y='activity', errorbar='se', col='mouse_id',
            row='reward_group', palette=palette,  kind='line')
fig.set_titles(col_template='{col_name}')

df_psth.loc[df_psth.reward_group=='R+', 'mouse_id'].unique()



# Plot PSTH.
# ----------

sns.set_theme(context='poster', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})
mice_rew = df_psth.loc[df_psth.reward_group=='R+', 'mouse_id'].unique()
mice_non_rew = df_psth.loc[df_psth.reward_group=='R-', 'mouse_id'].unique()

# psth across days and all cells

# all days
data = df_psth.loc[(df_psth.time>=0.5) & (df_psth.time<=3)]
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='session_id',
            kind='line', hue='reward_group',
            hue_order=['R-','R+'], palette=sns.color_palette(['#fe0404ff', '#0da50dff']))
# plt.ylim([-0.005,0.05])

# D-1 VS D+1
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1']) & (df_psth.reward_group=='R+')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=ax1, palette=sns.color_palette(['#212120','#0da50dff']))
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1']) & (df_psth.reward_group=='R-')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=ax2, palette=sns.color_palette(['#212120','#fe0404ff']))
ax1.set_ylim(-0.005,0.06)
ax2.set_ylim(-0.005,0.06)
sns.despine()

# Response peak point plot and test.
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
data = data.loc[(data.time>=1) & (data.time<=1.180)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'cell_type', 'roi'],
                    as_index=False).agg({'activity':np.nanmax})

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
data1 = data.loc[(data.reward_group=='R+')]
sns.pointplot(data=data1, x='session_id', y='activity', errorbar='se',
            ax=ax1, order=['D-1','D+1'], color='#0da50dff')
data2 = data.loc[(data.reward_group=='R-')]
sns.pointplot(data=data2, x='session_id', y='activity', errorbar='se',
            ax=ax2, order=['D-1','D+1'], color='#fe0404ff')
ax1.set_ylim(0.0,0.06)
ax2.set_ylim(0.0,0.06)
sns.despine()

# Stats
samples = data.loc[(data.reward_group=='R+')]
g1 = samples.loc[samples.session_id=='D-1', 'activity'].to_numpy()
g2 = samples.loc[samples.session_id=='D+1', 'activity'].to_numpy()
wilcoxon(g1, g2)



# Same for the projectors.
fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.reward_group=='R+')
                   & (df_psth.cell_type=='wM1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=axes[0,0], palette=sns.color_palette(['#212120','#3351ff']))
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.reward_group=='R+')
                   & (df_psth.cell_type=='wS2')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=axes[0,1], palette=sns.color_palette(['#212120','#c959af']))
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.reward_group=='R-')
                   & (df_psth.cell_type=='wM1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=axes[1,0], palette=sns.color_palette(['#212120','#3351ff']))
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.reward_group=='R-')
                   & (df_psth.cell_type=='wS2')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='session_id', ax=axes[1,1], palette=sns.color_palette(['#212120','#c959af']))
for ax in axes.flatten():
    ax.set_ylim(-0.01,0.08)
# ax2.set_ylim(-0.005,0.06)
sns.despine()


# Response peak point plot and test.
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
data = data.loc[(data.time>=1) & (data.time<=1.300)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'cell_type', 'roi'],
                    as_index=False).agg({'activity':np.nanmax})

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
data1 = data.loc[(data.reward_group=='R+') & (data.cell_type=='wM1')]
sns.pointplot(data=data1, x='session_id', y='activity', errorbar='se',
            ax=ax1, order=['D-1','D+1'], color='#3351ff')
data1 = data.loc[(data.reward_group=='R+') & (data.cell_type=='wS2')]
sns.pointplot(data=data1, x='session_id', y='activity', errorbar='se',
            ax=ax1, order=['D-1','D+1'], color='#c959af')
data2 = data.loc[(data.reward_group=='R-') & (data.cell_type=='wM1')]
sns.pointplot(data=data2, x='session_id', y='activity', errorbar='se',
            ax=ax2, order=['D-1','D+1'], color='#3351ff')
data2 = data.loc[(data.reward_group=='R-') & (data.cell_type=='wS2')]
sns.pointplot(data=data2, x='session_id', y='activity', errorbar='se',
            ax=ax2, order=['D-1','D+1'], color='#c959af')
ax1.set_ylim(0,0.1)
ax2.set_ylim(0,0.1)
sns.despine()

# Stats
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
data = data.loc[(data.time>=1) & (data.time<=1.3)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'cell_type', 'roi'],
                    as_index=False).agg({'activity':np.nanmean})

data1 = data.loc[(data.reward_group=='R+') & (data.cell_type=='wM1')]
g1 = data1.loc[data1.session_id=='D-1', 'activity'].to_numpy()
g2 = data1.loc[data1.session_id=='D+1', 'activity'].to_numpy()
wilcoxon(g1, g2)

data1 = data.loc[(data.reward_group=='R+') & (data.cell_type=='wS2')]
g1 = data1.loc[data1.session_id=='D-1', 'activity'].to_numpy()
g2 = data1.loc[data1.session_id=='D+1', 'activity'].to_numpy()
wilcoxon(g1, g2)

data1 = data.loc[(data.reward_group=='R-') & (data.cell_type=='wM1')]
g1 = data1.loc[data1.session_id=='D-1', 'activity'].to_numpy()
g2 = data1.loc[data1.session_id=='D+1', 'activity'].to_numpy()
wilcoxon(g1, g2)



# for projectors
# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#3351ff', '#c959af'])
data = df_psth.loc[(df_psth.cell_type.isin(['wM1','wS2']))]
data = data.loc[(data.time>=0.5) & (data.time<=3)]
fig = sns.relplot(data=data, x='time', y='activity', errorbar='se', row='reward_group', col='session_id',
            kind='line', palette=palette, hue='cell_type')
fig.set_titles(col_template='{col_name}')
plt.ylim([-0.01,0.08])

# D-1 VS D+1 for projection neurons
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1']) & (df_psth.cell_type.isin(['wM1','wS2']))]
sns.relplot(data=data, x='time', y='activity', errorbar='se', row='reward_group', col='session_id',
            kind='line', palette=palette[1:], hue='cell_type')
plt.ylim([-0.0,0.1])


# same plot for across populations rather than cells
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'time'], as_index=False).agg({'activity':np.nanmean})
sns.relplot(data=data, x='time', y='activity', errorbar='se', col='reward_group',
            kind='line', hue='session_id', hue_order=['D-1', 'D+1'], col_order=['R+','R-'])



# One plot per mouse.

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


# One plot per mouse projection neurons

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


# Non rewarded once plot per mouse

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


# ##############################
# PSTH during the whisker day 0.
# ##############################

sns.set_theme(context='poster', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})

# Plot again D-1 VS D+1 for the projectors but two populations on the same plot.
palette = sns.color_palette(['#3351ff','#c959af'])
data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group', 'time'], as_index=False).agg({'activity':np.nanmean})

fig, axes = plt.subplots(2,2, sharex=True, sharey=True)

data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.session_id=='D-1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,0], palette=palette)

data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.session_id=='D+1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,1], palette=palette)

data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R-') 
                   & (df_psth.session_id=='D-1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,0], palette=palette)

data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+1'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.session_id=='D+1')]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,1], palette=palette)

for ax in axes.flatten():
    ax.set_ylim(-0.01,0.08)
# ax2.set_ylim(-0.005,0.06)
sns.despine()


# 5 first WH vs remaining WH vs WH misses
# #######################################

# Load numpy datasets.
read_path = os.path.join(PROCESSED_DATA_PATH, 'WH_trials_rew_common.npy')
wh_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'WH_trials_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_wh_rew = pickle.load(fid)
    
read_path = os.path.join(PROCESSED_DATA_PATH, 'WH_trials_non_rew_common.npy')
wh_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'WH_trials_non_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_wh_non_rew = pickle.load(fid)
    
read_path = os.path.join(PROCESSED_DATA_PATH, 'WM_trials_rew_common.npy')
wm_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'WM_trials_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_wm_rew = pickle.load(fid)
    
read_path = os.path.join(PROCESSED_DATA_PATH, 'WM_trials_non_rew_common.npy')
wm_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'WM_trials_non_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_wm_non_rew = pickle.load(fid)

# Substract baseline.
wh_rew = wh_rew - np.nanmean(wh_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
wh_non_rew = wh_non_rew - np.nanmean(wh_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
wm_rew = wm_rew - np.nanmean(wm_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
wm_non_rew = wm_non_rew - np.nanmean(wm_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

wh_rew = wh_rew[:-2]
wm_rew = wm_rew[:-2]

# Non motivated trials.
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common.npy')
um_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_um_rew = pickle.load(fid)

read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common.npy')
um_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_um_non_rew = pickle.load(fid)           

# Substract baseline.
um_rew = um_rew - np.nanmean(um_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
um_non_rew = um_non_rew - np.nanmean(um_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Remove my two shitty mice.
um_rew = um_rew[:-2]

# Convert to pandas.
df_wh_rew = convert_psth_np_to_pd(wh_rew, metadata_wh_rew, 'R+', lmi_rew=None, axis='event')
df_wh_rew['event_type'] = 'WH'
df_wh_rew['motivation'] = 'm'
df_wm_rew = convert_psth_np_to_pd(wm_rew, metadata_wm_rew, 'R+', lmi_rew=None, axis='event')
df_wm_rew['event_type'] = 'WM'
df_wm_rew['motivation'] = 'm'
df_wh_non_rew = convert_psth_np_to_pd(wh_non_rew, metadata_wh_non_rew, 'R-', lmi_rew=None, axis='event')
df_wh_non_rew['event_type'] = 'WH'
df_wh_non_rew['motivation'] = 'm'
df_wm_non_rew = convert_psth_np_to_pd(wm_non_rew, metadata_wm_non_rew, 'R-', lmi_rew=None, axis='event')
df_wm_non_rew['event_type'] = 'WM'
df_wm_non_rew['motivation'] = 'm'

# Convert psth to pandas.
df_um_rew = convert_psth_np_to_pd(um_rew, metadata_um_rew, 'R+', lmi_rew=None, axis='event')
df_um_rew['event_type'] = 'WH'
df_um_rew['motivation'] = 'um'
df_um_non_rew = convert_psth_np_to_pd(um_non_rew, metadata_um_non_rew, 'R-', lmi_rew=None, axis='event')
df_um_non_rew['event_type'] = 'WH'
df_um_non_rew['motivation'] = 'um'

df_psth = pd.concat((df_wh_rew, df_wm_rew, df_wh_non_rew, df_wm_non_rew, df_um_rew, df_um_non_rew), ignore_index=True)
# Remove single wS2 cell.
df_psth.loc[(df_psth.mouse_id=='AR131') & (df_psth.cell_type=='wS2'), 'activity'] = np.nan
# Remove sessions with baseline aberrations.
df_psth.loc[(df_psth.mouse_id=='GF308') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='GF208') & (df_psth.session_id=='D-2'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='MI075') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='AR144') & (df_psth.session_id=='D-2'), 'activity'] = np.nan






sns.set_theme(context='poster', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})

fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
palette = sns.color_palette(['#3351ff','#c959af'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event<10)
                   & (df_psth.motivation=='m')
                   ]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,0], palette=palette, hue_order=['wM1', 'wS2'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                    & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event>=10)]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,1], palette=palette, hue_order=['wM1', 'wS2'])
   
# data = df_psth.loc[df_psth.session_id.isin(['D0'])
#                    & (df_psth.cell_type.isin(['wM1','wS2']))
#                    & (df_psth.time<=3)
#                    & (df_psth.reward_group=='R+')
#                    & (df_psth.event_type=='WM')
#                    & (df_psth.event<5)]
# sns.lineplot(data=data, x='time', y='activity', errorbar='se',
#             hue='cell_type', ax=axes[0,2], palette=palette, hue_order=['wM1', 'wS2'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                                      & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WM')
                   & (df_psth.event>=10)]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,2], palette=palette, hue_order=['wM1', 'wS2'])
   
data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')

                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event<10)]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,0], palette=palette, hue_order=['wM1', 'wS2'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')

                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event>=10)]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,1], palette=palette, hue_order=['wM1', 'wS2'])

# data = df_psth.loc[df_psth.session_id.isin(['D0'])
#                    & (df_psth.cell_type.isin(['wM1','wS2']))
#                    & (df_psth.time<=3)
#                    & (df_psth.reward_group=='R-')
#                    & (df_psth.event_type=='WM')
#                    & (df_psth.event<5)]
# sns.lineplot(data=data, x='time', y='activity', errorbar='se',
#             hue='cell_type', ax=axes[1,2], palette=palette, hue_order=['wM1', 'wS2'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')

                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time<=3)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WM')
                   & (df_psth.event>=10)]
sns.lineplot(data=data, x='time', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,2], palette=palette, hue_order=['wM1', 'wS2'])
for ax in axes.flatten():
    ax.set_ylim(-.02, 0.08)
sns.despine() 
 

# Quantification point plots and stats.

fig, axes = plt.subplots(2,4, sharex=True, sharey=True)
palette = sns.color_palette(['#3351ff','#c959af'])

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event<10)
                   & (df_psth.motivation=='m')
                   ]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,0], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[0,0].title.set_text(p)

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                    & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event>=10)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,1], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[0,1].title.set_text(p)


data = df_psth.loc[df_psth.session_id.isin(['D0'])
                                      & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WM')
                   & (df_psth.event>=10)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,2], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[0,2].title.set_text(p)


data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='um')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R+')
                   & (df_psth.event_type=='WH')]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[0,3], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[0,3].title.set_text(p)

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event<10)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,0], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[1,0].title.set_text(p)

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WH')
                   & (df_psth.event>=10)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,1], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[1,1].title.set_text(p)

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='m')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WM')
                   & (df_psth.event>=10)]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,2], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[1,2].title.set_text(p)

data = df_psth.loc[df_psth.session_id.isin(['D0'])
                   & (df_psth.motivation=='um')
                   & (df_psth.cell_type.isin(['wM1','wS2']))
                   & (df_psth.time>=1)
                   & (df_psth.time<=2)
                   & (df_psth.reward_group=='R-')
                   & (df_psth.event_type=='WH')]
data = data.groupby(['mouse_id', 'session_id', 'reward_group','cell_type'], as_index=False).agg({'activity':np.nanmean})
sns.barplot(data=data, x='cell_type', y='activity', errorbar='se',
            hue='cell_type', ax=axes[1,3], palette=palette, hue_order=['wM1', 'wS2'], estimator=np.nanmean)
g1 = data.loc[data.cell_type=='wM1', 'activity'].to_numpy()
g1 = g1[~np.isnan(g1)]
g2 = data.loc[data.cell_type=='wS2', 'activity'].to_numpy()
g2 = g2[~np.isnan(g2)]
_, p = mannwhitneyu(g1,g2)
axes[1,3].title.set_text(p)

for ax in axes.flatten():
    ax.set_ylim(0, 0.06)
sns.despine()

 

# ################################
# Select whisker responsive cells.
# ################################

def select_responsive_cell_wilcoxon(data, thr=0.01, resp_win=slice(30,36,1), base_win=slice(24,30,1)):
    """_summary_

    Args:
        data (_type_): 6d numpy array of activity.
        resp_win (_type_, optional): _description_. Defaults to slice(30,36,1).
        base_win (_type_, optional): _description_. Defaults to slice(24,30,1).
    """
    
    samples = np.nanmax(data[:,:,:,:,:,resp_win], axis=5) - np.nanmax(data[:,:,:,:,:,base_win], axis=5)
    samples = np.concatenate([samples[:,i] for i in range(samples.shape[1])], axis=3)
    mask = np.full((samples.shape[:3]), False)

    for imouse in range(samples.shape[0]):
        for itype in range(samples.shape[1]):
            for icell in range(samples.shape[2]):
                if not np.all(np.isnan(samples[imouse,itype,icell])):
                    _, p = wilcoxon(samples[imouse,itype,icell], nan_policy='omit', alternative='two-sided')
                    if p <= thr:
                        mask[imouse,itype,icell] = True
    
    # Expend mask to the session dim.
    mask = np.repeat(mask[:,None], repeats=data.shape[1], axis=1)
    
    return mask


# Load numpy datasets.
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

# Baseline substraction.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Remove day-3 if exists.
if traces_rew.shape[1] > 5:
    traces_rew = traces_rew[:,1:]
    traces_non_rew = traces_non_rew[:,1:]

# Parameters.
RESP_WIN = slice(30,36,1)
BASE_WIN = slice(24,30,1)

responsive_mask_rew = select_responsive_cell_wilcoxon(traces_rew, thr=0.01, resp_win=RESP_WIN, base_win=BASE_WIN)
responsive_mask_non_rew = select_responsive_cell_wilcoxon(traces_non_rew, thr=0.01, resp_win=RESP_WIN, base_win=BASE_WIN)

# Save masks.
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_rew_common.npy')
np.save(save_path, responsive_mask_rew)
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_non_rew_common.npy')
np.save(save_path, responsive_mask_non_rew)


# PSTH analysis with responsive cells.
# ####################################

# Load numpy datasets.
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_non_rew_common_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_rew_common.npy')
np.load(read_path, allow_pickle=True)
read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_non_rew_common.npy')
np.load(read_path, allow_pickle=True)

traces_rew[~responsive_mask_rew] = np.nan
traces_non_rew[~responsive_mask_non_rew] = np.nan
df_psth_resp = convert_psth_np_to_pd(traces_rew, traces_non_rew)


# Count prop of responsive cells.
# -------------------------------

(~np.isnan(traces_rew[:,0,:,:,0,0])).sum() / 2628
(~np.isnan(traces_non_rew[:,0,:,:,0,0])).sum() / 1926


# Plot PSTH's.
# ------------

sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})

# all days
palette = sns.color_palette(['#238443', '#d51a1c'])
sns.relplot(data=df_psth, x='time', y='activity', errorbar='se', col='session_id',
            kind='line', hue='reward_group',
            hue_order=['R+','R-'], palette=palette)
plt.ylim([-0.005,0.05])





# #####################################################
# PSTH analysis with learning modulation indices (LMI).
# #####################################################

# Compute LMI.
# ############

read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_rew_GF.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\traces_non_motivated_trials_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\traces_non_motivated_trials_non_rew_GF.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\traces_non_motivated_trials_non_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

# read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_rew_GF.npy')
# traces_rew = np.load(read_path, allow_pickle=True)
# read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_rew_GF_metadata.pickle')
# with open(read_path, 'rb') as fid:
#     metadata_rew = pickle.load(fid)

# read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_non_rew_GF.npy')
# traces_non_rew = np.load(read_path, allow_pickle=True)
# read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_non_rew_GF_metadata.pickle')
# with open(read_path, 'rb') as fid:
#     metadata_non_rew = pickle.load(fid)

# Baseline substraction.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Remove day-3 if exists.
if traces_rew.shape[1] > 5:
    traces_rew = traces_rew[:,1:]
    traces_non_rew = traces_non_rew[:,1:]

# Parameters.
RESP_WIN = slice(30,36,1)

# Compute single trial responses.
resp_rew = np.nanmean(traces_rew[:,:,:,:,:,RESP_WIN], axis=5)
resp_non_rew = np.nanmean(traces_non_rew[:,:,:,:,:,RESP_WIN], axis=5)

# Compute modulation index.
lmi_rew, lmi_p_rew = compute_lmi(resp_rew, nshuffles=1000)
lmi_non_rew, lmi_p_non_rew = compute_lmi(resp_non_rew, nshuffles=1000)

# Save LMI.
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\lmi_rew.npy')
np.save(save_path, lmi_rew)
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\lmi_p_rew.npy')
np.save(save_path, lmi_p_rew)
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\lmi_non_rew.npy')
np.save(save_path, lmi_non_rew)
save_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\lmi_p_non_rew.npy')
np.save(save_path, lmi_p_non_rew)


# PSTH analysis for LMI groups.
# #############################

# Load calcium data.
read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_rew_GF.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)
read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_non_rew_GF.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
read_path = (r'E:\anthony\analysis\data_processed\traces_non_motivated_trials_non_rew_GF_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_non_rew = pickle.load(fid)

# Baseline substraction.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Remove D-3 if exists.
if traces_rew.shape[1] > 5:
    traces_rew = traces_rew[:,1:]
    traces_non_rew = traces_non_rew[:,1:]

# Load LMI groups.
read_path = (r'E:\anthony\analysis\data_processed\lmi_rew.npy')
lmi_rew = np.load(read_path, allow_pickle=True)
read_path = (r'E:\anthony\analysis\data_processed\lmi_p_rew.npy')
lmi_p_rew = np.load(read_path, allow_pickle=True)
read_path = (r'E:\anthony\analysis\data_processed\lmi_non_rew.npy')
lmi_non_rew = np.load(read_path, allow_pickle=True)
read_path = (r'E:\anthony\analysis\data_processed\lmi_p_non_rew.npy')
lmi_p_non_rew = np.load(read_path, allow_pickle=True)

# Load responsive cell masks.
read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_rew.npy')
responsive_mask_rew = np.load(read_path, allow_pickle=True)
read_path = (r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\responsive_mask_non_rew.npy')
responsive_mask_non_rew = np.load(read_path, allow_pickle=True)

# Mask non-responsive cells.
traces_rew[~responsive_mask_rew] = np.nan
traces_non_rew[~responsive_mask_non_rew] = np.nan

# Convert numpy arrays to pandas with LMI.
df_psth = convert_psth_np_to_pd(traces_rew, traces_non_rew, lmi_p_rew, lmi_p_non_rew)
df_psth.loc[(df_psth.mouse_id=='GF308') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='GF208') & (df_psth.session_id=='D-2'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='MI075') & (df_psth.session_id=='D-3'), 'activity'] = np.nan
df_psth.loc[(df_psth.mouse_id=='AR144') & (df_psth.session_id=='D-2'), 'activity'] = np.nan


# Proportion of cells in LMI groups.
# ----------------------------------

nmouse, ndays, ntypes, ncells = lmi_rew.shape
lmi_rew_prop = lmi_p_rew[:,0].reshape((lmi_rew.shape[0], -1))
lmi_rew_prop = [remove_nan_segments_from_axis(x, 0) for x in lmi_rew_prop]

pos_prop = [(x==1).sum() / x.size for x in lmi_rew_prop]
null_prop = [(x==0).sum() / x.size for x in lmi_rew_prop]
neg_prop = [(x==-1).sum() / x.size for x in lmi_rew_prop]

props = np.r_[pos_prop, null_prop, neg_prop]
x = np.r_[['Pos' for _ in range(len(pos_prop))],
          ['Null' for _ in range(len(null_prop))],
          ['Neg' for _ in range(len(neg_prop))]]
df_lmi_prop = pd.DataFrame(np.r_[props[np.newaxis],x[np.newaxis]].T, columns=['lmi_prop','lmi_group'])
df_lmi_prop = df_lmi_prop.astype({'lmi_prop': np.float64})
sns.scatterplot(data=df_lmi_prop, x='lmi_group', y='lmi_prop')

sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'font.sans-serif':'Arial'})
palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])


# PSTH across days and all cells.
# -------------------------------

# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#238443', '#d51a1c'])
sns.relplot(data=df_psth.loc[df_psth.lmi==1], x='time', y='activity', errorbar='se', col='session_id',
            kind='line', hue='reward_group',
            hue_order=['R+','R-'], palette=palette)
# plt.ylim([-0.005,0.05])

# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#238443', '#d51a1c'])
sns.relplot(data=df_psth.loc[(df_psth.lmi==-1) & (df_psth.reward_group=='R+')], x='time', y='activity', errorbar='se', col='session_id',
            kind='line', hue='reward_group',
            hue_order=['R+','R-'], palette=palette)
# plt.ylim([-0.005,0.05])



neg_roi = df_psth.loc[df_psth.lmi==-1][['mouse_id','roi']].drop_duplicates()

for mouse, roi in neg_roi:
    print(mouse)
    
    

# for projectors
# all days
# data = df_psth.loc[df_psth.session_id.isin(['D-1', 'D+2'])]
palette = sns.color_palette(['#3351ff', '#c959af'])
data = df_psth.loc[ (df_psth.cell_type.isin(['wM1','wS2']))]
data = data.loc[data.lmi.isin([1])]
fig = sns.relplot(data=data, x='time', y='activity', errorbar='se', row='reward_group', col='session_id',
            kind='line', palette=palette, hue='cell_type')
fig.set_titles(col_template='{col_name}')

lmi_p_rew.shape

d = np.copy(traces_rew)
d[lmi_p_rew!=-1] = np.nan

d = d[:,4]
d = np.nanmean(d, axis=3)
d.shape
d = d.reshape((-1, 121))
d = remove_nan_segments_from_axis(d, axis=0)
d.shape

for icell in range(523):
    plt.plot(d[icell] + icell * 0.1)
plt.axvline(30)

plt.figure()
plt.plot(d.mean(axis=0))

d = np.nanmean(d, axis=(1,2))

for imouse in range(15):
    plt.plot(d[imouse] + imouse * 0.05)
plt.axvline(30)