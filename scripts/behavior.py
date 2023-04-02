"""_summary_
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from src.utils import ci_bootstrap
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def plot_single_mouse_performance(data,mouse_id,palette,nmaxtrial=300):

    # Set plot parameters.
    raster_marker = 2
    marker_width = 2
    figsize = (20,6)
    block_size = 20
    is_rewarded = data.loc[data.mouse_id==mouse_id, 'reward_group'].iloc[0]
    color_wh = palette[2] if is_rewarded else palette[3]  # Green for whisker R+, red for R-.

    # Find ID of sessions to plot. Plot the last 6 pre-training sessions.
    days = [f'-{i}' for i in range(6,0,-1)] + [f'{i}' for i in range(0,6)]
    days = [iday for iday in days if iday in data.session_day.drop_duplicates().to_list()]
    f = lambda x: x.map({iday: iorder for iorder, iday in enumerate(days)})
    data = data.sort_values(by='session_day', key=f)
    sessions = data.loc[(data.mouse_id==mouse_id) & (data.session_day.isin(days)),'session_id']
    sessions = sessions.drop_duplicates().to_list()

    f, axes = plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 5]},figsize=figsize)
    plt.subplots_adjust(wspace=.3)

    # Plot performance to auditory sessions.
    d = data.loc[(data.session_id.isin(sessions)) & (data.trial<=nmaxtrial)]
    d = d.groupby(['session_id','session_day'], as_index=False).agg(np.mean)
    sns.lineplot(data=d, x='session_day', y='hr_c', color=palette[4], ax=axes[0],
                marker='o')
    sns.lineplot(data=d, x='session_day', y='hr_a', color=palette[0], ax=axes[0],
                marker='o')
    sns.lineplot(data=d, x='session_day', y='hr_w', color=color_wh, ax=axes[0],
                marker='o')

    # Plot performance to whisker day.
    d = data.loc[(data.mouse_id==mouse_id) & (data.session_day=='0') & (data.trial<=nmaxtrial)]
    # Select entries with trial numbers at middle of each block for alignment with
    # raster plot.
    d = d.sort_values(['trial'])
    d = d.loc[d.early_lick==0][int(block_size/2)::block_size]
    sns.lineplot(data=d,x='trial',y='hr_c',color=palette[4],ax=axes[1],
                marker='o')
    sns.lineplot(data=d,x='trial',y='hr_a',color=palette[0],ax=axes[1],
                marker='o')
    sns.lineplot(data=d,x='trial',y='hr_w',color=color_wh,ax=axes[1],
                marker='o')
    
    d = d = data.loc[(data.mouse_id==mouse_id) & (data.session_day=='0') & (data.trial<=nmaxtrial)]
    axes[1].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_c']-0.1,
                    color=palette[4],marker=raster_marker, linewidths=marker_width)
    axes[1].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_c']-1.1,
                    color='k',marker=raster_marker, linewidths=marker_width)

    axes[1].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_a']-0.15,
                    color=palette[1],marker=raster_marker, linewidths=marker_width)
    axes[1].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_a']-1.15,
                    color=palette[0],marker=raster_marker, linewidths=marker_width)

    axes[1].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_w']-0.2,
                    color=palette[3],marker=raster_marker, linewidths=marker_width)
    axes[1].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_w']-1.2,
                    color=palette[2],marker=raster_marker, linewidths=marker_width)

    axes[0].set_ylim([-0.2,1.05])
    axes[1].set_ylim([-0.2,1.05])
    axes[1].set_xticks(range(0,301,20))
    axes[0].set_xlabel('Pre-training days')
    axes[1].set_xlabel('Trial number')
    axes[0].set_ylabel('Lick probability')
    axes[1].set_ylabel('Lick probability')
    axes[0].set_title('Auditory pre-training')
    axes[1].set_title('Whisker learning day')
    f.suptitle(f'{mouse_id}')
    sns.despine()


# def plot_single_mouse_all_sessions(data,mouse_id,palette,nmaxtrial=300):

#     # Set plot parameters.
#     raster_marker = 2
#     marker_width = 2
#     figsize = (20,6)
#     block_size = 20
#     is_rewarded = data.loc[data.mouse_id==mouse_id, 'reward_group'].iloc[0]
#     color_wh = palette[2] if is_rewarded else palette[3]  # Green for whisker R+, red for R-.

#     # Find ID of all auditory and whisker sessions.
#     days = [f'A-{i}' for i in range(20,0,-1)] + [f'W{i}' for i in range(20)]
#     sessions = data.loc[(data.mouse_id==mouse_id) & (data.session_day.isin(days)), 'session_id']
#     sessions = sessions.drop_duplicates().to_list()

#     f, axes = plt.subplots(len(sessions),1, figsize=(8.27, 11.69))

#     for isession, iax in enumerate(axes):

#         df = data.loc[(data.session_id==sessions[isession])]
#         # Select entries with trial numbers at middle of each block for alignment with
#         # raster plot.
#         df = df.loc[~df.trial.isna()][int(block_size/2)::block_size]
#         sns.lineplot(data=df,x='trial',y='hr_c',color=palette[4],ax=iax,
#                     marker='o')
#         sns.lineplot(data=df,x='trial',y='hr_a',color=palette[0],ax=iax,
#                     marker='o')
#         if df.session_day.iloc[0] in [f'W{i}' for i in range(20)]:
#             sns.lineplot(data=df,x='trial',y='hr_w',color=color_wh,ax=iax,
#                         marker='o')

#         df = data.loc[data.session_id==sessions[isession]]
#         iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_c']-0.1,
#                         color=palette[4],marker=raster_marker, linewidths=marker_width)
#         iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_c']-1.1,
#                         color='k',marker=raster_marker, linewidths=marker_width)

#         iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_a']-0.15,
#                         color=palette[1],marker=raster_marker, linewidths=marker_width)
#         iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_a']-1.15,
#                         color=palette[0],marker=raster_marker, linewidths=marker_width)
                    
#         if df.session_day.iloc[0] in [f'W{i}' for i in range(20)]:
#             iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_w']-0.2,
#                             color=palette[3],marker=raster_marker, linewidths=marker_width)
#             iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_w']-1.2,
#                             color=palette[2],marker=raster_marker, linewidths=marker_width)

#         iax.set_ylim([-0.2,1.05])
#         iax.set_xlabel('Trial number')
#         iax.set_ylabel('Lick probability')
#         sns.despine()
#         plt.tight_layout()
    
#     f.suptitle(f'{mouse_id}')


def write_result_df(sessions_db_path, data_folder_path):
    """Generates result dataframe from sessions result text files.
    Requires an excel session database.

    Args:
        sessions_db_path (string): Path to session database.
        data_folder_path (string): Path to server data folder.

    Returns:
        DataFrame: Result dataframe for all sessions in database.
    """
    sessions = pd.read_excel(sessions_db_path, dtype={'session_type':object})
    sessions = sessions.dropna(axis=0,how='all')
    data = []
    for _, isession in sessions.iterrows():
        mouse_id, session_folder, session_id, session_day, reward_group, weight_pre, weight_post, reward_amount, batch_id = isession
        if pd.isna(session_folder):
            session_folder = ''
        df = pd.read_csv(os.path.join(data_folder_path, mouse_id, session_folder, session_id, 'Results.txt'), sep=r'\s+', engine='python')
        df['mouse_id'] = mouse_id
        df['session_id'] = session_id
        df['session_day'] = str(session_day)
        df['reward_group'] = bool(reward_group)
        df['weight_pre'] = weight_pre
        df['weight_post'] = weight_post
        df['reward_amount'] = reward_amount
        df['batch_id'] = int(batch_id)
        # match_str = re.search(r'\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}',session_id).group()
        # df['date_time'] = pd.to_datetime(match_str, format='%Y%m%d_%H%M%S')
        
            # Rename columns for compatibility with old data.
        df = df.rename(columns={
            'trialnumber': 'trial_number',
            'Perf': 'perf',
            'TrialTime': 'trial_time',
            'Association': 'association_flag',
            'Quietwindow': 'quiet_window',
            'ITI': 'iti',
            'Stim/NoStim': 'is_stim',
            'Whisker/NoWhisker': 'is_whisker',
            'Auditory/NoAuditory': 'is_auditory',
            'Lick': 'lick_flag',
            'ReactionTime': 'reaction_time',
            'WhStimDuration':'wh_stim_duration',
            'WhStimAmp': 'wh_stim_amp',
            'WhRew': 'wh_reward',
            'Rew/NoRew': 'is_reward',
            'AudDur': 'aud_stim_duration',
            'AudDAmp': 'aud_stim_amp',
            'AudFreq': 'aud_stim_freq',
            'AudRew': 'aud_reward',
            'EarlyLick': 'early_lick',
            'Lick/NoLight': 'is_light',
            'LightAmp': 'light_amp',
            'LightDur': 'light_duration',
            'LightFreq': 'light_freq',
            'LightPreStim': 'light_prestim'
        })
        data.append(df)
    data = pd.concat(data)

    # Performance column code:
    # 0: whisker miss
    # 1: auditory miss
    # 2: whisker hit
    # 3: auditory hit
    # 4: correct rejection
    # 5: false alarm
    # 6: early lick

    map_hits = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 0,
        5: 1,
        6: np.nan,
    }
    data['outcome'] = data['perf'].map(map_hits)
    # data = data.sort_values(['mouse_id','session_id','trial_number']).reset_index(drop=True)

    return data


sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)

SESSIONS_DB_PATH = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs\\behavior_sessions.xlsx'
# SESSIONS_DB_PATH = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs\\behavior_Meriam.xlsx'
DATA_FOLDER_PATH = 'M:\\data'
PALETTE = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#666666']
PALETTE = sns.color_palette(PALETTE)

# Read behavior results.
df_result = write_result_df(SESSIONS_DB_PATH, DATA_FOLDER_PATH)


# Compute performance.
# --------------------

BLOCK_SIZE = 20  # Block size for hit rate computation.

# Exclude early licks.
df_result = df_result.loc[df_result.early_lick==0]
df_result = df_result.reset_index(drop=True)


# Add trial number excluding early lick trials.
#
# In case there exist aborted trials due to early lick during baseline before
# stimulus delivery. Although early licks are made obsolete with continuous
# aquisition, this is kept for compatibility with older data.
df_result = df_result.sort_values(['mouse_id','session_id','trial_number'])
df_result['trial'] = df_result.loc[(df_result.early_lick==0)].groupby('session_id').cumcount()
# Add trial number for each stimulus.
df_result['trial_w'] = df_result.loc[(df_result.early_lick==0) & (df_result.is_whisker==1.0)]\
                                .groupby('session_id').cumcount()
df_result['trial_a'] = df_result.loc[(df_result.early_lick==0) & (df_result.is_auditory==1.0)]\
                                .groupby('session_id').cumcount()
df_result['trial_c'] = df_result.loc[(df_result.early_lick==0) & (df_result.is_stim==0.0)]\
                                .groupby('session_id').cumcount()

# Add performance outcome column for each stimulus.
df_result['outcome_w'] = df_result.loc[(df_result.is_whisker==1.0)]['outcome']
df_result['outcome_a'] = df_result.loc[(df_result.is_auditory==1.0)]['outcome']
df_result['outcome_c'] = df_result.loc[(df_result.is_stim==0.0)]['outcome']

# Cumulative sum performance representation.
df_result['cumsum_w'] = df_result.loc[(df_result.is_whisker==1.0)]['outcome_w']\
                                 .map({0:-1,1:1,np.nan:0})
df_result['cumsum_w'] = df_result.groupby('session_id')['cumsum_w'].transform(np.cumsum)
df_result['cumsum_a'] = df_result.loc[(df_result.is_auditory==1.0)]['outcome_a']\
                                 .map({0:-1,1:1,np.nan:0})
df_result['cumsum_a'] = df_result.groupby('session_id')['cumsum_a'].transform(np.cumsum)
df_result['cumsum_c'] = df_result.loc[(df_result.is_stim==0.0)]['outcome_c']\
                                 .map({0:-1,1:1,np.nan:0})
df_result['cumsum_c'] = df_result.groupby('session_id')['cumsum_c'].transform(np.cumsum)

# Add block index of n=BLOCK_SIZE trials and compute hit rate on them.
df_result['block'] = df_result.loc[df_result.early_lick==0, 'trial'].transform(lambda x: x // BLOCK_SIZE)
# Compute hit rates. Use transform to propagate hit rate to all entries.
df_result['hr_w'] = df_result.groupby(['session_id', 'block'], as_index=False)['outcome_w']\
                             .transform(np.nanmean)
df_result['hr_a'] = df_result.groupby(['session_id', 'block'], as_index=False)['outcome_a']\
                             .transform(np.nanmean)
df_result['hr_c'] = df_result.groupby(['session_id', 'block'], as_index=False)['outcome_c']\
                             .transform(np.nanmean)

# Also store performance in new data frame with block as time steps for convenience.
cols = ['session_id','mouse_id','session_day','reward_group','block']
df_block_perf = df_result.loc[df_result.early_lick==0, cols].drop_duplicates()
df_block_perf = df_block_perf.reset_index()
# Compute hit rates.
df_block_perf['hr_w'] = df_result.groupby(['session_id', 'block'], as_index=False)\
                                 .agg(np.nanmean)['outcome_w']
df_block_perf['hr_a'] = df_result.groupby(['session_id', 'block'], as_index=False)\
                                 .agg(np.nanmean)['outcome_a']
df_block_perf['hr_c'] = df_result.groupby(['session_id', 'block'], as_index=False)\
                                 .agg(np.nanmean)['outcome_c']


def plot_average_across_days(data,mice,reward_group,palette,nmax_trials=200,ax=None):

    data = data.loc[(data.mouse_id.isin(mice)) & (data.trial <=nmax_trials) & (data.reward_group==reward_group)]
    data = data.groupby(['mouse_id','session_id','session_day'], as_index=False).agg(np.mean)
    # Sort dataframe by days.
    days = [f'-{i}' for i in range(4,0,-1)] + [f'{i}' for i in range(3)]
    data = data.loc[data.session_day.isin(days)]
    days = [iday for iday in days if iday in data.session_day.drop_duplicates().to_list()]
    f = lambda x: x.map({iday: iorder for iorder, iday in enumerate(days)})
    data = data.sort_values(by='session_day', key=f)
    
    sns.lineplot(data=data, x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_c', order=days, join=False, color=palette[4], ax=ax)

    sns.lineplot(data=data, x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_a', order=days, join=False, color=palette[0], ax=ax)

    color_wh = palette[2] if reward_group else palette[3]
    sns.lineplot(data=data, x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_w', order=days, join=False, color=color_wh, ax=ax)

    ax.set_yticks([i*0.2 for i in range(6)])
    ax.set_xlabel('Training days')
    ax.set_ylabel('Lick probability')


def plot_perf_across_blocks(data,mice,reward_group,palette,nmax_blocks=None,ax=None):
    data = data.loc[(data.mouse_id.isin(mice)) & (data.session_day=='0') & (data.early_lick==0)]
    data = data.groupby(['mouse_id','session_id','session_day', 'block'], as_index=False).agg(np.mean)

    sns.lineplot(data=data, x='block', y='hr_c', estimator=np.mean, color=palette[4] ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)
    sns.lineplot(data=data, x='block', y='hr_a', estimator=np.mean, color=palette[0] ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)
    color_wh = palette[2] if reward_group else palette[3]
    sns.lineplot(data=data, x='block', y='hr_w', estimator=np.mean, color=color_wh ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)
    
    nblocks = data.block.max()
    if np.isnan(nblocks):
        nblocks = 0
    ax.set_xticks(range(nblocks))
    ax.set_ylim([0,1.1])
    ax.set_yticks([i*0.2 for i in range(6)])
    ax.set_xlabel('Block (20 trials)')
    ax.set_ylabel('Lick probability')



batch = 2
palette = PALETTE
NMAX_TRIAL = 300
NAMX_BLOCK = None
pdf_path = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior'
pdf_fname = f'summary_batch_{int(batch):03d}.pdf'
mice = df_result.loc[(df_result.batch_id==batch), 'mouse_id']
mice = mice.drop_duplicates().to_list()
# mice.remove('AR040')

pdf=PdfPages(os.path.join(pdf_path,pdf_fname))

# Plot average performance across mice and days.
f, axes = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(10,6))
plt.subplots_adjust(wspace=.4, hspace=0.6)

REWARD_GROUP = True
data = df_result.loc[(df_result.mouse_id.isin(mice)) & (df_result.reward_group==REWARD_GROUP)]
plot_average_across_days(data, mice, REWARD_GROUP, PALETTE, ax=axes[0,0])
plot_perf_across_blocks(data, mice, REWARD_GROUP, PALETTE, ax=axes[0,1])

REWARD_GROUP = False
data = df_result.loc[(df_result.mouse_id.isin(mice)) & (df_result.reward_group==REWARD_GROUP)]
plot_average_across_days(data, mice, REWARD_GROUP, PALETTE, ax=axes[1,0])
plot_perf_across_blocks(data, mice, REWARD_GROUP, PALETTE, ax=axes[1,1])
sns.despine(fig=f)

pdf.savefig()
plt.close() 

for imouse in mice:
    plot_single_mouse_performance(df_result, imouse, PALETTE)
    pdf.savefig()
    plt.close()
    
pdf.close()



# Plotting performance.
batches = [2,3]
palette = PALETTE
NMAX_TRIAL = 300
NAMX_BLOCK = None
mice = df_result.loc[(df_result.batch_id.isin(batches)), 'mouse_id']
mice = mice.drop_duplicates().to_list()
# mice = [imouse for imouse in mice if imouse not in ['AR040','AR028','AR027']]

# Plot average performance across mice and days.
f, axes = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(10,6))
plt.subplots_adjust(wspace=.4, hspace=0.6)

REWARD_GROUP = True
data = df_result.loc[(df_result.mouse_id.isin(mice)) & (df_result.reward_group==REWARD_GROUP)]
plot_average_across_days(data, mice, REWARD_GROUP, PALETTE, ax=axes[0,0])
plot_perf_across_blocks(data, mice, REWARD_GROUP, PALETTE, ax=axes[0,1])

REWARD_GROUP = False
data = df_result.loc[(df_result.mouse_id.isin(mice)) & (df_result.reward_group==REWARD_GROUP)]
plot_average_across_days(data, mice, REWARD_GROUP, PALETTE, ax=axes[1,0])
plot_perf_across_blocks(data, mice, REWARD_GROUP, PALETTE, ax=axes[1,1])
sns.despine(fig=f)

fname = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior\\average_performance\\batch_2_3.svg'
plt.savefig(fname, dpi=300, transparent=True)

for imouse in mice:
    plot_single_mouse_performance(df_result, imouse, PALETTE)
    batch = df_result.loc[df_result.mouse_id==imouse, 'batch_id'].iloc[0]
    plt.suptitle(f'{imouse} -- Batch {batch}')
    plt.subplots_adjust(top=0.8)
    fname = f'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior\\single_mouse_performance\\{imouse}.png'
    plt.savefig(fname, dpi=300, transparent=True)   




# Plot results from particle test.

batches = [2]
mice = df_result.loc[(df_result.batch_id.isin(batches)), 'mouse_id']
mice = mice.drop_duplicates().to_list()
mice = [imouse for imouse in mice if imouse not in []]
reward_group = True
nmax_trials = 300
palette = PALETTE
ax = None

data = df_result.loc[(df_result.mouse_id.isin(mice)) & (df_result.trial <=nmax_trials) & (df_result.reward_group==reward_group)]
data = data.groupby(['mouse_id','session_id','session_day'], as_index=False).agg(np.mean)
# Sort dataframe by days.
days = ['P_ON1','P_OFF1', 'P_ON2']
data = data.loc[data.session_day.isin(days)]
data = data.sort_values(by='session_day', key=lambda x: x.map({'P_ON1':0,'P_OFF1':1,'P_ON2':2}))

for imouse in mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
    
# sns.lineplot(data=data, x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
sns.pointplot(data=data, x='session_day', y='hr_c', order=days, join=False, color=palette[4], ax=ax)

# sns.lineplot(data=data, x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
sns.pointplot(data=data, x='session_day', y='hr_a', order=days, join=False, color=palette[0], ax=ax)

color_wh = palette[2] if reward_group else palette[3]
# sns.lineplot(data=data, x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
sns.pointplot(data=data, x='session_day', y='hr_w', order=days, join=False, color=color_wh, ax=ax)

sns.despine()
ax = plt.gca()
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xticklabels(['ON', 'OFF', 'ON'])
ax.set_xlabel('Particle test')
ax.set_ylabel('Lick probability')





    
    # for it, isessions in enumerate(sessions):
    #     d = data.loc[data.session_id.isin(isessions)]
    #     d = d.loc[d.trial < nmax_trials]
    #     d = d.groupby(['mouse_id','session_id']).agg(np.nanmean)
    #     d['group'] = gp_labels[it]
    #     df.append(d)
    # df = pd.concat(df).reset_index()

    # color_wh = palette[2] if rewarded else palette[3]

    # plt.figure(figsize=(10,8))
    # sns.lineplot(data=df,x='group',y='outcome_c',color=palette[4], style='mouse_id',
    #              estimator=None, dashes=False, legend=False,alpha=.5)
    # sns.scatterplot(data=df,x='group',y='outcome_c',alpha=.5,color=palette[4])
    # sns.pointplot(data=df,x='group',y='outcome_c',color=palette[4], join=False)

    # sns.lineplot(data=df,x='group',y='outcome_a',color=palette[0], style='mouse_id',
    #              estimator=None, dashes=False, legend=False,alpha=.5)
    # sns.scatterplot(data=df,x='group',y='outcome_a',alpha=.5,color=palette[0])
    # sns.pointplot(data=df,x='group',y='outcome_a',color=palette[0], join=False)

    # sns.lineplot(data=df,x='group',y='outcome_w',color=color_wh, style='mouse_id',
    #              estimator=None, dashes=False, legend=False,alpha=.5)
    # sns.scatterplot(data=df,x='group',y='outcome_w',alpha=.5,color=color_wh)
    # sns.pointplot(data=df,x='group',y='outcome_w',color=color_wh, join=False)

    # plt.ylim([0,1.1])
    # plt.xlabel('Particle test')
    # plt.ylabel('Lick probability')
    # sns.despine()
    # plt.show()


def perf_stats(data,g1_sessions,g2_sessions,g1_label,g2_label,nmax_trials,test,alternative='greater'):

    d_g1 = df_result.loc[df_result.session_id.isin(g1_sessions)]
    d_g1 = d_g1.loc[d_g1.trial < nmax_trials]
    d_g1 = d_g1.groupby(['mouse_id','session_id']).agg(np.nanmean)
    d_g1['group'] = g1_label

    d_g2 = df_result.loc[df_result.session_id.isin(g2_sessions)]
    d_g2 = d_g2.loc[d_g2.trial < nmax_trials]
    d_g2 = d_g2.groupby(['mouse_id','session_id']).agg(np.nanmean)
    d_g2['group'] = g2_label

    g1 = d_g1.outcome_w.to_numpy()
    g2 = d_g2.outcome_w.to_numpy()
    ci1_left, ci1_right = ci_bootstrap(g1)
    ci2_left, ci2_right = ci_bootstrap(g2)

    if test == 'wilcoxon':
        _, pval = stats.wilcoxon(g1,g2,alternative=alternative)
    elif test == 'mann_whitney':
        _, pval = stats.mannwhitneyu(g1,g2,alternative=alternative)

    stats_strg = f'''mean {g1.mean():.4f} (95% ci: {ci1_left:.4f}, {ci1_right:.4f})\n
                  \mean {g2.mean():.4f} (95% ci: {ci2_left:.4f}, {ci2_right:.4f})\n
                  pval: {pval:.4f}'''
    print(stats_strg)


# TODO: make sure ordering of sessions is correct
# TODO: remove the rewarded parameter in function, guess it from df
# TODO: improve stat function, currently I'm changing which trial type manually

A2_R = df_result.loc[(df_result.reward_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'A-2')]['session_id'].drop_duplicates()
A1_R = df_result.loc[(df_result.reward_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'A-1')]['session_id'].drop_duplicates()
A2_NR = df_result.loc[(df_result.reward_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'A-2')]['session_id'].drop_duplicates()
A1_NR = df_result.loc[(df_result.reward_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'A-1')]['session_id'].drop_duplicates()
W1_R = df_result.loc[(df_result.reward_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'W1')]['session_id'].drop_duplicates()
W1_NR = df_result.loc[(df_result.reward_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'W1')]['session_id'].drop_duplicates()
P_ON1 = df_result.loc[(df_result.batch_id==2) & (df_result.session_day == 'P_ON1')]['session_id'].drop_duplicates()
P_OFF1 = df_result.loc[(df_result.batch_id==2) & (df_result.session_day == 'P_OFF1')]['session_id'].drop_duplicates()
P_ON2 = df_result.loc[(df_result.batch_id==2) & (df_result.session_day == 'P_ON2')]['session_id'].drop_duplicates()


session_lists = [A2_R, A1_R, W1_R]
labels = ['D-2','D-1','D0']
plot_average_perf(df_result, session_lists, labels, True,300)

session_lists = [A2_NR, A1_NR, W1_NR]
labels = ['D-2','D-1','D0']
plot_average_perf(df_result, session_lists, labels, False,300)

session_lists = [P_ON1,P_OFF1,P_ON2]
labels = ['ON pre','OFF','ON post']
plot_average_perf(df_result, session_lists, labels, True,300)

perf_stats(df_result,P_ON1,P_OFF1,'P_ON','P_OFF',300,test='wilcoxon',alternative='greater')
perf_stats(df_result,P_ON2,P_OFF1,'P_ON','P_OFF',300,test='wilcoxon',alternative='greater')

perf_stats(df_result,W1,A1,'W1','A1',250,test='wilcoxon',alternative='greater')

perf_stats(df_result,W1,CNO1_S1,'W1','CNO1_S1',250,test='mann_whitney',alternative='greater')
perf_stats(df_result,W1,CNO1_S2p_S1,'W1','CNO1_S2',250,test='mann_whitney',alternative='greater')

perf_stats(df_result,P_ON2,P_OFF1,'SAL2','CNO1',250,test='mann_whitney',alternative='greater')
perf_stats(df_result,W3,CNO1_S2p_S1,'SAL2','CNO1',250,test='mann_whitney',alternative='greater')



session_lists = [A1_S2p_S1, A2_S2p_S1, CNO1_S2p_S1, CNO2_S2p_S1, CNO3_S2p_S1, SAL1_S2p_S1, SAL2_S2p_S1]
labels = ['A1', 'A2', 'CNO1', 'CNO2', 'CNO3', 'SAL1', 'SAL2']
plot_average_perf(df_result, session_lists, labels, 100)
plt.title('wS2 projecting neurons inactivation')

session_lists = [A1_S1, A2_S1, CNO1_S1, CNO2_S1, CNO3_S1, SAL1_S1, SAL2_S1]
labels = ['A1', 'A2', 'CNO1', 'CNO2', 'CNO3', 'SAL1', 'SAL2']
plot_average_perf(df_result, session_lists, labels, 100)
plt.title('wS1 inactivation')
