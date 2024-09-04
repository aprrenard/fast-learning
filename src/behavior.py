"""This script contains functions to analyze behavior data.
It assumes that a excel sheet contains sessions metadata
and that data is stored on the server with the usual folder structure.
This code will be refactored to read behavioral data from NWB files.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from src.utils import utils_io as io
from nwb_wrappers import nwb_reader_functions as nwb_read

# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def compute_performance(table, session_id, db_path, block_size=20):
    
    table['session_id'] = session_id
    table['mouse_id'] = session_id[:5]
    # Get reward group from session metadata.
    reward_group = io.get_reward_group_from_db(db_path, session_id)
    table['reward_group'] = reward_group

    # Add trial number for each stimulus.
    table['trial_w'] = table.loc[(table.early_lick==0) & (table.whisker_stim==1.0)]\
                                    .groupby('session_id').cumcount()+1
    table['trial_a'] = table.loc[(table.early_lick==0) & (table.auditory_stim==1.0)]\
                                    .groupby('session_id').cumcount()+1
    table['trial_c'] = table.loc[(table.early_lick==0) & (table.no_stim==0.0)]\
                                    .groupby('session_id').cumcount()+1

    # Add performance outcome column for each stimulus.
    table['outcome_w'] = table.loc[(table.whisker_stim==1.0)]['lick_flag']
    table['outcome_a'] = table.loc[(table.auditory_stim==1.0)]['lick_flag']
    table['outcome_c'] = table.loc[(table.no_stim==1.0)]['lick_flag']

    # Cumulative sum performance representation.
    table['cumsum_w'] = table.loc[(table.whisker_stim==1.0)]['outcome_w']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_w'] = table.groupby('session_id')['cumsum_w'].transform('cumsum')
    table['cumsum_a'] = table.loc[(table.auditory_stim==1.0)]['outcome_a']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_a'] = table.groupby('session_id')['cumsum_a'].transform('cumsum')
    table['cumsum_c'] = table.loc[(table.no_stim==1.0)]['outcome_c']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_c'] = table.groupby('session_id')['cumsum_c'].transform('cumsum')


    # Add block index of n=BLOCK_SIZE trials and compute hit rate on them.
    table['block_id'] = table.loc[table.early_lick==0, 'trial_id'].transform(lambda x: x // block_size)
    # Compute hit rates. Use transform to propagate hit rate to all entries.
    table['hr_w'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_w']\
                                .transform('mean')
    table['hr_a'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_a']\
                                .transform('mean')
    table['hr_c'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_c']\
                                .transform('mean')

    # # Also store performance in new data frame with block as time steps for convenience.
    # cols = ['session_id','mouse_id','session_day','reward_group','block_id']
    # df_block_perf = table.loc[table.early_lick==0, cols].drop_duplicates()
    # df_block_perf = df_block_perf.reset_index()
    # # Compute hit rates.
    # df_block_perf['hr_w'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_w']
    # df_block_perf['hr_a'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_a']
    # df_block_perf['hr_c'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_c']

    return table

def plot_single_mouse_performance_across_days(df,mouse_id,color_palette):
    """ Plots performance of a single mouse across all auditory and whisker sessions.

    Args:
        df (_type_): _description_
        mouse_id (_type_): _description_
        color_palette (_type_): _description_
    """

    # Set plot parameters.
    raster_marker = 2
    marker_width = 2
    block_size = 20
    reward_group = df.loc[df.mouse_id==mouse_id, 'reward_group'].iloc[0]
    color_wh = color_palette[2]
    if reward_group == 'R-':  # green for whisker R+, red for R-.
        color_wh = color_palette[3]

    # Find ID of sessions to plot.
    # Start with more days than needed in the right order and filter out.
    days = [f'-{i}' for i in range(30,0,-1)] + ['0'] + [f'+{i}' for i in range(1,30)]
    days  = [iday for iday in days if iday in df.session_day.drop_duplicates().to_list()]
    # Order dataframe by days.
    f = lambda x: x.map({iday: iorder for iorder, iday in enumerate(days)})
    df = df.sort_values(by='session_day', key=f)
    sessions = df.loc[(df.mouse_id==mouse_id) & (df.session_day.isin(days)),'session_id']
    sessions = sessions.drop_duplicates().to_list()
    
    # Initialize figure.
    # f, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]}, figsize=(20,6))
    fig = plt.figure(figsize=(4,6))
    ax = plt.gca()

    # Plot performance.
    d = df.loc[(df.session_id.isin(sessions))]
    d = d.groupby(['session_id','session_day'], as_index=False).agg(np.mean)
    d = d.sort_values(by='session_day', key=f)
    sns.lineplot(data=d, x='session_day', y='hr_c', color=color_palette[5], ax=ax,
                 marker='o')
    sns.lineplot(data=d, x='session_day', y='hr_a', color=color_palette[0], ax=ax,
                 marker='o')
    if reward_group in ['R+', 'R-']:
        sns.lineplot(data=d, x='session_day', y='hr_w', color=color_wh, ax=ax,
                     marker='o')

    ax.set_ylim([-0.2,1.05])
    ax.set_xlabel('Performance across training days')
    ax.set_ylabel('Lick probability')
    ax.set_title('Training days')
    fig.suptitle(f'{mouse_id}')
    sns.despine()


def plot_single_session(table, session_id, ax=None):
    """ Plots performance of a single session with average per block and single tri al raster plot.

    Args:
        table (_type_): _description_
        mouse_id (_type_): _description_
        palette (_type_): _description_
    """
    
    # Set plot parameters.
    raster_marker = 2
    marker_width = 2
    figsize = (15,6)
    block_size = 20
    reward_group = table.loc[table.session_id==session_id, 'reward_group'].iloc[0]
    palette = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc', '#333333']
    color_wh = palette[2]
    if reward_group == 'R-':  # #238443 for whisker R+, #d51a1c for R-.
        color_wh = palette[3]    

    # Initialize figure.
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = plt.gca()

    # Plot performance of the session across time.
    d = table.loc[table.session_id==session_id]
    # Select entries with trial numbers at middle of each block for alignment with
    # raster plot.
    d = d.sort_values(['trial_id'])
    d = d.loc[d.early_lick==0][int(block_size/2)::block_size]
    if not d.hr_c.isna().all():
        sns.lineplot(data=d,x='trial_id',y='hr_c',color=palette[5],ax=ax,
                    marker='o')
    if not d.hr_a.isna().all():
        sns.lineplot(data=d,x='trial_id',y='hr_a',color=palette[0],ax=ax,
                    marker='o')
    if not d.hr_w.isna().all():
        sns.lineplot(data=d,x='trial_id',y='hr_w',color=color_wh,ax=ax,
                     marker='o')

    # Plot single trial raster plot.
    d = table.loc[table.session_id==session_id]
    if not d.hr_c.isna().all():
        ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'],y=d.loc[d.lick_flag==0]['outcome_c']-0.1,
                        color=palette[4],marker=raster_marker, linewidths=marker_width)
        ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'],y=d.loc[d.lick_flag==1]['outcome_c']-1.1,
                        color=palette[5],marker=raster_marker, linewidths=marker_width)

    if not d.hr_a.isna().all():
        ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'],y=d.loc[d.lick_flag==0]['outcome_a']-0.15,
                        color=palette[1],marker=raster_marker, linewidths=marker_width)
        ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'],y=d.loc[d.lick_flag==1]['outcome_a']-1.15,
                        color=palette[0],marker=raster_marker, linewidths=marker_width)

    if not d.hr_w.isna().all():
        ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'],y=d.loc[d.lick_flag==0]['outcome_w']-0.2,
                        color=palette[3],marker=raster_marker, linewidths=marker_width)
        ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'],y=d.loc[d.lick_flag==1]['outcome_w']-1.2,
                        color=palette[2],marker=raster_marker, linewidths=marker_width)

    nmax_trials = d.trial_id.max()
    ax.set_ylim([-0.2,1.05])
    ax.set_xticks(range(0,nmax_trials + nmax_trials%20,20))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Lick probability')
    ax.set_title(f'{session_id}')
    sns.despine()
    plt.show()


def plot_average_across_days(data,mice,reward_group,palette,nmax_trials=300,ax=None):

    data = data.loc[(data.mouse_id.isin(mice)) & (data.trial <=nmax_trials) & (data.reward_group==reward_group)]
    data = data.groupby(['mouse_id','session_id','session_day'], as_index=False).agg(np.mean)
    # Sort dataframe by days.
    days = [f'-{i}' for i in range(4,0,-1)] + ['0'] + [f'+{i}' for i in range(1,6)]
    data = data.loc[data.session_day.isin(days)]
    days = [iday for iday in days if iday in data.session_day.drop_duplicates().to_list()]
    f = lambda x: x.map({iday: iorder for iorder, iday in enumerate(days)})
    data = data.sort_values(by='session_day', key=f)
    sns.lineplot(data=data, x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_c', order=days, join=False, color=palette[4], ax=ax)

    sns.lineplot(data=data, x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_a', order=days, join=False, color=palette[0], ax=ax)

    color_wh = palette[2]
    if reward_group=='R-':
        color_wh = palette[3]
    sns.lineplot(data=data, x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    # sns.pointplot(data=data, x='session_day', y='hr_w', order=days, join=False, color=color_wh, ax=ax)

    ax.set_yticks([i*0.2 for i in range(6)])
    ax.set_xlabel('Training days')
    ax.set_ylabel('Lick probability')


def plot_perf_across_blocks(data,mice,reward_group,palette,nmax_trials=300,ax=None):
    data = data.loc[(data.mouse_id.isin(mice)) & (data.session_day=='0') & (data.trial <=nmax_trials) & (data.early_lick==0)]
    data = data.groupby(['mouse_id','session_id','session_day', 'block'], as_index=False).agg(np.mean)

    sns.lineplot(data=data, x='block', y='hr_c', estimator=np.mean, color=palette[4] ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)
    sns.lineplot(data=data, x='block', y='hr_a', estimator=np.mean, color=palette[0] ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)
    color_wh = palette[2]
    if reward_group=='R-':
        color_wh = palette[3]
    sns.lineplot(data=data, x='block', y='hr_w', estimator=np.mean, color=color_wh ,alpha=1, legend=False, marker='o', errorbar='ci', err_style='band', ax=ax)

    nblocks = data.block.max()
    if np.isnan(nblocks):
        nblocks = 0
    ax.set_xticks(range(nblocks))
    ax.set_ylim([0,1.1])
    ax.set_yticks([i*0.2 for i in range(6)])
    ax.set_xlabel('Block (20 trials)')
    ax.set_ylabel('Lick probability')


# data = table
# mouse_id = 'AR071'
# palette = PALETTE

# def plot_single_mouse_all_sessions(data,mouse_id,palette):

#     # Set plot parameters.
#     raster_marker = 2
#     marker_width = 1.5
#     figsize = (18,6)
#     block_size = 20
#     reward_group = data.loc[data.mouse_id==mouse_id, 'reward_group'].iloc[0]
#     color_wh = palette[2]
#     if reward_group == 'R-':  # green for whisker R+, red for R-.
#         color_wh = palette[3]

#     # Find ID of all auditory and whisker sessions.
#     days = [f'-{i}' for i in range(20,0,-1)] + ['0'] + [f'+{i}' for i in range(1,20)]
#     sessions = data.loc[(data.mouse_id==mouse_id) & (data.session_day.isin(days)), 'session_id']
#     sessions = sessions.drop_duplicates().to_list()

#     pdf_path = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior\\single_mouse_performance'
#     pdf_fname = 'AR071.pdf'
#     pdf = PdfPages(os.path.join(pdf_path,pdf_fname))

#     for isession, _ in enumerate(sessions):
#         f = plt.figure(figsize=figsize)
#         iax = plt.gca()
#         df = data.loc[(data.session_id==sessions[isession])]
#         # Select entries with trial numbers at middle of each block for alignment with
#         # raster plot.
#         df = df.loc[~df.trial.isna()][int(block_size/2)::block_size]
#         sns.lineplot(data=df,x='trial',y='hr_c',color='#333333',ax=iax,
#                     marker='o')
#         sns.lineplot(data=df,x='trial',y='hr_a',color=palette[0],ax=iax,
#                     marker='o')
#         if df.session_day.iloc[0] in ['0'] + [f'+{i}' for i in range(1,20)]:
#             sns.lineplot(data=df,x='trial',y='hr_w',color=color_wh,ax=iax,
#                         marker='o')

#         df = data.loc[data.session_id==sessions[isession]]
#         iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_c']-0.1,
#                         color=palette[4],marker=raster_marker, linewidths=marker_width)
#         iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_c']-1.1,
#                         color='#333333',marker=raster_marker, linewidths=marker_width)

#         iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_a']-0.15,
#                         color=palette[1],marker=raster_marker, linewidths=marker_width)
#         iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_a']-1.15,
#                         color=palette[0],marker=raster_marker, linewidths=marker_width)

#         if df.session_day.iloc[0] in ['0'] + [f'+{i}' for i in range(1,20)]:
#             iax.scatter(x=df.loc[df.outcome==0]['trial'],y=df.loc[df.outcome==0]['outcome_w']-0.2,
#                             color=palette[3],marker=raster_marker, linewidths=marker_width)
#             iax.scatter(x=df.loc[df.outcome==1]['trial'],y=df.loc[df.outcome==1]['outcome_w']-1.2,
#                             color=palette[2],marker=raster_marker, linewidths=marker_width)

#         iax.set_ylim([-0.2,1.05])
#         iax.set_xlabel('Trial number')
#         iax.set_ylabel('Lick probability')
#         sns.despine()
#         # plt.tight_layout()
#         plt.title(f"{mouse_id} -- Day {df.session_day.iloc[0]}")

#         pdf.savefig()
#         plt.close()

#     pdf.close()


# def read_results_excel(sessions_db_path, data_folder_path):
#     """Generates result dataframe from sessions result text files.
#     Requires an excel session database.

#     Args:
#         sessions_db_path (string): Path to session database.
#         data_folder_path (string): Path to server data folder.

#     Returns:
#         DataFrame: Result dataframe for all sessions in database.
#     """

#     sessions = pd.read_excel(sessions_db_path, dtype={'session_day':str})
#     sessions = sessions.dropna(axis=0, how='all')
#     # sessions = sessions.loc[sessions.batch_id.isin([4,5])]

#     data = []
#     for _, isession in sessions.iterrows():
#         # mouse_id, session_folder, session_id, session_day, reward_group, weight_initial, weight_before, batch_id = isession
#         mouse_id, session_id, session_day, reward_group, inactivation, inactivation_type = isession
#         results_path = os.path.join(data_folder_path, mouse_id, 'Training', session_id, 'Results.csv')
#         if not os.path.exists(results_path):
#             results_path = os.path.join(data_folder_path, mouse_id, 'Training', session_id, 'Results.txt')

#         if os.path.splitext(results_path)[1] == '.csv':
#             df = pd.read_csv(results_path, sep=',', engine='python')
#         else:
#             df = pd.read_csv(results_path, sep=r'\s+', engine='python')
#         df['mouse_id'] = mouse_id
#         df['session_id'] = session_id
#         df['session_day'] = str(session_day)
#         df['reward_group'] = str(reward_group)
#         df['inactivation'] = str(inactivation)
#         df['inactivation_type'] = str(inactivation_type)
#         # df['weight_initial'] = weight_initial
#         # df['weight_before'] = weight_before
#         # df['batch_id'] = int(batch_id)

#         # match_str = re.search(r'\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}',session_id).group()
#         # df['date_time'] = pd.to_datetime(match_str, format='%Y%m%d_%H%M%S')

#             # Rename columns for compatibility with old data.
#         df = df.rename(columns={
#             'trialnumber': 'trial_number',
#             'Perf': 'perf',
#             'TrialTime': 'trial_time',
#             'Association': 'association_flag',
#             'Quietwindow': 'quiet_window',
#             'ITI': 'iti',
#             'Stim/NoStim': 'is_stim',
#             'Whisker/NoWhisker': 'is_whisker',
#             'Auditory/NoAuditory': 'is_auditory',
#             'Lick': 'lick_flag',
#             'ReactionTime': 'reaction_time',
#             'WhStimDuration':'wh_stim_duration',
#             'WhStimAmp': 'wh_stim_amp',
#             'WhRew': 'wh_reward',
#             'Rew/NoRew': 'is_reward',
#             'AudDur': 'aud_stim_duration',
#             'AudDAmp': 'aud_stim_amp',
#             'AudFreq': 'aud_stim_freq',
#             'AudRew': 'aud_reward',
#             'EarlyLick': 'early_lick',
#             'Lick/NoLight': 'is_light',
#             'LightAmp': 'light_amp',
#             'LightDur': 'light_duration',
#             'LightFreq': 'light_freq',
#             'LightPreStim': 'light_prestim'
#         })
#         data.append(df)
#     data = pd.concat(data)
#     # Performance column code:
#     # 0: whisker miss
#     # 1: auditory miss
#     # 2: whisker hit
#     # 3: auditory hit
#     # 4: correct rejection
#     # 5: false alarm
#     # 6: early lick

#     map_hits = {
#         0: 0,
#         1: 0,
#         2: 1,
#         3: 1,
#         4: 0,
#         5: 1,
#         6: np.nan,
#     }
#     data['lick_flag'] = data['perf'].map(map_hits)
#     # data = data.sort_values(['mouse_id','session_id','trial_number']).reset_index(drop=True)

#     return data


# def read_results_GF(sessions_db_path, data_folder_path):

#     sessions = pd.read_excel(sessions_db_path, converters={'session_day':str})
#     sessions = sessions.dropna(axis=0, how='all')
#     # sessions = sessions.loc[sessions.batch_id.isin([4,5])]

#     data = []
#     for _, isession in sessions.iterrows():
#         # mouse_id, session_folder, session_id, session_day, reward_group, weight_initial, weight_before, batch_id = isession
#         mouse_id, session_id, session_day, reward_group, inactivation, inactivation_type = isession
#         json_path = os.path.join(data_folder_path, mouse_id, 'Recordings', 'BehaviourData', session_id, 'performanceResults.json')
#         with open(json_path) as f:
#             js = json.load(f)
#         df = pd.DataFrame(js['results'], columns=js['headers'])

#         df['mouse_id'] = mouse_id
#         df['session_id'] = session_id
#         df['session_day'] = str(session_day)
#         df['reward_group'] = str(reward_group)
#         df['inactivation'] = str(inactivation)
#         df['inactivation_type'] = str(inactivation_type)

#         # Rename columns for compatibility with old data.
#         df = df.rename(columns={
#             'trialnumber': 'trial_number',
#             'trial': 'trial_number',
#             'Perf': 'perf',
#             'performance': 'perf',
#             'TrialTime': 'trial_time',
#             'Association': 'association_flag',
#             'Quietwindow': 'quiet_window',
#             'ITI': 'iti',
#             'Stim/NoStim': 'is_stim',
#             'stim_nostim': 'is_stim',
#             'Whisker/NoWhisker': 'is_whisker',
#             'wh_nowh': 'is_whisker',
#             'Auditory/NoAuditory': 'is_auditory',
#             'aud_noaud': 'is_auditory',
#             'Lick': 'lick_flag',
#             'lick': 'lick_flag',
#             'ReactionTime': 'reaction_time',
#             'WhStimDuration':'wh_stim_duration',
#             'whiskerstim_dur': 'wh_stim_duration',
#             'WhStimAmp': 'wh_stim_amp',
#             'whamp': 'wh_stim_amp',
#             'WhRew': 'wh_reward',
#             'whrew': 'wh_reward',
#             'Rew/NoRew': 'is_reward',
#             'rew_norew': 'is_reward',
#             'AudDur': 'aud_stim_duration',
#             'audstimdur': 'aud_stim_duration',
#             'AudDAmp': 'aud_stim_amp',
#             'audstimamp': 'aud_stim_amp',
#             'AudFreq': 'aud_stim_freq',
#             'audstimfreq': 'aud_stim_freq',
#             'AudRew': 'aud_reward',
#             'audrew': 'aud_reward',
#             'EarlyLick': 'early_lick',
#             'Lick/NoLight': 'is_light',
#             'light_nolight': 'is_light',
#             'LightAmp': 'light_amp',
#             'lightamp': 'light_amp',
#             'LightDur': 'light_duration',
#             'lightdur': 'light_duration',
#             'LightFreq': 'light_freq',
#             'lightfreq': 'light_freq',
#             'LightPreStim': 'light_prestim',
#             'lightprestim': 'light_prestim',
#         })
#         data.append(df)
#     data = pd.concat(data)
#     # Performance column code:
#     # 0: whisker miss
#     # 1: auditory miss
#     # 2: whisker hit
#     # 3: auditory hit
#     # 4: correct rejection
#     # 5: false alarm
#     # 6: early lick

#     map_hits = {
#         0: 0,
#         1: 0,
#         2: 1,
#         3: 1,
#         4: 0,
#         5: 1,
#         6: np.nan,
#     }
#     data['lick_flag'] = data['perf'].map(map_hits)
#     # data = data.sort_values(['mouse_id','session_id','trial_number']).reset_index(drop=True)

#     return data


# def read_results(sessions_db_path, data_folder_path):
#     sessions = pd.read_excel(sessions_db_path, dtype={'session_type':object})
#     sessions = sessions.dropna(axis=0,how='all')
#     data = []
#     for _, isession in sessions.iterrows():
#         mouse_id, session_folder, session_id, session_day, reward_group, weight_initial, weight_before, batch_id = isession
#         if pd.isna(session_folder):
#             session_folder = ''
#         if os.path.exists(os.path.join(data_folder_path, mouse_id, session_folder, session_id, 'Results.txt')):
#             df = pd.read_csv(os.path.join(data_folder_path, mouse_id, session_folder, session_id, 'Results.txt'), sep=r'\s+', engine='python')
#             df['mouse_id'] = mouse_id
#             df['session_id'] = session_id
#             df['session_day'] = str(session_day)
#             df['reward_group'] = str(reward_group)
#             df['weight_initial'] = weight_initial
#             df['weight_before'] = weight_before
#             df['batch_id'] = int(batch_id)
#             # match_str = re.search(r'\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}',session_id).group()
#             # df['date_time'] = pd.to_datetime(match_str, format='%Y%m%d_%H%M%S')

#                 # Rename columns for compatibility with old data.
#             df = df.rename(columns={
#                 'trialnumber': 'trial_number',
#                 'Perf': 'perf',
#                 'TrialTime': 'trial_time',
#                 'Association': 'association_flag',
#                 'Quietwindow': 'quiet_window',
#                 'ITI': 'iti',
#                 'Stim/NoStim': 'is_stim',
#                 'Whisker/NoWhisker': 'is_whisker',
#                 'Auditory/NoAuditory': 'is_auditory',
#                 'Lick': 'lick_flag',
#                 'ReactionTime': 'reaction_time',
#                 'WhStimDuration':'wh_stim_duration',
#                 'WhStimAmp': 'wh_stim_amp',
#                 'WhRew': 'wh_reward',
#                 'Rew/NoRew': 'is_reward',
#                 'AudDur': 'aud_stim_duration',
#                 'AudDAmp': 'aud_stim_amp',
#                 'AudFreq': 'aud_stim_freq',
#                 'AudRew': 'aud_reward',
#                 'EarlyLick': 'early_lick',
#                 'Lick/NoLight': 'is_light',
#                 'LightAmp': 'light_amp',
#                 'LightDur': 'light_duration',
#                 'LightFreq': 'light_freq',
#                 'LightPreStim': 'light_prestim'
#             })
#             data.append(df)
#     data = pd.concat(data)
#     # Performance column code:
#     # 0: whisker miss
#     # 1: auditory miss
#     # 2: whisker hit
#     # 3: auditory hit
#     # 4: correct rejection
#     # 5: false alarm
#     # 6: early lick

#     map_hits = {
#         0: 0,
#         1: 0,
#         2: 1,
#         3: 1,
#         4: 0,
#         5: 1,
#         6: np.nan,
#     }
#     data['lick_flag'] = data['perf'].map(map_hits)
#     # data = data.sort_values(['mouse_id','session_id','trial_number']).reset_index(drop=True)

#     return data
if __name__ == '__main__':

    # Read behavior results.
    SESSIONS_DB_PATH = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\mice_info\session_metadata.xlsx'
    # SESSIONS_DB_PATH = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs\\sessions_muscimol_GF.xlsx'
    DATA_FOLDER_PATH = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\data'
    # table = read_results_excel(SESSIONS_DB_PATH, DATA_FOLDER_PATH)
    if ('GF' in SESSIONS_DB_PATH) or ('MI' in SESSIONS_DB_PATH):
        table = read_results_GF(SESSIONS_DB_PATH, DATA_FOLDER_PATH)
    else:
        table = read_results_excel(SESSIONS_DB_PATH, DATA_FOLDER_PATH)

    # Plot parameters.
    sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1,
                rc={'font.sans-serif':'Arial'})
    PALETTE = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc', '#333333']
    PALETTE = sns.color_palette(PALETTE)


    SESSIONS_DB_PATH = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\behavior_sessions.xlsx"
    DATA_FOLDER_PATH = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard'
    table = read_results_excel(SESSIONS_DB_PATH, DATA_FOLDER_PATH)


    # Compute performance.
    # --------------------


    # Plotting performance for a group of mice.

    # batch = 4
    palette = PALETTE
    NMAX_TRIAL = 300
    sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=0.8,
                rc={'font.sans-serif':'Arial'})
    pdf_path = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior'
    pdf_fname = f'summary_batch_{int(batch):03d}.pdf'
    # mice = table.loc[(table.batch_id==batch), 'mouse_id']
    mice = table.loc[(table.inactivation_type=='execution'), 'mouse_id']
    mice = mice.drop_duplicates().to_list()
    # mice = [m for m in mice if m not in ['AR111', 'AR112', 'AR113', 'AR114', 'AR081']]

    pdf = PdfPages(os.path.join(pdf_path,pdf_fname))

    # Plot average performance across mice and days.
    f, axes = plt.subplots(2,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(12,6))
    plt.subplots_adjust(wspace=.4, hspace=0.6)

    reward_group = 'R+'
    data = table.loc[(table.mouse_id.isin(mice)) & (table.reward_group==reward_group)]
    plot_average_across_days(data, mice, reward_group, PALETTE, nmax_trials=NMAX_TRIAL, ax=axes[0,0])
    plot_perf_across_blocks(data, mice, reward_group, PALETTE, nmax_trials=NMAX_TRIAL, ax=axes[0,1])

    reward_group = 'R-'
    data = table.loc[(table.mouse_id.isin(mice)) & (table.reward_group==reward_group)]
    plot_average_across_days(data, mice, reward_group, PALETTE, nmax_trials=NMAX_TRIAL, ax=axes[1,0])
    plot_perf_across_blocks(data, mice, reward_group, PALETTE, nmax_trials=NMAX_TRIAL, ax=axes[1,1])
    sns.despine(fig=f)
    pdf.savefig()
    plt.close()


    # Plot performance across days for each mouse.

    # batch = 4
    palette = PALETTE
    nmax_trials = 200
    sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=0.8,
                rc={'font.sans-serif':'Arial'})
    mice = table.mouse_id.unique()
    # mice = [m for m in mice if m not in ['AR111', 'AR112', 'AR113', 'AR114', 'AR081']]
    pdf_path = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior'
    pdf_fname = f'performance_across_days.pdf'
    pdf = PdfPages(os.path.join(pdf_path,pdf_fname))

    data = table.loc[table.trial<=nmax_trials]

    for imouse in mice:
        plot_single_mouse_performance_across_days(data, imouse, PALETTE)
        pdf.savefig()   
        plt.close()
    pdf.close()


    # Plot single session for each mouse.

    save_path = 'C:\\Users\\aprenard\\recherches\\fast-learning\\results\\behavior\\mice'

    mice = table.mouse_id.unique()

    for mouse_id in mice:
        print(mouse_id)
        pdf_path = os.path.join(save_path, mouse_id)
        if not os.path.isdir(pdf_path):
            os.mkdir(pdf_path)

        # Create a new pdf file
        pdf_fname = f'{mouse_id}_single_sessions.pdf'
        pdf = PdfPages(os.path.join(pdf_path, pdf_fname))

        for session_id in table.loc[table.mouse_id==mouse_id, 'session_id'].unique():
            print(session_id)
            # Call the plot_single_session function with the current session and mouse ID
            plot_single_session(table, session_id, PALETTE)
            # Save the current figure to the pdf file
            pdf.savefig()
            plt.close()
        # Close the pdf file
        pdf.close()

    # Particle test plot.
    # ###################

    # batches = [4]
    # mice = table.loc[(table.batch_id.isin(batches)), 'mouse_id']
    # mice = mice.drop_duplicates().to_list()
    # mice = [imouse for imouse in mice if imouse not in []]
    reward_group = 'R+'
    nmax_trials = 200
    palette = PALETTE
    ax = None

    data = table.loc[(table.mouse_id.isin(mice)) & (table.trial <=nmax_trials) & (table.reward_group==reward_group)]
    data = data.groupby(['mouse_id','session_id','session_day'], as_index=False).agg(np.mean)
    # Sort dataframe by days.
    days = ['P_ON_1','P_OFF', 'P_ON_2']
    data = data.loc[data.session_day.isin(days)]
    data = data.sort_values(by='session_day', key=lambda x: x.map({'P_ON_1':0,'P_OFF':1,'P_ON_2':2}))
    color_wh = palette[2]
    if reward_group=='R-':
        color_wh = palette[3]

    for imouse in mice:
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')

    # sns.lineplot(data=data, x='session_day', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='session_day', y='hr_c', order=days, join=False, color=palette[4], ax=ax)

    # sns.lineplot(data=data, x='session_day', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='session_day', y='hr_a', order=days, join=False, color=palette[0], ax=ax)

    # sns.lineplot(data=data, x='session_day', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='session_day', y='hr_w', order=days, join=False, color=color_wh, ax=ax)

    ax = plt.gca()
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels(['ON', 'OFF', 'ON'])
    ax.set_xlabel('Particle test')
    ax.set_ylabel('Lick probability')
    sns.despine()


    # Average inactivation plots.
    # ###########################

    # Inactivation during learning.
    # -----------------------------

    nmax_trials = 200
    palette = PALETTE
    ax = None
    # data = table.loc[table.inactivation_type=='learning']
    # inactivation_labels = ['pre1', 'pre2', 'muscimol1','muscimol2','muscimol3','recovery1','recovery2','recovery3']
    # data = table.loc[table.inactivation_type=='execution']
    # inactivation_labels = ['pre', 'muscimol1','ringer', 'muscimol2',]
    experimenter = 'AR'
    mice = table.mouse_id.unique()
    mice = [imouse for imouse in mice if experimenter in imouse]
    data = table.loc[table.mouse_id.isin(mice)]
    data = data.loc[data.inactivation_type=='learning']
    inactivation_labels = ['pre-2', 'pre-1', 'muscimol1', 'muscimol2',
                        'muscimol3', 'recovery1', 'recovery2', 'recovery3']
    data = data.loc[data.inactivation.isin(inactivation_labels)]
    data = data.loc[data.trial<=nmax_trials]

    data = data.groupby(['mouse_id','session_id','inactivation'], as_index=False).agg(np.mean)
    # Sort dataframe by days.
    m = dict(zip(inactivation_labels, range(6)))
    data = data.sort_values(by='inactivation', key=lambda x: x.map(m))
    color_wh = palette[2]

    fig = plt.figure(dpi=300)
    ax = plt.gca()


    for imouse in mice:
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_c', estimator=np.mean, color=palette[5],
                    alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_a', estimator=np.mean, color=palette[0],
                    alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_w', estimator=np.mean, color=color_wh,
                    alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')

    # sns.lineplot(data=data, x='muscimol', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_c', order=inactivation_labels, join=False, color=palette[5], ax=ax)

    # sns.lineplot(data=data, x='inactivation', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_a', order=inactivation_labels, join=False, color=palette[0], ax=ax)

    # sns.lineplot(data=data, x='inactivation', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_w', order=inactivation_labels, join=False, color=color_wh, ax=ax)

    ax = plt.gca()
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels(['-2', '-1', 'M 1', 'M 2', 'M 3', 'R 1', 'R 2', 'R 3'])
    ax.set_xlabel('Muscimol inactivation during learning')
    # ax.set_ylabel('Lick probability')
    # ax.set_xticklabels(['D0', 'M 1', 'Recovery'])
    # ax.set_xticklabels(['Pre -3', 'Pre -2', 'Pre -1', 'M'])
    # ax.set_xlabel('Muscimol inactivation during execution')
    ax.set_ylabel('Lick probability')
    # plt.title(f'Session length = {nmax_trials} trials')
    plt.title(f'{experimenter}')
    sns.despine()




    # Muscimol execution.
    # -------------------

    nmax_trials = 200
    palette = PALETTE
    ax = None
    # data = table.loc[table.inactivation_type=='learning']
    # inactivation_labels = ['pre1', 'pre2', 'muscimol1','muscimol2','muscimol3','recovery1','recovery2','recovery3']
    # data = table.loc[table.inactivation_type=='execution']
    # inactivation_labels = ['pre', 'muscimol1','ringer', 'muscimol2',]
    experimenter = 'AR'
    mice = table.mouse_id.unique()
    mice = [imouse for imouse in mice if experimenter in imouse]
    mice = [imouse for imouse in mice if imouse in ['AR115', 'AR116']]

    data = table.loc[table.mouse_id.isin(mice)]
    data = data.loc[data.inactivation_type=='execution']
    inactivation_labels = ['pre-1', 'muscimol1']
    data = data.loc[data.inactivation.isin(inactivation_labels)]
    data = data.loc[data.trial<=nmax_trials]

    data = data.groupby(['mouse_id','session_id','inactivation'], as_index=False).agg(np.mean)
    # Sort dataframe by days.
    m = dict(zip(inactivation_labels, range(6)))
    data = data.sort_values(by='inactivation', key=lambda x: x.map(m))
    color_wh = palette[2]

    fig = plt.figure(dpi=300)
    ax = plt.gca()

    for imouse in mice:
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_c', estimator=np.mean, color=palette[4], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_a', estimator=np.mean, color=palette[0], alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')
        sns.lineplot(data=data.loc[data.mouse_id==imouse], x='inactivation', y='hr_w', estimator=np.mean, color=color_wh, alpha=.6, legend=False, ax=ax, marker=None, err_style='bars')

    # sns.lineplot(data=data, x='inactivation', y='hr_c', estimator=np.mean, color=palette[4], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_c', order=inactivation_labels, join=False, color=palette[4], ax=ax)

    # sns.lineplot(data=data, x='inactivation', y='hr_a', estimator=np.mean, color=palette[0], alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_a', order=inactivation_labels, join=False, color=palette[0], ax=ax)

    # sns.lineplot(data=data, x='inactivation', y='hr_w', estimator=np.mean, color=color_wh, alpha=1, legend=False, ax=ax, marker='o', err_style='bars')
    sns.pointplot(data=data, x='inactivation', y='hr_w', order=inactivation_labels, join=False, color=color_wh, ax=ax)

    ax = plt.gca()
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels(['Day before', 'Muscimol'])
    ax.set_xlabel('Muscimol inactivation during execution.')
    ax.set_ylabel('Lick probability')
    plt.title(f'Session length = {nmax_trials}')
    sns.despine()




    # Number of whisker trials before first whisker hit.
    # ##################################################


    df = table.loc[table.session_day=='0']
    df = df.loc[df.is_whisker==1]

    f = lambda x: x.reset_index(drop=True).idxmax()+1
    fh = df.groupby(['mouse_id', 'reward_group'], as_index=False)[['outcome_w']].agg(f)
    fh.reward_group = fh.reward_group.astype("category")
    fh.reward_group = fh.reward_group.cat.set_categories(['R+','R-'])
    fh = fh.sort_values(['reward_group'])

    plt.figure(figsize=(4,4))
    sns.swarmplot(data=fh, x='reward_group', y='outcome_w', hue='reward_group',palette=PALETTE[2:4],
                    hue_order=['R+','R-'])
    plt.ylim([0,16])
    ax = plt.gca()
    ax.set_yticks(range(0,18,1))
    ax.set_xticklabels(['R+', 'R-'])
    ax.set_xlabel('Reward group')
    ax.set_ylabel('First whisker hit')
    ax.get_legend().remove()
    sns.despine()
    plt.tight_layout()



    # Whisker performance averaged across mice.
    # #########################################


    #----

    df = table.loc[table.session_day=='0']
    df = df.loc[(df.is_whisker==1) & (df.trial_w<=20)]

    # df.loc[df.mouse_id=='AR060'].plot(x='trial_w', y='outcome_w')
    # df.loc[df.mouse_id=='AR060'].plot(x='trial_w', y='cumsum_w')

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x='trial_w', y='cumsum_w', hue='reward_group', palette=PALETTE[2:4],
                hue_order=['R+','R-'], units='mouse_id', estimator=None, style='mouse_id')
    plt.ylim([-20,20])
    plt.xlim([0,20])
    ax = plt.gca()
    ax.set_xticks([0,5,10,15,20])
    ax.get_legend().remove()
    sns.despine()
    plt.xlabel('Whisker trials')
    plt.ylabel('Sum (Hits - Misses)')


    # Same but starting at first whisker hit.

    # Remove rows before first whisker hit.
    df = table.loc[table.session_day=='0']
    df = df.loc[(df.is_whisker==1) & (df.trial_w<=20)]
    f = lambda x: x.loc[x.index>=x.outcome_w.idxmax()]
    df_fh = df.groupby(['mouse_id'], as_index=False).apply(f)

    # Reindex from first hit and recompute cumsum.
    df_fh['trial_w'] = df_fh.loc[(df_fh.is_whisker==1.0)]\
                                    .groupby('session_id').cumcount()+1
    df_fh['cumsum_w'] = df_fh.loc[(df_fh.is_whisker==1.0)]['outcome_w']\
                                    .map({0:-1,1:1,np.nan:0})
    df_fh['cumsum_w'] = df_fh.groupby('session_id')['cumsum_w'].transform(np.cumsum)


    plt.figure()
    sns.lineplot(data=df_fh, x='trial_w', y='cumsum_w', hue='reward_group', palette=PALETTE[2:4],
                hue_order=['R+','R-'], units='mouse_id', estimator=None, style='mouse_id',alpha=.8)
    plt.ylim([-20,20])
    plt.xlim([0,20])
    ax = plt.gca()
    ax.set_xticks([0,5,10,15,20])
    ax.get_legend().remove()
    sns.despine()
    plt.xlabel('Whisker trials')
    plt.ylabel('Sum (Hits - Misses)')

    # ----


    # Remove rows before first whisker hit.
    df = table.loc[table.session_day=='0']
    df = df.loc[(df.is_whisker==1) & (df.trial_w<=20)]
    f = lambda x: x.loc[x.index>=x.outcome_w.idxmax()]
    df_fh = df.groupby(['mouse_id'], as_index=False).apply(f)

    # Remove mice that didn't lick the first 20 trials
    df_fh = df_fh.loc[df_fh.mouse_id != 'AR104']

    # Reindex from first hit and recompute cumsum.
    df_fh['trial_w'] = df_fh.loc[(df_fh.is_whisker==1.0)]\
                                    .groupby('session_id').cumcount()+1

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_fh, x='trial_w', y='outcome_w', hue='reward_group', palette=PALETTE[2:4],
                hue_order=['R+','R-'])
    plt.xlim([0,20])
    ax = plt.gca()
    ax.set_xticks([0,5,10,15,20])
    ax.get_legend().remove()
    sns.despine()
    plt.xlabel('Whisker trials')
    plt.ylabel('Lick probability trial-wise')


    # Reading GF behavior data.
    # #########################

    # mice_inactivation_learning = [
    #     'GF271',
    #     'GF290',
    #     'GF291',
    #     'GF292',
    #     'GF293',
    #     'MI027',
    #     ]

    # mice_inactivation_execution = [
    #     'GF240','GF241', 'GF248', 'GF252', 'GF253', 'GF256', 'GF267', 'GF272', 'GF278']



    # for it,
    # isessions in enumerate(sessions):
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


    # def perf_stats(data,g1_sessions,g2_sessions,g1_label,g2_label,nmax_trials,test,alternative='greater'):

    #     d_g1 = table.loc[table.session_id.isin(g1_sessions)]
    #     d_g1 = d_g1.loc[d_g1.trial < nmax_trials]
    #     d_g1 = d_g1.groupby(['mouse_id','session_id']).agg(np.nanmean)
    #     d_g1['group'] = g1_label

    #     d_g2 = table.loc[table.session_id.isin(g2_sessions)]
    #     d_g2 = d_g2.loc[d_g2.trial < nmax_trials]
    #     d_g2 = d_g2.groupby(['mouse_id','session_id']).agg(np.nanmean)
    #     d_g2['group'] = g2_label

    #     g1 = d_g1.outcome_w.to_numpy()
    #     g2 = d_g2.outcome_w.to_numpy()
    #     ci1_left, ci1_right = ci_bootstrap(g1)
    #     ci2_left, ci2_right = ci_bootstrap(g2)

    #     if test == 'wilcoxon':
    #         _, pval = stats.wilcoxon(g1,g2,alternative=alternative)
    #     elif test == 'mann_whitney':
    #         _, pval = stats.mannwhitneyu(g1,g2,alternative=alternative)

    #     stats_strg = f'''mean {g1.mean():.4f} (95% ci: {ci1_left:.4f}, {ci1_right:.4f})\n
    #                   \mean {g2.mean():.4f} (95% ci: {ci2_left:.4f}, {ci2_right:.4f})\n
    #                   pval: {pval:.4f}'''
    #     print(stats_strg)


    # # # TODO: make sure ordering of sessions is correct
    # # # TODO: remove the rewarded parameter in function, guess it from df
    # # # TODO: improve stat function, currently I'm changing which trial type manually

    # # A2_R = table.loc[(table.reward_group==1) & (table.batch_id==2) & (table.session_day == 'A-2')]['session_id'].drop_duplicates()
    # # A1_R = table.loc[(table.reward_group==1) & (table.batch_id==2) & (table.session_day == 'A-1')]['session_id'].drop_duplicates()
    # # A2_NR = table.loc[(table.reward_group==0) & (table.batch_id==2) & (table.session_day == 'A-2')]['session_id'].drop_duplicates()
    # # A1_NR = table.loc[(table.reward_group==0) & (table.batch_id==2) & (table.session_day == 'A-1')]['session_id'].drop_duplicates()
    # # W1_R = table.loc[(table.reward_group==1) & (table.batch_id==2) & (table.session_day == 'W1')]['session_id'].drop_duplicates()
    # # W1_NR = table.loc[(table.reward_group==0) & (table.batch_id==2) & (table.session_day == 'W1')]['session_id'].drop_duplicates()
    # # P_ON1 = table.loc[(table.session_day == 'P_ON1')]['session_id'].drop_duplicates()
    # # P_OFF1 = table.loc[(table.session_day == 'P_OFF1')]['session_id'].drop_duplicates()
    # # P_ON2 = table.loc[(table.session_day == 'P_ON2')]['session_id'].drop_duplicates()


    # # session_lists = [A2_R, A1_R, W1_R]
    # # labels = ['D-2','D-1','D0']
    # # plot_average_perf(table, session_lists, labels, True,300)

    # # session_lists = [A2_NR, A1_NR, W1_NR]
    # # labels = ['D-2','D-1','D0']
    # # plot_average_perf(table, session_lists, labels, False,300)

    # # session_lists = [P_ON1,P_OFF1,P_ON2]
    # # labels = ['ON pre','OFF','ON post']
    # # plot_average_perf(table, session_lists, labels, True,300)

    # # perf_stats(table,P_ON1,P_OFF1,'P_ON','P_OFF',300,test='wilcoxon',alternative='greater')
    # # perf_stats(table,P_ON2,P_OFF1,'P_ON','P_OFF',300,test='wilcoxon',alternative='greater')

    # # perf_stats(table,W1,A1,'W1','A1',250,test='wilcoxon',alternative='greater')

    # # perf_stats(table,W1,CNO1_S1,'W1','CNO1_S1',250,test='mann_whitney',alternative='greater')
    # # perf_stats(table,W1,CNO1_S2p_S1,'W1','CNO1_S2',250,test='mann_whitney',alternative='greater')

    # # perf_stats(table,P_ON2,P_OFF1,'SAL2','CNO1',250,test='mann_whitney',alternative='greater')
    # # perf_stats(table,W3,CNO1_S2p_S1,'SAL2','CNO1',250,test='mann_whitney',alternative='greater')



    # # session_lists = [A1_S2p_S1, A2_S2p_S1, CNO1_S2p_S1, CNO2_S2p_S1, CNO3_S2p_S1, SAL1_S2p_S1, SAL2_S2p_S1]
    # # labels = ['A1', 'A2', 'CNO1', 'CNO2', 'CNO3', 'SAL1', 'SAL2']
    # # plot_average_perf(table, session_lists, labels, 100)
    # # plt.title('wS2 projecting neurons inactivation')

    # # session_lists = [A1_S1, A2_S1, CNO1_S1, CNO2_S1, CNO3_S1, SAL1_S1, SAL2_S1]
    # # labels = ['A1', 'A2', 'CNO1', 'CNO2', 'CNO3', 'SAL1', 'SAL2']
    # # plot_average_perf(table, session_lists, labels, 100)
    # # plt.title('wS1 inactivation')
