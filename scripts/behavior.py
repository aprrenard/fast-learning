"""_summary_
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.utils import ci_bootstrap


def plot_perf(df,m_id,s_list=None,representation='bar',nmaxtrial=None,figsize=None):
    """Plots performance over time.

    Args:
        df (_type_): _description_
        m_id (_type_): _description_
        s_list (_type_, optional): _description_. Defaults to None.
        representation (str, optional): _description_. Defaults to 'win_fixtime'.
        nmaxtrial (_type_, optional): _description_. Defaults to None.
        figsize (_type_, optional): _description_. Defaults to None.
    """

    # Set plot parameters.
    pal = ['#225ea8', '#238443', '#e31a1c', '#666666', '#00FFFF', '#e36f8a']
    # pal = ['#7570b3', '#1b9e77', '#d95f02', '#666666']
    pal = sns.color_palette(pal)
    raster_marker = 2
    # Green for whisker rewarded, red for non-rewarded.
    wh_c = pal[1] if (df.loc[df.mouse_id==m_id, 'rewarded_group'].iloc[0] == 1) else pal[5]

    # If sessions are not specified, plot auditory and whisker days.
    if not s_list:
        labels = [f'A{i}' for i in range(1,13)] + [f'W{i}' for i in range(1,13)]
        s_list = df.loc[(df.mouse_id==m_id) & (df.session_day.isin(labels))]
        s_list = s_list['session_day'].drop_duplicates().sort_values(ascending=True).to_numpy()

    if len(s_list) == 1:
        marker_width = 2
    else:
        marker_width = 1.5

    if not figsize:
        figsize = (10,8) if len(s_list) == 1 else (25,8)

    if representation == 'bar':
        perf_c = 'hr_c'
        perf_a = 'hr_a'
        perf_w = 'hr_w'
    elif representation == 'cumsum':
        perf_c = 'cumsum_c'
        perf_a = 'cumsum_a'
        perf_w = 'cumsum_w'
    else:
        raise ValueError('Could not interpret representation.')

    _, axes = plt.subplots(1,len(s_list), figsize=figsize)
    if len(s_list) == 1:
        axes = np.array([axes])  # In case of single axis.

    for i, isession in enumerate(s_list):

        d = df.loc[(df.early_lick==0) & (df.mouse_id==m_id) & (df.session_day==isession)]

        if representation == 'bar':
            sns.lineplot(data=d,x='trial',y=perf_c,color=pal[3],ax=axes[i],
                         marker='o')
        elif representation != 'cumsum':
            axes[i].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_c']-0.1,
                            color=pal[3],marker=raster_marker, linewidths=marker_width)
            axes[i].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_c']-1.1,
                            color='k',marker=raster_marker, linewidths=marker_width)

        if representation == 'bar':
            sns.lineplot(data=d,x='trial',y=perf_a,color=pal[0],ax=axes[i],
                        marker='o')
        elif representation != 'cumsum':
            axes[i].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_a']-0.14,
                            color=pal[4],marker=raster_marker, linewidths=marker_width)
            axes[i].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_a']-1.14,
                            color=pal[0],marker=raster_marker, linewidths=marker_width)

        if representation == 'bar':
            sns.lineplot(data=d,x='trial',y=perf_w,color=wh_c,ax=axes[i],
                         marker='o')
        elif representation != 'cumsum':
            axes[i].scatter(x=d.loc[d.outcome==0]['trial'],y=d.loc[d.outcome==0]['outcome_w']-0.18,
                            color=pal[5],marker=raster_marker, linewidths=marker_width)
            axes[i].scatter(x=d.loc[d.outcome==1]['trial'],y=d.loc[d.outcome==1]['outcome_w']-1.18,
                            color=pal[1],marker=raster_marker, linewidths=marker_width)

        s_labels = [f'D-{i}' for i in range(len(s_list)-1,0,-1)]
        s_labels.append('D0')
        axes[i].set_title(s_labels[i])
        axes[i].set_xlabel('Trials')

    if representation != 'cumsum':
        for axis in axes.flatten():
            axis.set_xlim(0,nmaxtrial)
            axis.set_ylim(-.2,1.05)
            axis.set_ylabel('')

    sns.despine()
    plt.show()
    plt.tight_layout()


sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)

SESSION_FILE = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs\\behavior_sessions.xlsx'
sessions = pd.read_excel(SESSION_FILE)
sessions = sessions.dropna(axis=0,how='all')

PATH_READ = 'M:\\data'
df_result = []
for i in range(sessions.shape[0]):
    mouse_id, session_folder, session_id, session_day, rewarded_group, _, _, _, batch_id = sessions.iloc[i]
    if pd.isna(session_folder):
        session_folder = ''
    d = pd.read_csv(os.path.join(PATH_READ, mouse_id, session_folder, session_id, 'Results.txt'), sep='\s+', engine='python')
    d['mouse_id'] = mouse_id
    d['session_id'] = session_id
    d['session_day'] = session_day
    d['rewarded_group'] = rewarded_group
    d['batch_id'] = batch_id
    df_result.append(d)
df_result = pd.concat(df_result)

# path = 'K:\LickTrace2.bin'
# x = np.fromfile(path)
# x = x.reshape(-1,2).T
# plt.plot(x[1])

# Rename columns.
df_result = df_result.rename(columns={
    'trialnumber': 'trial_number',
    'WhStimDuration':'wh_duration',
    'Quietwindow': 'quiet_window',
    'ITI': 'ITI',
    'Association': 'association',
    'Stim/NoStim': 'stim',
    'Whisker/NoWhisker': 'wh_stim',
    'Auditory/NoAuditory': 'aud_stim',
    'Lick': 'lick',
    'Perf': 'perf',
    'Lick/NoLight': 'light',
    'ReactionTime': 'reaction_time',
    'WhStimAmp': 'wh_amp',
    'TrialTime': 'trial_time',
    'Rew/NoRew': 'reward',
    'AudRew': 'aud_reward',
    'WhRew': 'wh_reward',
    'AudDur': 'aud_duration',
    'AudDAmp': 'aud_amp',
    'AudFreq': 'aud_freq',
    'EarlyLick': 'early_lick',
    'LightAmp': 'light_amp',
    'LightDur': 'light_duration',
    'LightFreq': 'light_freq',
    'LightPreStim': 'light_prestim'
    })

# Order and reset indices.
df_result = df_result.sort_values(['session_id','session_day','trial_number']).reset_index(drop=True)

# perf:
# 0: W miss
# 1: A miss
# 2: W hit
# 3: A hit
# 4: CR
# 5: FA
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

df_result['outcome'] = df_result['perf'].map(map_hits)


# Compute performance.
# --------------------

WIN_SIZE = 20
BIN_SIZE = 20  # For bar plot representation.

# Adding trial number excluding early lick trials. In case there are aborted
# trials due to early lick during baseline before stimulus delivery.
df_result['trial'] = df_result.loc[(df_result.early_lick==0)].groupby('session_id').cumcount()
# Adding trial number for each stimulus.
df_result['trial_w'] = df_result.loc[(df_result.early_lick==0) & (df_result.wh_stim==1.0)]\
                      .groupby('session_id').cumcount()
df_result['trial_a'] = df_result.loc[(df_result.early_lick==0) & (df_result.aud_stim==1.0)]\
                      .groupby('session_id').cumcount()
df_result['trial_c'] = df_result.loc[(df_result.early_lick==0) & (df_result.stim==0.0)]\
                      .groupby('session_id').cumcount()

# Adding performance outcome column for each stimulus.
df_result['outcome_w'] = df_result.loc[(df_result.early_lick==0) & (df_result.wh_stim==1.0)]['outcome']
df_result['outcome_a'] = df_result.loc[(df_result.early_lick==0) & (df_result.aud_stim==1.0)]['outcome']
df_result['outcome_c'] = df_result.loc[(df_result.early_lick==0) & (df_result.stim==0.0)]['outcome']

# Compute over blocks of 20 trials.
df_result['hr_w'] = df_result.loc[(df_result.early_lick==0)]\
                       .groupby('session_id', as_index=False)\
                       .rolling(BIN_SIZE,min_periods=1,on='trial')\
                       .agg(np.nanmean)['outcome_w'][BIN_SIZE-1::BIN_SIZE]
df_result['hr_a'] = df_result.loc[(df_result.early_lick==0)]\
                       .groupby('session_id', as_index=False)\
                       .rolling(BIN_SIZE,min_periods=1,on='trial')\
                       .agg(np.nanmean)['outcome_a'][BIN_SIZE-1::BIN_SIZE]
df_result['hr_c'] = df_result.loc[(df_result.early_lick==0)]\
                       .groupby('session_id', as_index=False)\
                       .rolling(BIN_SIZE,min_periods=1,on='trial')\
                       .agg(np.nanmean)['outcome_c'][BIN_SIZE-1::BIN_SIZE]


d = df_result.loc[df_result.session_id=='AR013_A1_20220831_145508']

d.rolling(BIN_SIZE,min_periods=1,on='trial').agg(np.nanmean)['outcome_c'][BIN_SIZE-1::BIN_SIZE]

d.trial
# Cumulative sum representation.
df_result['cumsum_w'] = df_result.loc[(df_result.early_lick==0) & (df_result.wh_stim==1.0)]['outcome_w'].map({0:-1,1:1,np.nan:0})
df_result['cumsum_w'] = df_result.groupby('session_id')['cumsum_w'].transform(np.cumsum)
df_result['cumsum_a'] = df_result.loc[(df_result.early_lick==0) & (df_result.aud_stim==1.0)]['outcome_a'].map({0:-1,1:1,np.nan:0})
df_result['cumsum_a'] = df_result.groupby('session_id')['cumsum_a'].transform(np.cumsum)
df_result['cumsum_c'] = df_result.loc[(df_result.early_lick==0) & (df_result.stim==0.0)]['outcome_c'].map({0:-1,1:1,np.nan:0})
df_result['cumsum_c'] = df_result.groupby('session_id')['cumsum_c'].transform(np.cumsum)


plot_perf(df_result,'AR032',s_list=['A-2','A-1','W1'],representation='bar',nmaxtrial=None)
plt.title('AR013 D0 -- Intervals of 20 trials (no slidding)')
# stats.norm.ppf(0.4) - stats.norm.ppf(0.05)


# d = pd.DataFrame(columns=['a','b'],index=range(10))
# d['a'] = range(100,110)
# d['b'] = range(10)
# d.rolling(4,on='a',min_periods=1).agg(np.nanmean)[3::4]

# d = df_result.loc[(df_result.early_lick==0) & (df_result.session_day=='W1') & (df_result.mouse_id=='AR013')]
# d.plot(x='trial',y='outcome_w',marker='o')
# d.rolling(10,on='trial',min_periods=10).agg(np.mean).plot(x='trial',y='outcome',marker='o')
# d[0:10]

# Single day performance

pal = ['#225ea8', '#238443', '#e31a1c', '#666666', '#00FFFF', "#e36f8a"]

sessions = [
    'AR013_W1_R_20220903_115260',
    'AR016_W1_R_20220904_123356',
    'AR017_W1_R_20220904_123359',
]
wh_c = pal[3]
wh_a = pal[0]
wh_w = pal[1]


plt.figure()
ax = plt.gca()
for isession in sessions:
    sns.lineplot(data=df_result.loc[df_result.session_id==isession],x='trial',y='hr_c',color=wh_c,marker='o',ax=ax)
    sns.lineplot(data=df_result.loc[df_result.session_id==isession],x='trial',y='hr_a',color=wh_a,marker='o',ax=ax)
    sns.lineplot(data=df_result.loc[df_result.session_id==isession],x='trial',y='hr_w',color=wh_w,marker='o',ax=ax)



# TODO: add this block column so we can plot different mice across time
d = df_result.loc[df_result.session_id.isin(sessions)][['session_id','hr_c','hr_a','hr_w']].dropna()
f = lambda x: range(x.shape[0])
d['block'] = d.groupby('session_id')['hr_w'].transform(f)

plt.figure()
ax = plt.gca()
for isession in sessions:
    sns.lineplot(data=d,x='block',y='hr_c',color=wh_c,marker='o',ax=ax)
    sns.lineplot(data=d,x='block',y='hr_a',color=wh_a,marker='o',ax=ax)
    sns.lineplot(data=d,x='block',y='hr_w',color=wh_w,marker='o',ax=ax)
ax.set_xticks(range(11))
ax.set_xticklabels(range(1,12))

wh_w =  '#e31a1c'

sessions = [
    'AR015_W1_NR_20220903_170051',
    'AR018_W1_NR_20220904_133527',
]

plt.figure()
ax = plt.gca()
for isession in sessions:
    sns.lineplot(data=df_result.loc[(df_result.session_id==isession) & (df_result.trial<=250)],x='trial',y='hr_c',color=wh_c,marker='o',ax=ax)
    sns.lineplot(data=df_result.loc[(df_result.session_id==isession) & (df_result.trial<=250)],x='trial',y='hr_a',color=wh_a,marker='o',ax=ax)
    sns.lineplot(data=df_result.loc[(df_result.session_id==isession) & (df_result.trial<=250)],x='trial',y='hr_w',color=wh_w,marker='o',ax=ax)


d = df_result.loc[(df_result.session_id.isin(sessions)) & (df_result.trial<=250)][['session_id','hr_c','hr_a','hr_w']].dropna()
f = lambda x: range(x.shape[0])
d['block'] = d.groupby('session_id')['hr_w'].transform(f)

plt.figure()
ax = plt.gca()
for isession in sessions:
    sns.lineplot(data=d,x='block',y='hr_c',color=wh_c,marker='o',ax=ax)
    sns.lineplot(data=d,x='block',y='hr_a',color=wh_a,marker='o',ax=ax)
    sns.lineplot(data=d,x='block',y='hr_w',color=wh_w,marker='o',ax=ax)
ax.set_xticks(range(11))
ax.set_xticklabels(range(1,12))

plt.figure()
ax = plt.gca()
sns.lineplot(data=d,x='block',y='hr_w',color=wh_w,marker='o',ax=ax)


# Plot average performance for different days.
# ############################################


def plot_average_perf(data,s_list,gp_labels,rewarded,nmax_trials):
    df = []
    for it, isessions in enumerate(s_list):
        d = df_result.loc[df_result.session_id.isin(isessions)]
        d = d.loc[d.trial < nmax_trials]
        d = d.groupby(['mouse_id','session_id']).agg(np.nanmean)
        d['group'] = gp_labels[it]
        df.append(d)
    df = pd.concat(df).reset_index()

    pal = ['#225ea8', '#238443', '#e31a1c', '#666666', '#00FFFF', "#e36f8a"]

    wh_c = pal[1] if rewarded else pal[2]

    plt.figure(figsize=(10,8))
    sns.lineplot(data=df,x='group',y='outcome_c',color=pal[3], style='mouse_id',
                 estimator=None, dashes=False, legend=False,alpha=.5)
    sns.scatterplot(data=df,x='group',y='outcome_c',alpha=.5,color=pal[3])
    sns.pointplot(data=df,x='group',y='outcome_c',color=pal[3], join=False)

    sns.lineplot(data=df,x='group',y='outcome_a',color=pal[0], style='mouse_id',
                 estimator=None, dashes=False, legend=False,alpha=.5)
    sns.scatterplot(data=df,x='group',y='outcome_a',alpha=.5,color=pal[0])
    sns.pointplot(data=df,x='group',y='outcome_a',color=pal[0], join=False)

    sns.lineplot(data=df,x='group',y='outcome_w',color=wh_c, style='mouse_id',
                 estimator=None, dashes=False, legend=False,alpha=.5)
    sns.scatterplot(data=df,x='group',y='outcome_w',alpha=.5,color=wh_c)
    sns.pointplot(data=df,x='group',y='outcome_w',color=wh_c, join=False)

    plt.ylim([0,1.1])
    plt.xlabel('Particle test')
    plt.ylabel('Lick probability')
    sns.despine()
    plt.show()


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

    stats_strg = 'mean {:.4f} (95% ci: {:.4f}, {:.4f})\nmean {:.4f} (95% ci: {:.4f}, {:.4f})\npval: {:.4f}'.format(
        g1.mean(), ci1_left, ci1_right,
        g2.mean(), ci2_left, ci2_right,
        pval
    )
    print(stats_strg)


# TODO: make sure ordering of sessions is correct
# TODO: remove the rewarded parameter in function, guess it from df
# TODO: improve stat function, currently I'm changing which trial type manually

A2_R = df_result.loc[(df_result.rewarded_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'A-2')]['session_id'].drop_duplicates()
A1_R = df_result.loc[(df_result.rewarded_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'A-1')]['session_id'].drop_duplicates()
A2_NR = df_result.loc[(df_result.rewarded_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'A-2')]['session_id'].drop_duplicates()
A1_NR = df_result.loc[(df_result.rewarded_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'A-1')]['session_id'].drop_duplicates()
W1_R = df_result.loc[(df_result.rewarded_group==1) & (df_result.batch_id==2) & (df_result.session_day == 'W1')]['session_id'].drop_duplicates()
W1_NR = df_result.loc[(df_result.rewarded_group==0) & (df_result.batch_id==2) & (df_result.session_day == 'W1')]['session_id'].drop_duplicates()
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
