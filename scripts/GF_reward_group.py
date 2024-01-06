import os
import numpy as np
import pandas as pd
import json


def read_excel_database(folder, file_name):
    excel_path = os.path.join(folder, file_name)
    database = pd.read_excel(excel_path, converters={'session_day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    # Change yes/no columns to booleans.
    database = database.replace('yes', True)
    database = database.replace('no', False)
    database = database.astype({'2P_calcium_imaging': bool, 'optogenetic': bool,
                     'pharmacology': bool})

    return database


def map_result_columns(behavior_results):

    column_map = {
        # Results.txt columns
        'trialnumber': 'trial_number',
        'WhStimDuration': 'wh_stim_duration',
        'Quietwindow': 'quiet_window',
        'ITI': 'iti',
        'Association': 'association_flag',
        'Stim/NoStim': 'is_stim',
        'Whisker/NoWhisker': 'is_whisker' ,
        'Auditory/NoAuditory': 'is_auditory',
        'Lick': 'lick_flag',
        'Perf': 'perf',
        'Light/NoLight': 'is_light',
        'ReactionTime': 'reaction_time',
        'WhStimAmp': 'wh_stim_amp',
        'TrialTime': 'trial_time',
        'Rew/NoRew': 'is_reward',
        'AudRew': 'aud_reward',
        'WhRew': 'wh_reward',
        'AudDur': 'aud_stim_duration',
        'AudDAmp': 'aud_stim_amp',
        'AudFreq': 'aud_stim_freq',
        'EarlyLick': 'early_lick',
        'LightAmp': 'light_amp',
        'LightDur': 'light_duration',
        'LightFreq': 'light_freq',
        'LightPreStim': 'light_prestim',

        # perf.json columns
        'trial_number': 'trial_number',
        'trial_number_or': 'trial_number_or',
        'whiskerstim_dur': 'wh_stim_duration',
        'quiet_window': 'quiet_window',
        'iti': 'iti',
        'stim_nostim': 'is_stim',
        'wh_nowh': 'is_whisker',
        'aud_noaud': 'is_auditory',
        'lick': 'lick_flag',
        'performance': 'perf',
        'light_nolight': 'is_light',
        'reaction_time': 'reaction_time',
        'whamp': 'wh_stim_amp',
        'trial_time': 'trial_time',
        'rew_norew': 'is_reward',
        'audrew': 'aud_reward',
        'whrew': 'wh_reward',
        'audstimdur': 'aud_stim_duration',
        'audstimamp': 'aud_stim_amp',
        'audstimfreq': 'aud_stim_freq',
        'early_lick': 'early_lick',
        'lightamp': 'light_amp',
        'lightdur': 'light_duration',
        'lightfreq': 'light_freq',
        'lightprestim': 'light_prestim',
    }

    f = lambda x:  column_map[x] if x in column_map.keys() else x
    behavior_results.columns = list(map(f, behavior_results.columns))
    behavior_results = behavior_results.astype({'trial_number': int, 'perf': int})

    # Deal with perf = -1. Label them as early licks.
    behavior_results.loc[behavior_results.perf==-1, 'perf'] = 6

    # Add what is missing.
    if 'association_flag' not in behavior_results.columns:
        behavior_results['association_flag'] = 0
    if 'baseline_window' not in behavior_results.columns:
        behavior_results['baseline_window'] = 2000
    if 'artifact_window' not in behavior_results.columns:
        behavior_results['artifact_window'] = 50
    if 'response_window' not in behavior_results.columns:
        behavior_results['response_window'] = 1000

    return behavior_results


db_folder = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard'
db_name = 'sessions_GF.xlsx'
db = read_excel_database(db_folder, db_name)


mice = db.loc[db.session_day=='0', 'subject_id']
sessions = db.loc[db.session_day=='0', 'session_id']

mouse_name = 'GF333'
session_name = 'GF333_24012021_145617'

rewarded = []
non_rewarded = []

for mouse_name, session_name in zip(mice, sessions):
    perf_json = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                        'Anthony_Renard', 'data', mouse_name, 'Recordings', 'BehaviourData',
                                        session_name, 'performanceResults.json')
    with open(perf_json, 'r') as f:
        perf_json = json.load(f)
    behav = pd.DataFrame(perf_json['results'], columns=perf_json['headers'])
    # Remap GF columns.
    behav = map_result_columns(behav)
    
    if behav.loc[behav.is_whisker==1, 'wh_reward'].product() == 1:
        rewarded.append(mouse_name)
    else:
        non_rewarded.append(mouse_name)
        
len(rewarded)
len(non_rewarded)

mouse_name = 'GF173'
session_name = 'GF173_17012019_091621'
behav.loc[behav.is_whisker==1, 'wh_reward'].plot()
behav.wh_reward.plot()