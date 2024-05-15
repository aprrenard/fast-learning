"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('C:\\Users\\aprenard\\recherches\\repos\\NWB_analysis')
import nwb_utils.server_path as server_path
from analysis.psth_analysis import make_events_aligned_array
from nwb_wrappers import nwb_reader_functions as nwb_read
from src.utils.utils_excel import read_excel_db


# Select NWB files.
# #################

excel_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\session_metadata.xlsx'
db = read_excel_db(excel_path)
experimenters = ['AR']

mice_rew = db.loc[(db['2P_calcium_imaging']==True)
                  & (db.exclude != 'exclude')
                  & (db.reward_group == 'R+'), 'subject_id'].unique()
mice_rew = list(mice_rew)
mice_non_rew = db.loc[(db['2P_calcium_imaging']==True)
                      & (db.exclude != 'exclude')
                      & (db.reward_group == 'R-'), 'subject_id'].unique()
mice_non_rew = list(mice_non_rew)

rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (1,3)
epoch_name = None
behavior_types = ['auditory', 'whisker']
days = [-2, -1, 0, 1, 2]

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = []
# Reduce the number of files, restricting to mice with imaging.
for mouse_id in mice_rew + mice_non_rew:
    nwb_list.extend([nwb for nwb in os.listdir(nwb_path) if mouse_id in nwb])
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])

nwb_list = [nwb for nwb in nwb_list if os.path.basename(nwb)[-25:-23] in experimenters]

nwb_metadata = [nwb_read.get_session_metadata(nwb) for nwb in nwb_list]
nwb_list_rew = [nwb for nwb, metadata in zip(nwb_list, nwb_metadata)
                if (os.path.basename(nwb)[-25:-20] in mice_rew)
                & ('twophoton' in metadata['session_type'])
                & (metadata['behavior_type'] in behavior_types)
                & (metadata['day'] in days)
                ]
nwb_list_non_rew = [nwb for nwb, metadata in zip(nwb_list, nwb_metadata)
                if (os.path.basename(nwb)[-25:-20] in mice_non_rew)
                & ('twophoton' in metadata['session_type'])
                & (metadata['behavior_type'] in behavior_types)
                & (metadata['day'] in days)
                ]


# Find unmotivated trials.
# ########################

# Set criteria for end of session i.e. 1 or 3 auditory misses. After which unmotivated epoch starts.
# Initialise df to check session stop criteria: when it happens and how many non motivated trials that gives.


temp = []
trial_idx_table_GF = []
trial_idx_table_AR = []

# Stop flags GF actually used.
stop_flag_yaml = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\stop_flag_GF.yaml"
with open(stop_flag_yaml, 'r') as stream:
    stop_flags_GF = yaml.load(stream, yaml.Loader)
stop_flags_GF = pd.DataFrame(stop_flags_GF, columns=['session_id', 'stop_flag'])

for nwb_file in nwb_list_rew + nwb_list_non_rew:

    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]
    table = nwb_read.get_trial_table(nwb_file)

    # Specific of GF data.
    if ('GF' in session_id) or ('MI' in session_id):
        flag_trial = stop_flags_GF.loc[stop_flags_GF.session_id==session_id, 'stop_flag'].values[0]
        n_wh_miss = table.loc[(table.trial_id>flag_trial) & (table.whisker_stim == 1) & (table.lick_flag == 0)].shape[0]
        n_wh_hit = table.loc[(table.trial_id>flag_trial) & (table.whisker_stim == 1) & (table.lick_flag == 1)].shape[0]
        
        # Get the indices of those trials.
        non_mot_idx = table.loc[(table.trial_id>flag_trial), 'trial_id'].to_list()
        # Select only the 50 last non-motivated trials (in case more than 50 were presented).
        non_mot_idx = non_mot_idx[-50:]
        trial_idx_table_GF.append([session_id, non_mot_idx])
    else:
        flag_trial = 'na'
        n_wh_miss = 'na'
        n_wh_hit = 'na'

    # Check if it is different from taking the misses once the experimenter played wh trials only
    # at the end of the session.
    # End of motivated block i.e. trial just before start of the 50 non-motivated trials.
    end_mot = (table.whisker_stim != table.whisker_stim.shift()).cumsum().idxmax() - 1
    n_wh_miss_2 = table.loc[(table.trial_id>end_mot) & (table.whisker_stim == 1) & (table.lick_flag == 0)].shape[0]
    n_wh_hit_2 = table.loc[(table.trial_id>end_mot) & (table.whisker_stim == 1) & (table.lick_flag == 1)].shape[0]
        
    # Exceptions. Forgot to remove the auditory trials.
    if session_id == 'AR132_20240427_122605':
        end_mot = 570
        n_wh_miss_2 = table.loc[(table.trial_id>end_mot) & (table.whisker_stim == 1) & (table.lick_flag == 0)].shape[0]
        n_wh_hit_2 = table.loc[(table.trial_id>end_mot) & (table.whisker_stim == 1) & (table.lick_flag == 1)].shape[0]
    
    # Get the indices of those trials.
    non_mot_idx = table.loc[(table.trial_id>end_mot), 'trial_id'].to_list()
    # Select only the 50 last non-motivated trials (in case more than 50 were presented).
    non_mot_idx = non_mot_idx[-50:]
    trial_idx_table_AR.append([session_id, non_mot_idx])

    temp.append([mouse_id, session_id, flag_trial, n_wh_miss, n_wh_hit, end_mot, n_wh_miss_2, n_wh_hit_2,])
stop_flag_table = pd.DataFrame(temp, columns = ['mouse_id', 'session_id', 'stop_flag_GF', 'n_wh_miss_GF', 'n_wh_hit_GF', 'stop_flag_AR', 'n_wh_miss_AR', 'n_wh_hit_AR'])
trial_idx_table_GF = pd.DataFrame(trial_idx_table_GF, columns=['session_id', 'trial_idx'])
trial_idx_table_AR = pd.DataFrame(trial_idx_table_AR, columns=['session_id', 'trial_idx'])


stop_flag_table[['session_id', 'stop_flag_GF', 'n_wh_miss_GF', 'n_wh_hit_GF']]


# Quick plot of behavior outcome.
nwb_file = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\AR132_20240427_122605.nwb'
table = nwb_read.get_trial_table(nwb_file)
table = table.reset_index()

plt.scatter(table.loc[table.auditory_stim==1, 'id'], table.loc[table.auditory_stim==1, 'lick_flag'])
plt.scatter(table.loc[table.whisker_stim==1, 'id'], table.loc[table.whisker_stim==1, 'lick_flag'])
plt.scatter(table.loc[table.no_stim==1, 'id'], table.loc[table.no_stim==1, 'lick_flag'])

nwb_file = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\GF208_01102019_083121.nwb'
trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF208_01102019_083121', 'trial_idx'].values[0]
stop_flag_table.loc[stop_flag_table.session_id=='GF208_01102019_083121']
stop_flag_table.loc[stop_flag_table.mouse_id=='GF208']


# Make and save PSTH numpy arrays.
# --------------------------------

# Exclude mice without non-motivated trials at the end.
excluded = ['GF264', 'GF278', 'GF208', 'GF340']
nwb_list_rew = [nwb_file for nwb_file in nwb_list_rew
                if os.path.basename(nwb_file)[:5] not in excluded]
nwb_list_non_rew = [nwb_file for nwb_file in nwb_list_non_rew
                if os.path.basename(nwb_file)[:5] not in excluded]

trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
traces_rew, metadata_rew = make_events_aligned_array(nwb_list_rew, rrs_keys, time_range,
                                             trial_selection, epoch_name, trial_idx_table_AR)
traces_non_rew, metadata_non_rew = make_events_aligned_array(nwb_list_non_rew, rrs_keys, time_range,
                                             trial_selection, epoch_name, trial_idx_table_AR)

# table = nwb_read.get_trial_table(nwb_file)
# table = table.reset_index()
# table.loc[table['id'].isin(list(np.arange(150,300)))]

# column_name = 'id'
# column_requirements = np.arange(150,300)

# column_values_type = type(table[column_name].values[0])
# column_requirements = [column_values_type(requirement) for requirement in column_requirements]
# test = table.loc[table[column_name].isin(column_requirements)]

# Save activity dict.

save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_AR.npy')
np.save(save_path, traces_rew)
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_AR_metadata.pickle')
metadata_rew['cell_types'] = list(metadata_rew['cell_types'])
with open(save_path, 'wb') as fid:
    pickle.dump(metadata_rew, fid)
    
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_AR.npy')
np.save(save_path, traces_non_rew)
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_AR_metadata.pickle')
metadata_non_rew['cell_types'] = list(metadata_non_rew['cell_types'])
with open(save_path, 'wb') as fid:
    pickle.dump(metadata_non_rew, fid)
    




# # ---------------
# import yaml
# import json
# path = 'C:\\Users\\aprenard\\Downloads\\TrialStop-20240510T122435Z-001\\TrialStop'
# save_yaml = 'C:\\Users\\aprenard\\recherches\\repos\\fast-learning\\docs\\stop_flag_GF.yaml'
# stop_flags = []

# mice = os.listdir(path)
# for mouse in mice:
#     mouse_path = os.path.join(path, mouse)
#     sessions = os.listdir(mouse_path)
#     for session in sessions:
#         session_id = session[:-5]
#         json_path = os.path.join(mouse_path, session)
#         with open(json_path) as f:
#             d = json.load(f)
#             stop_flags.append([session_id, d])
# with open(save_yaml, 'w') as stream:
#     yaml.dump(stop_flags, stream)
