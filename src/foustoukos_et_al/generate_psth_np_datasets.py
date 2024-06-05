"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from datetime import datetime

sys.path.append('H:\\anthony\\repos\\NWB_analysis')
import nwb_utils.server_path as server_path
from analysis.psth_analysis import make_events_aligned_array, make_events_aligned_data_table
from nwb_wrappers import nwb_reader_functions as nwb_read
from src.utils.utils_excel import read_excel_db


# Select NWB files.
# #################

excel_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\session_metadata.xlsx'
db = read_excel_db(excel_path)
experimenters = ['AR', 'GF', 'MI']

mice_rew = db.loc[(db['2P_calcium_imaging']==True)
                  & (db.exclude != 'exclude')
                  & (db.reward_group == 'R+'), 'subject_id'].unique()
mice_rew = list(mice_rew)
mice_non_rew = db.loc[(db['2P_calcium_imaging']==True)
                      & (db.exclude != 'exclude')
                      & (db.reward_group == 'R-'), 'subject_id'].unique()
mice_non_rew = list(mice_non_rew)

behavior_types = ['auditory', 'whisker']
days = [-2, -1, 0, 1, 2]

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = []
# Reduce the number of files, restricting to mice with imaging.
for mouse_id in mice_rew + mice_non_rew:
    nwb_list.extend([nwb for nwb in os.listdir(nwb_path) if mouse_id in nwb])
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])

nwb_list = [nwb for nwb in nwb_list if os.path.basename(nwb)[-25:-23] in experimenters]

nwb_metadata = {nwb: nwb_read.get_session_metadata(nwb) for nwb in nwb_list}
nwb_list_rew = [nwb for nwb, metadata in nwb_metadata.items()
                if (os.path.basename(nwb)[-25:-20] in mice_rew)
                & ('twophoton' in metadata['session_type'])
                & (metadata['behavior_type'] in behavior_types)
                & (metadata['day'] in days)
                ]
nwb_list_non_rew = [nwb for nwb, metadata in nwb_metadata.items()
                if (os.path.basename(nwb)[-25:-20] in mice_non_rew)
                & ('twophoton' in metadata['session_type'])
                & (metadata['behavior_type'] in behavior_types)
                & (metadata['day'] in days)
                ]

def get_date(x):
    if x[-25:-23] in ['GF', 'MI']:
        return datetime.strptime(x[-19:-11], '%d%m%Y')
    else:
        return datetime.strptime(x[-19:-11], '%Y%m%d')

# Reorder nwb list in chronological order mouse wise (GF does not use americain dates).
temp = []
for mouse in mice_rew:
    l = [nwb for nwb in nwb_list_rew if mouse in nwb]
    l = sorted(l, key=get_date)
    temp.extend(l)
nwb_list_rew = temp

temp = []
for mouse in mice_non_rew:
    l = [nwb for nwb in nwb_list_non_rew if mouse in nwb]
    l = sorted(l, key=get_date)
    temp.extend(l)
nwb_list_non_rew = temp


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
        n_wh_miss = table.loc[(table.trial_id>=flag_trial) & (table.whisker_stim == 1) & (table.lick_flag == 0)].shape[0]
        n_wh_hit = table.loc[(table.trial_id>=flag_trial) & (table.whisker_stim == 1) & (table.lick_flag == 1)].shape[0]
        
        # Get the indices of those trials.
        non_mot_idx = table.loc[(table.trial_id>=flag_trial), 'trial_id'].to_list()
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


# # Quick plot of behavior outcome.
# nwb_file = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\GF264_02072020_104646.nwb'
# table = nwb_read.get_trial_table(nwb_file)
# table = table.reset_index()

# plt.figure()
# plt.scatter(table.loc[table.auditory_stim==1, 'id'], table.loc[table.auditory_stim==1, 'lick_flag'])
# plt.scatter(table.loc[table.whisker_stim==1, 'id'], table.loc[table.whisker_stim==1, 'lick_flag'])
# plt.scatter(table.loc[table.no_stim==1, 'id'], table.loc[table.no_stim==1, 'lick_flag'])

# nwb_file = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\GF208_01102019_083121.nwb'
# trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF208_01102019_083121', 'trial_idx'].values[0]
# stop_flag_table.loc[stop_flag_table.session_id=='GF208_01102019_083121']
# stop_flag_table.loc[stop_flag_table.mouse_id=='GF208']


# Make and save PSTH numpy arrays.
# --------------------------------

# Exclude mice without non-motivated trials at the end.
excluded = ['GF264', 'GF278', 'GF208', 'GF340']
nwb_list_rew = [nwb_file for nwb_file in nwb_list_rew
                if os.path.basename(nwb_file)[:5] not in excluded]
nwb_list_non_rew = [nwb_file for nwb_file in nwb_list_non_rew
                if os.path.basename(nwb_file)[:5] not in excluded]

trial_idx = trial_idx_table_AR
# trial_idx = None
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
cell_types = ['na', 'wM1', 'wS2']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (1,3)
epoch_name = None

traces_rew, metadata_rew = make_events_aligned_array(nwb_list_rew, rrs_keys, time_range,
                                                     trial_selection, epoch_name, cell_types, trial_idx)
traces_non_rew, metadata_non_rew = make_events_aligned_array(nwb_list_non_rew, rrs_keys, time_range,
                                                             trial_selection, epoch_name, cell_types, trial_idx)


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
             'data_processed\\traces_non_motivated_trials_rew_common.npy')
np.save(save_path, traces_rew)
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_common_metadata.pickle')
metadata_rew['cell_types'] = list(metadata_rew['cell_types'])
with open(save_path, 'wb') as fid:
    pickle.dump(metadata_rew, fid)

save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_common.npy')
np.save(save_path, traces_non_rew)
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_common_metadata.pickle')
metadata_non_rew['cell_types'] = list(metadata_non_rew['cell_types'])
with open(save_path, 'wb') as fid:
    pickle.dump(metadata_non_rew, fid)


# save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
#              'data_processed\\traces_non_motivated_trials_rew_GF_epoch.npy')
# np.save(save_path, traces_rew)

    
# save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
#              'data_processed\\traces_non_motivated_trials_non_rew_GF_epoch.npy')
# np.save(save_path, traces_non_rew)


# Generate dataset of motivated trials with all three stimulus types.
# ###################################################################






# Make pandas dataset to check that I have the same than with numpy.
# ##################################################################

trial_idx = trial_idx_table_GF
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
cell_types = ['na', 'wM1', 'wS2']

traces_rew_pd = make_events_aligned_data_table(nwb_list_rew, rrs_keys, time_range,
                                                trial_selection, epoch_name, True, trial_idx)
traces_non_rew_pd = make_events_aligned_data_table(nwb_list_non_rew, rrs_keys, time_range,
                                                trial_selection, epoch_name, True, trial_idx)

stop_flag = {session_id: trial_idx_table_GF.loc[trial_idx_table_GF.session_id==session_id, 'trial_idx'].values[0]
             for session_id in trial_idx_table_GF.session_id.unique()}
save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data_processed\\stop_flag_GF.yaml'
with open(save_path, 'w') as outfile:
    yaml.safe_dump(stop_flag, outfile)

with open(save_path, 'r') as fid:
    test = yaml.safe_load(fid)

trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF264_02072020_104646', 'trial_idx'].to_list()


# traces_non_rew_pd.loc[(traces_non_rew_pd.mouse_id.isin(['GF208']))].cell_type.unique()

# # traces_non_rew_pd.loc[(traces_non_rew_pd.behavior_day.isin([0])) & (d.roi==0) & (d.event==3)]
# event = 0
# d = traces_non_rew_pd.loc[(traces_non_rew_pd.mouse_id.isin(['GF208']))
#                           &  (traces_non_rew_pd.behavior_day.isin([0]))
#                           & (traces_non_rew_pd.roi==0)
#                           & (traces_non_rew_pd.event==event)]
# plt.figure()
# plt.plot(d.activity.to_numpy())

# traces_non_rew_np.shape
# psth = traces_non_rew_np[0,2,0,0,event]
# plt.plot(psth.flatten())



# # Compare with data from nwb file.
# nwb_file = nwb_list_non_rew[2]
# rrs = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
# table = nwb_read.get_trial_table(nwb_file)

# trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF208_02102019_094122', 'trial_idx'].values[0]

# event = 769

# start_time = table.loc[table.trial_id==event].start_time.values[0]
# rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])

# central_frame = find_nearest(rrs_ts, start_time+2)
# nwb_d = rrs[0,central_frame-30:central_frame+90]
# nwb_d -= np.mean(nwb_d[0:30], axis=0)

# plt.figure()
# plt.plot(nwb_d)


d = traces_rew_pd.loc[traces_rew_pd.behavior_day.isin([-1,1])]
d = d.groupby(['mouse_id','session_id', 'behavior_type', 'behavior_day', 'roi', 'time'], as_index=False).agg({'activity':np.nanmean})
# d = d.groupby(['behavior_day', 'time'], as_index=False).agg({'activity':np.nanmean})
plt.figure()
# plt.plot(np.mean(d.activity.to_numpy().reshape((2,121)), axis=0))
# plt.plot(d.activity)
sns.lineplot(data=d, x='time', y="activity", hue='behavior_day', errorbar=None, estimator=np.nanmean)

d = traces_non_rew_pd.loc[traces_non_rew_pd.behavior_day.isin([-1,1])]
d = d.groupby(['mouse_id','session_id', 'behavior_type', 'behavior_day', 'roi', 'time'], as_index=False).agg({'activity':np.nanmean})
# d = d.groupby(['behavior_day', 'time'], as_index=False).agg({'activity':np.nanmean})
plt.figure()
# plt.plot(np.mean(d.activity.to_numpy().reshape((2,121)), axis=0))
# plt.plot(d.activity)
sns.lineplot(data=d, x='time', y="activity", hue='behavior_day', errorbar=None, estimator=np.nanmean)

psth = np.nanmean(traces_non_rew_np, axis=(4))
days = np.nanmean(psth, axis=(0,2,3))

plt.figure()
plt.plot(days[0])
plt.plot(days[1])
plt.plot(days[2])
plt.plot(days[3])
plt.plot(days[4])



# Checking that data correspond for GF.
# #####################################

import json

nwb_file = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\NWB\AR144_20240518_193553.nwb'
rrs_fissa = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'F_cor'])
rrs_dff = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
rrs_dff.shape
fissa_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FISSASessionData\GF264\GF264_02072020\F_fissa.npy"
baseline_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\Baselines\GF264\GF264_02072020\baselines.npy"
dff_json = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FoustoukosData\Data\GF264\Recordings\CalciumData\GF264_02072020_104646\trial_0252\dffRois.json"
fissa = np.load(fissa_file, allow_pickle=True)
f0 = np.load(baseline_file, allow_pickle=True)
with open(dff_json, 'r') as stream:
    dff = json.load(stream)
dff = np.array(dff)

result_file = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FoustoukosData\Data\GF264\Recordings\BehaviourFiles\GF264_02072020_104646\Results.txt'
df = pd.read_csv(result_file, sep=r'\s+', engine='python')
df.trial_number
df.loc[df.Perf==6]
df.loc[df.Perf!=6]

plt.figure()
plt.plot(np.mean(rrs_dff[:,-303:], axis=0))
plt.plot(np.mean(dff[:,:], axis=0))

plt.figure()
plt.plot(np.mean(rrs_dff[:,-303:], axis=0)[30:150])
plt.plot(np.mean(dff[:,:], axis=0)[30:150])



plt.plot(rrs_dff[0])
trial_table = nwb_read.get_trial_table(nwb_file)
trial_table.trial_id

ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
ts.shape
rrs_dff.shape

ts[-303]
# last frame 3404.8333333333335

traces_rew

# Flatten mean response across cells
a = np.nanmean(traces_rew[1,1,:,:,:52,:], axis=(0,1))
a.shape
a = a.flatten()
b = np.nanmean(rrs_dff, axis=0)

f, axes = plt.subplots(2,1)
axes[0].plot(b[-303:])
axes[1].plot(a[-120:])


a.shape
b.shape
rrs_dff.shape

metadata_rew

traces_rew.shape
np.sum(~(np.isnan(traces_rew[1,1,0,0,:,0])))
np.sum(~(np.isnan(traces_rew[1,1,:,:,0,0])))



path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR144\suite2p\plane0\F.npy"
d = np.load(path)

plt.plot(d[0])
# 3260.900000


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
