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

# sys.path.append('H:\\anthony\\repos\\NWB_analysis')
import nwb_utils.utils_io as io
from analysis.psth_analysis import (make_events_aligned_array_6d,
                                   make_events_aligned_array_3d)
from nwb_wrappers import nwb_reader_functions as nwb_read


# =============================================================================
# Make activity 6d array of sensory mapping trials (non-motivated trials).
# =============================================================================

group_yaml = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/groups/imaging_non_rewarded.yaml"
trial_indices_yaml = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/stop_flags/trial_indices_sensory_map.yaml"
processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/'
                      r'Anthony_Renard/data_processed/')
dataset_name = 'psth_sensory_map_trials_non_rewarded.npy'
metadata_name = 'psth_sensory_map_trials_non_rewarded.pickle'

with open(group_yaml, 'r') as stream:
    nwb_list = yaml.safe_load(stream)
with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# trial_indices = None
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
cell_types = ['na', 'wM1', 'wS2']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (1,3)
epoch_name = None


traces, metadata = make_events_aligned_array(nwb_list, rrs_keys,
                                             time_range, trial_selection,
                                             epoch_name, cell_types,
                                             trial_indices)

# Save dataset.
save_path = os.path.join(processed_data_dir, dataset_name)
np.save(save_path, traces)
save_path = os.path.join(processed_data_dir, metadata_name)
# metadata['cell_types'] = list(metadata['cell_types'])
with open(save_path, 'wb') as fid:
    pickle.dump(metadata, fid)


# =============================================================================
# Make activity 6d array of each trial type (motivated).
# =============================================================================

# Rewarded and non-rewarded NWB files.
group_yaml_rew = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/groups/imaging_rewarded.yaml"
group_yaml_non_rew = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/groups/imaging_non_rewarded.yaml"

# Stop flags for each session.
trial_indices_yaml = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/stop_flags/trial_indices_end_session.yaml"

processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/'
                      r'Anthony_Renard/data_processed/')

dataset_names = ['psth_WH.npy',
                 'psth_WM.npy',
                 'psth_AH.npy',
                 'psth_AM.npy',
                 'psth_FA.npy',
                 'psth_CR.npy',
                 ]
metadata_names = ['psth_WH_metadata.pickle',
                 'psth_WM_metadata.pickle',
                 'psth_AH_metadata.pickle',
                 'psth_AM_metadata.pickle',
                 'psth_FA_metadata.pickle',
                 'psth_CR_metadata.pickle',
                 ]
trial_selections = [{'whisker_stim': [1], 'lick_flag':[1]},
                    {'whisker_stim': [1], 'lick_flag':[0]},
                    {'auditory_stim': [1], 'lick_flag':[1]},
                    {'auditory_stim': [1], 'lick_flag':[0]},
                    {'no_stim': [1], 'lick_flag':[1]},
                    {'no_stim': [1], 'lick_flag':[0]},
                    ]

with open(group_yaml_rew, 'r') as stream:
    nwb_list_rew = yaml.safe_load(stream)
with open(group_yaml_non_rew, 'r') as stream:
    nwb_list_non_rew = yaml.safe_load(stream)
nwb_list = nwb_list_rew + nwb_list_non_rew

with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])

# Parameters for PSTH.
cell_types = ['na', 'wM1', 'wS2']
rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (1,5)
epoch_name = None

# Loop on each trial type and save the dataset.
for dataset_name, metadata_name, trial_selection in zip(dataset_names,
                                                        metadata_names,
                                                        trial_selections):
    print(f'Processing {dataset_name}')
    traces, metadata = make_events_aligned_array_6d(nwb_list, rrs_keys,
                                                 time_range, trial_selection,
                                                 epoch_name, cell_types,
                                                 trial_indices)
    # Save dataset.
    save_path = os.path.join(processed_data_dir, dataset_name)
    np.save(save_path, traces)
    save_path = os.path.join(processed_data_dir, metadata_name)
    # metadata['cell_types'] = list(metadata['cell_types'])
    with open(save_path, 'wb') as fid:
        pickle.dump(metadata, fid)


data_path =r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\psth_WH.npy"
metadata_path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\psth_WH_metadata.pickle"
with open(metadata_path, 'rb') as fid:
    metadata = pickle.load(fid)
data = np.load(data_path)




# =============================================================================
# Create a 4d array for each sessions.
# =============================================================================

# Rewarded and non-rewarded NWB files.
group_yaml_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_rewarded.yaml"
group_yaml_non_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_non_rewarded.yaml"

# Trial indices for each session.
trial_indices_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_end_session.yaml"
trial_indices_sensory_map_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_sensory_map.yaml"

processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/'
                      r'Anthony_Renard/data_processed/mice')

with open(group_yaml_rew, 'r') as stream:
    nwb_list_rew = yaml.safe_load(stream)
with open(group_yaml_non_rew, 'r') as stream:
    nwb_list_non_rew = yaml.safe_load(stream)
nwb_list = nwb_list_rew + nwb_list_non_rew

with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# For "non motivated" sensory mapping trials at the end of the session.
with open(trial_indices_sensory_map_yaml, 'r') as stream:
    trial_indices_sensory_map = yaml.load(stream, yaml.Loader)
trial_indices_sensory_map = pd.DataFrame(trial_indices_sensory_map.items(), columns=['session_id', 'trial_idx'])

for nwb_file in nwb_list:
    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]

    # Parameters for tensor array.
    cell_types = ['na', 'wM1', 'wS2']
    rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
    time_range = (1,5)
    epoch_name = None

    trial_selections = [{'whisker_stim': [1], 'lick_flag':[1]},
                        {'whisker_stim': [1], 'lick_flag':[0]},
                        {'auditory_stim': [1], 'lick_flag':[1]},
                        {'auditory_stim': [1], 'lick_flag':[0]},
                        {'no_stim': [1], 'lick_flag':[1]},
                        {'no_stim': [1], 'lick_flag':[0]},
                        {'whisker_stim': [1], 'lick_flag':[0]},
                        ]
    trial_type_labels = ['WH', 'WM', 'AH', 'AM', 'FA', 'CR', 'UM']

    idx = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]
    idx_sensory_map = trial_indices_sensory_map.loc[trial_indices_sensory_map.session_id==session_id, 'trial_idx'].values[0]
    trial_idx_selections = [idx for _ in range(6)] + [idx_sensory_map]

    # Generate a 3d array for each trial type.
    stack = []
    metadatas = []
    for trial_selection, trial_idx_selection in zip(trial_selections, trial_idx_selections):
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(nwb_file,
                                                        rrs_keys,
                                                        time_range,
                                                        trial_selection,
                                                        epoch_name,
                                                        cell_types,
                                                        trial_idx_selection)
        stack.append(traces)
        metadatas.append(metadata)

    # Remove None given by empty trial types.
    trial_type_labels = [trial_type_labels[i] for i in range(len(trial_type_labels)) if stack[i] is not None]
    metadatas = [metadata for metadata in metadatas if metadata is not None]
    stack = [traces for traces in stack if traces is not None]
    # Concatenate trial indices for each trial type.
    metadata_trials = [metadatas[i]['trials'] for i in range(len(metadatas))]
    metadata_trials = np.concatenate(metadata_trials, axis=0)
    # Note that metadata is the same for all trial types
    # (same cells) except for trial ids.
    metadata_stacked = metadatas[0]
    metadata_stacked['trials'] = metadata_trials
    metadata_stacked['trial_types'] = trial_type_labels

    # Stack trial type to shape (n_cells, n_trial_type, n_trials, n_t).
    max_trials = max([a.shape[1] for a in stack])
    new_shape = (stack[0].shape[0],  len(stack), max_trials, stack[0].shape[2])
    tensor = np.full(new_shape, np.nan)
    for i, arr in enumerate(stack):
        tensor[:, i, :arr.shape[1], :] = arr
    # Reduce precision to save space.
    tensor = tensor.astype(np.float16)

    # Save dataset.
    save_dir = os.path.join(processed_data_dir, mouse_id, session_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_4d.npy')
    np.save(save_path, tensor)
    save_path = os.path.join(save_dir, 'tensor_4d_metadata.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(metadata_stacked, f)


# # Make pandas dataset to check that I have the same than with numpy.
# # ##################################################################

# trial_idx = trial_idx_table_GF
# trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
# cell_types = ['na', 'wM1', 'wS2']

# traces_rew_pd = make_events_aligned_data_table(nwb_list_rew, rrs_keys, time_range,
#                                                 trial_selection, epoch_name, True, trial_idx)
# traces_non_rew_pd = make_events_aligned_data_table(nwb_list_non_rew, rrs_keys, time_range,
#                                                 trial_selection, epoch_name, True, trial_idx)

# stop_flag = {session_id: trial_idx_table_GF.loc[trial_idx_table_GF.session_id==session_id, 'trial_idx'].values[0]
#              for session_id in trial_idx_table_GF.session_id.unique()}
# save_path = '////sv-nas1.rcp.epfl.ch//Petersen-Lab/\analysis\\Anthony_Renard\\data_processed\\stop_flag_GF.yaml'
# with open(save_path, 'w') as outfile:
#     yaml.safe_dump(stop_flag, outfile)

# with open(save_path, 'r') as fid:
#     test = yaml.safe_load(fid)

# trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF264_02072020_104646', 'trial_idx'].to_list()


# # traces_non_rew_pd.loc[(traces_non_rew_pd.mouse_id.isin(['GF208']))].cell_type.unique()

# # # traces_non_rew_pd.loc[(traces_non_rew_pd.behavior_day.isin([0])) & (d.roi==0) & (d.event==3)]
# # event = 0
# # d = traces_non_rew_pd.loc[(traces_non_rew_pd.mouse_id.isin(['GF208']))
# #                           &  (traces_non_rew_pd.behavior_day.isin([0]))
# #                           & (traces_non_rew_pd.roi==0)
# #                           & (traces_non_rew_pd.event==event)]
# # plt.figure()
# # plt.plot(d.activity.to_numpy())

# # traces_non_rew_np.shape
# # psth = traces_non_rew_np[0,2,0,0,event]
# # plt.plot(psth.flatten())



# # # Compare with data from nwb file.
# # nwb_file = nwb_list_non_rew[2]
# # rrs = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
# # table = nwb_read.get_trial_table(nwb_file)

# # trial_idx_table_GF.loc[trial_idx_table_GF.session_id=='GF208_02102019_094122', 'trial_idx'].values[0]

# # event = 769

# # start_time = table.loc[table.trial_id==event].start_time.values[0]
# # rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])

# # central_frame = find_nearest(rrs_ts, start_time+2)
# # nwb_d = rrs[0,central_frame-30:central_frame+90]
# # nwb_d -= np.mean(nwb_d[0:30], axis=0)

# # plt.figure()
# # plt.plot(nwb_d)


# d = traces_rew_pd.loc[traces_rew_pd.behavior_day.isin([-1,1])]
# d = d.groupby(['mouse_id','session_id', 'behavior_type', 'behavior_day', 'roi', 'time'], as_index=False).agg({'activity':np.nanmean})
# # d = d.groupby(['behavior_day', 'time'], as_index=False).agg({'activity':np.nanmean})
# plt.figure()
# # plt.plot(np.mean(d.activity.to_numpy().reshape((2,121)), axis=0))
# # plt.plot(d.activity)
# sns.lineplot(data=d, x='time', y="activity", hue='behavior_day', errorbar=None, estimator=np.nanmean)

# d = traces_non_rew_pd.loc[traces_non_rew_pd.behavior_day.isin([-1,1])]
# d = d.groupby(['mouse_id','session_id', 'behavior_type', 'behavior_day', 'roi', 'time'], as_index=False).agg({'activity':np.nanmean})
# # d = d.groupby(['behavior_day', 'time'], as_index=False).agg({'activity':np.nanmean})
# plt.figure()
# # plt.plot(np.mean(d.activity.to_numpy().reshape((2,121)), axis=0))
# # plt.plot(d.activity)
# sns.lineplot(data=d, x='time', y="activity", hue='behavior_day', errorbar=None, estimator=np.nanmean)

# psth = np.nanmean(traces_non_rew_np, axis=(4))
# days = np.nanmean(psth, axis=(0,2,3))

# plt.figure()
# plt.plot(days[0])
# plt.plot(days[1])
# plt.plot(days[2])
# plt.plot(days[3])
# plt.plot(days[4])



# # Checking that data correspond for GF.
# # #####################################

# import json

# nwb_file = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\NWB\AR144_20240518_193553.nwb'
# rrs_fissa = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'F_cor'])
# rrs_dff = nwb_read.get_roi_response_serie_data(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
# rrs_dff.shape
# fissa_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FISSASessionData\GF264\GF264_02072020\F_fissa.npy"
# baseline_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\Baselines\GF264\GF264_02072020\baselines.npy"
# dff_json = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FoustoukosData\Data\GF264\Recordings\CalciumData\GF264_02072020_104646\trial_0252\dffRois.json"
# fissa = np.load(fissa_file, allow_pickle=True)
# f0 = np.load(baseline_file, allow_pickle=True)
# with open(dff_json, 'r') as stream:
#     dff = json.load(stream)
# dff = np.array(dff)

# result_file = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Georgios_Foustoukos\FoustoukosData\Data\GF264\Recordings\BehaviourFiles\GF264_02072020_104646\Results.txt'
# df = pd.read_csv(result_file, sep=r'\s+', engine='python')
# df.trial_number
# df.loc[df.Perf==6]
# df.loc[df.Perf!=6]

# plt.figure()
# plt.plot(np.mean(rrs_dff[:,-303:], axis=0))
# plt.plot(np.mean(dff[:,:], axis=0))

# plt.figure()
# plt.plot(np.mean(rrs_dff[:,-303:], axis=0)[30:150])
# plt.plot(np.mean(dff[:,:], axis=0)[30:150])



# plt.plot(rrs_dff[0])
# trial_table = nwb_read.get_trial_table(nwb_file)
# trial_table.trial_id

# ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, ['ophys', 'fluorescence_all_cells', 'dff'])
# ts.shape
# rrs_dff.shape

# ts[-303]
# # last frame 3404.8333333333335

# traces_rew

# # Flatten mean response across cells
# a = np.nanmean(traces_rew[1,1,:,:,:52,:], axis=(0,1))
# a.shape
# a = a.flatten()
# b = np.nanmean(rrs_dff, axis=0)

# f, axes = plt.subplots(2,1)
# axes[0].plot(b[-303:])
# axes[1].plot(a[-120:])


# a.shape
# b.shape
# rrs_dff.shape

# metadata_rew

# traces_rew.shape
# np.sum(~(np.isnan(traces_rew[1,1,0,0,:,0])))
# np.sum(~(np.isnan(traces_rew[1,1,:,:,0,0])))



# path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR144\suite2p\plane0\F.npy"
# d = np.load(path)

# plt.plot(d[0])
# # 3260.900000


# # # ---------------
# # import yaml
# # import json
# # path = 'C:\\Users\\aprenard\\Downloads\\TrialStop-20240510T122435Z-001\\TrialStop'
# # save_yaml = 'C:\\Users\\aprenard\\recherches\\repos\\fast-learning\\docs\\stop_flag_GF.yaml'
# # stop_flags = []

# # mice = os.listdir(path)
# # for mouse in mice:
# #     mouse_path = os.path.join(path, mouse)
# #     sessions = os.listdir(mouse_path)
# #     for session in sessions:
# #         session_id = session[:-5]
# #         json_path = os.path.join(mouse_path, session)
# #         with open(json_path) as f:
# #             d = json.load(f)
# #             stop_flags.append([session_id, d])
# # with open(save_yaml, 'w') as stream:
# #     yaml.dump(stop_flags, stream)
