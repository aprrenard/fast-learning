"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import xarray as xr
import scipy.stats as stats

# sys.path.append('H:\\anthony\\repos\\NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as imaging_utils
from analysis.psth_analysis import (make_events_aligned_array_6d,
                                   make_events_aligned_array_3d)
from nwb_wrappers import nwb_reader_functions as nwb_read
from src.behavior import make_behavior_table
from src.utils.utils_imaging import compute_lmi


def test_response(data, trial_selection, response_win, baseline_win):
    """Test if cells are responsive to a given trial type
    with a Wilcoxon test comparing response versus baseline.
    """
    # Select trials.
    for key, val in trial_selection.items():
        data = data.sel(trial=data.coords[key] == val)
    # If no trials of that type, return nan for all cells.
    if data.shape[1] == 0:
        return np.full(data.shape[0], np.nan)
    
    # Select time window.
    response = data.sel(time=slice(*response_win)).mean(dim='time')    
    baseline = data.sel(time=slice(*baseline_win)).mean(dim='time')

    # Test if response is significant for each cell.
    pval = np.zeros(response.shape[0])
    for cell in range(response.shape[0]):
        # Special case of a artefactual cell with 0 at all time points.
        if (response[cell]==0).all() or (baseline[cell]==0).all():
            pval[cell] = 1
            continue
        t, p = stats.wilcoxon(response[cell], baseline[cell])
        pval[cell] = p
        if np.isnan(p):
            print(f'Cell {cell} has NaN values.')       
    return pval


# =============================================================================
# Test responsive cells.
# =============================================================================

# db_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\mice_info\session_metadata.xlsx'
# db = io.read_excel_db(db_path)


# def filter_nwb_based_on_excel_db(db_path, nwb_path, experimenters, exclude_cols, **filters):
    
#     nwb_list = io.read_excel_db(db_path)
#     for key, val in filters.items():
#         if type(val) is list:
#             nwb_list = nwb_list.loc[(nwb_list[key].isin(val))]
#         else:
#             nwb_list = nwb_list.loc[(nwb_list[key]==val)]

#     # Remove excluded sessions.
#     for col in exclude_cols:
#         nwb_list = nwb_list.loc[(nwb_list[col]!='exclude')]
    
#     nwb_list = list(nwb_list.session_id)
#     nwb_list = [os.path.join(nwb_path, f + '.nwb') for f in nwb_list]

#     if experimenters:
#         nwb_list = [nwb for nwb in nwb_list if os.path.basename(nwb)[-25:-23] in experimenters]

#     return nwb_list





# # =============================================================================
# # Make activity 6d array of sensory mapping trials (non-motivated trials).
# # =============================================================================

# # Change those for R+/R-.
# group_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_rewarded.yaml"
# dataset_name = 'psth_sensory_map_trials_rewarded.npy'
# metadata_name = 'psth_sensory_map_trials_rewarded_metadata.pickle'

# trial_indices_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_sensory_map.yaml"
# processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/'
#                       r'Anthony_Renard/data_processed/')

# with open(group_yaml, 'r') as stream:
#     nwb_list = yaml.safe_load(stream)
# with open(trial_indices_yaml, 'r') as stream:
#     trial_indices = yaml.load(stream, yaml.Loader)
# trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# # trial_indices = None
# trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
# cell_types = ['na', 'wM1', 'wS2']
# rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
# time_range = (1,3)
# epoch_name = None


# traces, metadata = make_events_aligned_array_6d(nwb_list, rrs_keys,
#                                              time_range, trial_selection,
#                                              epoch_name, cell_types,
#                                              trial_indices)

# # Save dataset.
# save_path = os.path.join(processed_data_dir, dataset_name)
# np.save(save_path, traces)
# save_path = os.path.join(processed_data_dir, metadata_name)
# # metadata['cell_types'] = list(metadata['cell_types'])
# with open(save_path, 'wb') as fid:
#     pickle.dump(metadata, fid)


# # =============================================================================
# # Make activity 6d array of each trial type (motivated).
# # =============================================================================

# # Rewarded and non-rewarded NWB files.
# group_yaml_rew = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/groups/imaging_rewarded.yaml"
# group_yaml_non_rew = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/groups/imaging_non_rewarded.yaml"

# # Stop flags for each session.
# trial_indices_yaml = r"C:/Users/aprenard/recherches/repos/fast-learning/docs/stop_flags/trial_indices_end_session.yaml"

# processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/'
#                       r'Anthony_Renard/data_processed/')

# dataset_names = ['psth_WH.npy',
#                  'psth_WM.npy',
#                  'psth_AH.npy',
#                  'psth_AM.npy',
#                  'psth_FA.npy',
#                  'psth_CR.npy',
#                  ]
# metadata_names = ['psth_WH_metadata.pickle',
#                  'psth_WM_metadata.pickle',
#                  'psth_AH_metadata.pickle',
#                  'psth_AM_metadata.pickle',
#                  'psth_FA_metadata.pickle',
#                  'psth_CR_metadata.pickle',
#                  ]
# trial_selections = [{'whisker_stim': [1], 'lick_flag':[1]},
#                     {'whisker_stim': [1], 'lick_flag':[0]},
#                     {'auditory_stim': [1], 'lick_flag':[1]},
#                     {'auditory_stim': [1], 'lick_flag':[0]},
#                     {'no_stim': [1], 'lick_flag':[1]},
#                     {'no_stim': [1], 'lick_flag':[0]},
#                     ]

# with open(group_yaml_rew, 'r') as stream:
#     nwb_list_rew = yaml.safe_load(stream)
# with open(group_yaml_non_rew, 'r') as stream:
#     nwb_list_non_rew = yaml.safe_load(stream)
# nwb_list = nwb_list_rew + nwb_list_non_rew

# with open(trial_indices_yaml, 'r') as stream:
#     trial_indices = yaml.load(stream, yaml.Loader)
# trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])

# # Parameters for PSTH.
# cell_types = ['na', 'wM1', 'wS2']
# rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
# time_range = (1,5)
# epoch_name = None

# # Loop on each trial type and save the dataset.
# for dataset_name, metadata_name, trial_selection in zip(dataset_names,
#                                                         metadata_names,
#                                                         trial_selections):
#     print(f'Processing {dataset_name}')
#     traces, metadata = make_events_aligned_array_6d(nwb_list, rrs_keys,
#                                                  time_range, trial_selection,
#                                                  epoch_name, cell_types,
#                                                  trial_indices)
#     # Save dataset.
#     save_path = os.path.join(processed_data_dir, dataset_name)
#     np.save(save_path, traces)
#     save_path = os.path.join(processed_data_dir, metadata_name)
#     # metadata['cell_types'] = list(metadata['cell_types'])
#     with open(save_path, 'wb') as fid:
#         pickle.dump(metadata, fid)


# data_path =r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\psth_WH.npy"
# metadata_path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\psth_WH_metadata.pickle"
# with open(metadata_path, 'rb') as fid:
#     metadata = pickle.load(fid)
# data = np.load(data_path)



# # =============================================================================
# # Create a 4d array for each sessions.
# # =============================================================================

# # Rewarded and non-rewarded NWB files.
# group_yaml_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_rewarded.yaml"
# group_yaml_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_rewarded.yaml"
# group_yaml_non_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_non_rewarded.yaml"

# # Trial indices for each session.
# trial_indices_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_end_session.yaml"
# trial_indices_sensory_map_yaml = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/stop_flags/trial_indices_sensory_map.yaml"

# processed_data_dir = (r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/mice')

# with open(group_yaml_rew, 'r') as stream:
#     nwb_list_rew = yaml.safe_load(stream)
# with open(group_yaml_non_rew, 'r') as stream:
#     nwb_list_non_rew = yaml.safe_load(stream)
# nwb_list = nwb_list_rew + nwb_list_non_rew



# with open(trial_indices_yaml, 'r') as stream:
#     trial_indices = yaml.load(stream, yaml.Loader)
# trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# # For "non motivated" sensory mapping trials at the end of the session.
# with open(trial_indices_sensory_map_yaml, 'r') as stream:
#     trial_indices_sensory_map = yaml.load(stream, yaml.Loader)
# trial_indices_sensory_map = pd.DataFrame(trial_indices_sensory_map.items(), columns=['session_id', 'trial_idx'])

# nwb_list = [nwb for nwb in nwb_list if 'AR163' in nwb]

# for nwb_file in nwb_list:
    
    
#     mouse_id = nwb_file[-25:-20]
#     session_id = nwb_file[-25:-4]
    
#     # Check if dataset is already created.
#     save_dir = os.path.join(processed_data_dir, mouse_id, session_id)
#     os.makedirs(save_dir, exist_ok=True)
#     save_path_data = os.path.join(save_dir, 'tensor_4d.npy')
#     save_path_metadata = os.path.join(save_dir, 'tensor_4d_metadata.pickle')
#     # if os.path.exists(save_path_data):
#     #     continue

#     # Parameters for tensor array.
#     cell_types = ['na', 'wM1', 'wS2']
#     rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
#     time_range = (1,5)
#     epoch_name = None

#     trial_selections = [{'whisker_stim': [1], 'lick_flag':[1]},
#                         {'whisker_stim': [1], 'lick_flag':[0]},
#                         {'auditory_stim': [1], 'lick_flag':[0]},
#                         {'auditory_stim': [1], 'lick_flag':[1]},
#                         {'no_stim': [1], 'lick_flag':[1]},
#                         {'no_stim': [1], 'lick_flag':[0]},
#                         {'whisker_stim': [1]},
#                         {'auditory_stim': [1]},
#                         {'no_stim': [1]},
#                         {'whisker_stim': [1], 'lick_flag':[0]},
#                         ]
#     trial_type_labels = ['WH', 'WM', 'AH', 'AM', 'FA', 'CR', 'W', 'A', 'NS', 'UM']

#     idx = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]
#     idx_sensory_map = trial_indices_sensory_map.loc[trial_indices_sensory_map.session_id==session_id, 'trial_idx'].values[0]
#     trial_idx_selections = [idx for _ in range(9)] + [idx_sensory_map]

#     # Generate a 3d array for each trial type.
#     stack = []
#     metadatas = []
#     for trial_selection, trial_idx_selection in zip(trial_selections, trial_idx_selections):
#         print(f'Processing {session_id} {trial_selection}')
#         traces, metadata = make_events_aligned_array_3d(nwb_file,
#                                                         rrs_keys,
#                                                         time_range,
#                                                         trial_selection,
#                                                         epoch_name,
#                                                         cell_types,
#                                                         trial_idx_selection)
#         stack.append(traces)
#         metadatas.append(metadata)

#     # Remove None given by empty trial types.
#     trial_type_labels = [trial_type_labels[i]
#                          for i in range(len(trial_type_labels))
#                          if stack[i] is not None]
#     metadatas = [metadata for metadata in metadatas if metadata is not None]
#     stack = [traces for traces in stack if traces is not None]
#     # Concatenate trial indices for each trial type.
#     # metadata_trials = [metadatas[i]['trials'] for i in range(len(metadatas))]
#     # metadata_trials = np.concatenate(metadata_trials, axis=0)
#     # Note that metadata is the same for all trial types
#     # (same cells) except for trial ids.
#     metadata_stacked = metadatas[0]
#     metadata_stacked['trials'] = {i: metadatas[i]
#                                   for i in range(len(trial_type_labels))}
#     metadata_stacked['trial_types'] = trial_type_labels

#     # Stack trial type to shape (n_cells, n_trial_type, n_trials, n_t).
#     max_trials = max([a.shape[1] for a in stack])
#     new_shape = (stack[0].shape[0],  len(stack), max_trials, stack[0].shape[2])
#     tensor = np.full(new_shape, np.nan)
#     for i, arr in enumerate(stack):
#         tensor[:, i, :arr.shape[1], :] = arr
#     # Reduce precision to save space.
#     tensor = tensor.astype(np.float16)

#     # Save dataset.
#     np.save(save_path_data, tensor)
#     with open(save_path_metadata, 'wb') as f:
#         pickle.dump(metadata_stacked, f)


# =============================================================================
# Create mice tensors with xarrays for session data (not mapping trials).
# =============================================================================

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
trial_indices_yaml = io.solve_common_paths('trial_indices')
stop_flag_yaml = io.solve_common_paths('stop_flags')
trial_indices_sensory_map_yaml = io.solve_common_paths('trial_indices_sensory_map')
stop_flag_sensory_map_yaml = io.solve_common_paths('stop_flags_sensory_map')
processed_data_dir = io.solve_common_paths('processed_data')

days = ['-3', '-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes')

with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# # For "non motivated" sensory mapping trials at the end of the session.
# with open(trial_indices_sensory_map_yaml, 'r') as stream:
#     trial_indices_sensory_map = yaml.load(stream, yaml.Loader)
# trial_indices_sensory_map = pd.DataFrame(trial_indices_sensory_map.items(), columns=['session_id', 'trial_idx'])

# mice_list = [mouse for mouse in mice_list if 'AR143'==mouse]
# nwb_list = [nwb for nwb in nwb_list if 'AR143' in nwb]

for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_learning_data.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]
    
    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []

    for nwb_file in session_nwb:
        
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)
        
        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (1,5)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]
        # idx_sensory_map = trial_indices_sensory_map.loc[trial_indices_sensory_map.session_id==session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(nwb_file,
                                                        rrs_keys,
                                                        time_range,
                                                        trial_selection,
                                                        epoch_name,
                                                        cell_types,
                                                        idx_selection)

        data.append(traces)
        metadatas.append(metadata)
    
    # Sessions are concatenated on the trial dim.
    tensor = np.concatenate(data, axis=1)

    # Load trial table and compute performance for those sessions.
    print('Make behavior table')
    behav_table = make_behavior_table(session_nwb, sessions, db_path,
                                      cut_session=True,
                                      stop_flag_yaml=stop_flag_yaml,
                                      trial_indices_yaml=trial_indices_yaml)
        
    time = np.linspace(-time_range[0], time_range[1], traces.shape[2])
    # Create xarray.
    ds = xr.DataArray(tensor, dims=['cell', 'trial', 'time'],
                        coords={'roi': ('cell', metadata['rois']),
                                'cell_type': ('cell', metadata['cell_types']),
                                'time': time,
                                })
    for col in behav_table.columns:
        ds[col] = ('trial', behav_table[col].values)
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)



# =============================================================================
# Create mice tensors with xarrays for mapping trials.
# =============================================================================

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
trial_indices_sensory_map_yaml = io.solve_common_paths('trial_indices_sensory_map')
stop_flag_sensory_map_yaml = io.solve_common_paths('stop_flags_sensory_map')
processed_data_dir = io.solve_common_paths('processed_data')

days = ['-3', '-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# For "non motivated" sensory mapping trials at the end of the session.
with open(trial_indices_sensory_map_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])

# mice_list = [mouse for mouse in mice_list if 'AR163'==mouse]
# nwb_list = [nwb for nwb in nwb_list if 'AR163' in nwb]

for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_mapping_data.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]
    
    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []
    behavior_days = []

    for nwb_file in session_nwb:
        
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)
        
        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (1,5)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id==session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(nwb_file,
                                                        rrs_keys,
                                                        time_range,
                                                        trial_selection,
                                                        epoch_name,
                                                        cell_types,
                                                        idx_selection)

        print(f'{np.isnan(traces).sum()} nan values in tensor.')
        
        data.append(traces)
        metadatas.append(metadata)
        # Get session day.
        # I don't load the trial table with the mapping trials, so get it for
        # nwb file.
        d = nwb_read.get_session_metadata(nwb_file)['day']
        behavior_days.extend([d for _ in range(traces.shape[1])])
        
    # Sessions are concatenated on the trial dim.
    tensor = np.concatenate(data, axis=1)
    
    time = np.linspace(-time_range[0], time_range[1], traces.shape[2])
    # Create xarray.
    ds = xr.DataArray(tensor, dims=['cell', 'trial', 'time'],
                        coords={'roi': ('cell', metadata['rois']),
                                'cell_type': ('cell', metadata['cell_types']),
                                'time': time,
                                'day': ('trial', behavior_days),
                                })
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)


# =============================================================================
# Test cell responsiveness and selectivity.
# =============================================================================

# Parameters.
response_win = (0, 0.300)
baseline_win = (-1, 0)

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')

# Get mice list.
days = ['-3', '-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

df = []
for mouse_id in mice_list:

    print(f'Testing responses of {mouse_id}')
    data_learning = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_learning_data.nc'))
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))
    sessions = data_learning.attrs['session_ids']
    # # Substract baseline.
    # data_learning = data_learning - np.nanmean(data_learning.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    # data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    days = list(np.sort(np.unique(data_learning.day)))

    # Test auditory responses for each day.
    response_test_aud = []
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'auditory_stim': 1, 'day': day}
        rois = data_learning.roi.values
        cell_types = data_learning.cell_type.values
        p = test_response(data_learning, trial_selection, response_win, baseline_win)
        response_test_aud.append(p)
        df.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                                'day': day, 'roi':rois, 'cell_type': cell_types,
                                'pval_aud': p}))
    # response_test_aud = np.vstack(response_test_aud)

    # Test whisker responses for each day.
    response_test_wh = []
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'whisker_stim': 1, 'day': day}
        p = test_response(data_learning, trial_selection, response_win, baseline_win)
        response_test_wh.append(p)
        df.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                            'day': day, 'roi':rois, 'cell_type': cell_types,
                            'pval_wh': p}))
    # response_test_wh = np.vstack(response_test_wh)

    # Test whisker responses for each day.
    response_test_mapping = []
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'day': day}
        p = test_response(data_mapping, trial_selection, response_win, baseline_win)
        response_test_mapping.append(p)
        df.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                            'day': day, 'roi':rois, 'cell_type': cell_types,
                            'pval_mapping': p}))
    # response_test_mapping = np.vstack(response_test_mapping)

# Save test results in dataframe.
df = pd.concat(df)
df = df.reset_index(drop=True)
df.to_csv(os.path.join(processed_data_folder, f'response_test_results_win_300ms.csv'))


# =============================================================================
# Compute LMI.
# =============================================================================

# This perfornms ROC analysis on each cell with mapping trials.
# Mapping trial of Day 0 are not included in the analysis.

# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-1, 0)
nshuffles = 1000

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')
result_file = os.path.join(processed_data_folder, 'lmi_results.csv')

# Get mice list.
days = ['-3', '-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# Load results if already computed.
if not os.path.exists(result_file):
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'lmi', 'lmi_p'])
else:
    df_results = pd.read_csv(result_file)
if not append_results:
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'lmi', 'lmi_p'])

df = []
for mouse_id in mice_list:
    if df_results.loc[df_results.mouse_id==mouse_id].shape[0] > 0:
        print(f'Mouse {mouse_id} already done. Skipping.')
        continue
    print(f'Processing {mouse_id}')
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))
    data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    data_pre = data_mapping.sel(trial=data_mapping.coords['day'].isin([-2, -1]))
    data_pre = data_pre.sel(time=slice(*response_win)).mean(dim='time')
    data_post = data_mapping.sel(trial=data_mapping.coords['day'].isin([1, 2]))
    data_post = data_post.sel(time=slice(*response_win)).mean(dim='time')
    lmi, lmi_p = compute_lmi(data_pre, data_post, nshuffles=nshuffles)
    df.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi': data_mapping.roi.values,
                            'cell_type': data_mapping.cell_type.values,
                            'lmi': lmi, 'lmi_p': lmi_p}))
if len(df)>0:
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    df_results = pd.concat([df_results, df])
    df_results.to_csv(result_file)
else:
    print('No new data to process.')

# data_mapping.shape
# np.isnan(data_mapping).sum()
# data_mapping[0].mean('time')







# dff = np.load(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR180\AR180_20241212_155749\suite2p\plane0\dff.npy")
# F_cor = np.load(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR180\AR180_20241212_155749\suite2p\plane0\F_cor.npy")
# F_raw = np.load(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR180\AR180_20241212_155749\suite2p\plane0\F_raw.npy")
# F = np.load(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR180\suite2p\plane0\F.npy")
# iscell = np.load(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR180\suite2p\plane0\iscell.npy")

# F = F[iscell[:,0]==1.]
# dff.shape
# dff[79]
# F_cor[79]
# plt.plot(F[78])

# np.where(iscell[:,0]==1)[0][79]

# np.isnan(dff[79]).sum()

# np.where(np.isnan(dff[:,0]))[0]

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

# import glob

# path = '/mnt/lsens-analysis/Anthony_Renard/data_processed/mice'

# for subdir in os.listdir(path):
#     subdir_path = os.path.join(path, subdir)
#     if os.path.isdir(subdir_path):
#         for file in os.listdir(subdir_path):
#             print(file)
#             if file.endswith('.npy'):
#                 old_file = os.path.join(subdir_path, file)
#                 new_file = old_file.replace('.npy', '.nc')
#                 os.rename(old_file, new_file)
#             if 'tensor_xarray_session_data.' in file:
#                 old_file = os.path.join(subdir_path, file)
#                 new_file = old_file.replace('tensor_xarray_session_data.', 'tensor_xarray_learning_data.')
#                 os.rename(old_file, new_file)
