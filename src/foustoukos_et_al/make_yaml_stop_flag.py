"""This script generates yaml files associating a (start, stop) tuple trial
index for each session to know when to cut the session.
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import yaml

from nwb_wrappers import nwb_reader_functions as nwb_read
from src.utils.utils_io import read_excel_db


# =============================================================================
# Path configuation.
# =============================================================================

nwb_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\NWB'
excel_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\mice_info\session_metadata.xlsx'


# =============================================================================
# Generate stop flags for the whisker sensory mapping block.
# =============================================================================
# Imaging sessions have a block of 50 whisker trials at the end of the session.
# Find the trial index of the first whisker trial of this block.

# List nwb files.
nwb_list = read_excel_db(excel_path)
nwb_list = nwb_list.loc[(nwb_list['exclude']!='exclude')]
nwb_list = nwb_list.loc[(nwb_list['two_p_imaging'] == 'yes')
                        & (nwb_list['sensory_mapping'] == 'yes')]
nwb_list = list(nwb_list.session_id)
nwb_list = [os.path.join(nwb_path, f + '.nwb') for f in nwb_list]

stop_flags = {}
trial_indices = {}
trial_count = []

for nwb_file in nwb_list:

    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]
    table = nwb_read.get_trial_table(nwb_file)
    print(f"Processing {session_id} ({nwb_list.index(nwb_file) + 1}/{len(nwb_list)})")

    # Beginning of the whisker sensory mapping block.
    # Find final segment of one's in the whisker_stim column.
    start = table.whisker_stim != table.whisker_stim.shift()
    plt.plot(start.cumsum())
    start = int(start.cumsum().idxmax())
    # If more than 50 trial of whisker stimulation, keep only the last 50.
    if table.loc[start:].shape[0] > 50:
        start = start + table.loc[start:].shape[0] - 50
    # Exceptions.
    if session_id == 'AR132_20240427_122605':
        start = 578
    if session_id == 'AR139_20240428_180459':
        start = 414
    if session_id == 'GF350_28052021_145113':
        start = 224
    stop_flags[session_id] = (start, int(table.index.max()))

    # Get the indices of those trials.
    trial_ids = table.loc[start:, 'trial_id'].to_list()
    trial_indices[session_id] = trial_ids

    # Count trial for sanity check.
    n_wh_miss = table.loc[(table.trial_id >= start)
                          & (table.whisker_stim == 1)
                          & (table.lick_flag == 0)].shape[0]
    n_wh_hit = table.loc[(table.trial_id >= start)
                         & (table.whisker_stim == 1)
                         & (table.lick_flag == 1)].shape[0]
    trial_count.append([mouse_id, session_id, start, n_wh_miss, n_wh_hit])
# 
# Save yaml files.
yaml_save = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\stop_flags\stop_flags_sensory_map.yaml'
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(stop_flags, stream)
yaml_save = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\stop_flags\trial_indices_sensory_map.yaml'
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(trial_indices, stream)

# # Sanity check.
# trial_count = pd.DataFrame(trial_count, columns = ['mouse_id', 'session_id', 'start', 'n_wh_miss', 'n_wh_hit'])

# nwb_file = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\NWB\\AR115_20231116_142507.nwb'
# table = nwb_read.get_trial_table(nwb_file)
# table = table.reset_index()
# plt.figure()
# plt.scatter(table.loc[table.auditory_stim==1, 'trial_id'], table.loc[table.auditory_stim==1, 'lick_flag'])
# plt.scatter(table.loc[table.whisker_stim==1, 'trial_id'], table.loc[table.whisker_stim==1, 'lick_flag'])
# plt.scatter(table.loc[table.no_stim==1, 'trial_id'], table.loc[table.no_stim==1, 'lick_flag'])


# =============================================================================
# Generate session stop flags.
# =============================================================================
# The criteria to cut the session is the mouse is not performing the auditory
# defined as three auditory misses in a row and no more than two auditory
# hits in the rest.

# List nwb files.
nwb_list = read_excel_db(excel_path)
nwb_list = nwb_list.loc[(nwb_list['exclude']!='exclude')]
nwb_list = list(nwb_list.session_id)
nwb_list = [os.path.join(nwb_path, f + '.nwb') for f in nwb_list]

stop_flags = {}
trial_indices = {}

for nwb_file in nwb_list[:10]:

    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]
    table = nwb_read.get_trial_table(nwb_file)
    print(f"\rProcessing {session_id} ({nwb_list.index(nwb_file) + 1}/{len(nwb_list)})", end="")

    # If a session does not contain auditory trials, keep whole session.
    # This happens for some test sessions of the GF mice.
    if table.auditory_stim.sum() == 0:
        stop = table.trial_id.idxmax()
        stop_flags[session_id] = (0, int(stop))
        trial_indices[session_id] = table['trial_id'].to_list()
        continue

    # Get index after which no more than two hits until session end.
    two_hits_left = table.loc[table.auditory_stim==1, 'lick_flag']
    two_hits_left = (two_hits_left.cumsum() >= (two_hits_left.sum() - 2)).idxmax()
    # Find three auditory misses in a row.
    three_aud_misses = table.loc[table.auditory_stim==1, 'lick_flag']
    three_aud_misses = three_aud_misses.rolling(window=3).sum() == 0
    # Stop flag is the first occurence of three auditory misses in a row
    # followed by no more than three hits in the rest of the session.
    # In case no three auditory misses keep the whole session.
    if three_aud_misses.sum() == 0:
        stop = table.loc[table.auditory_stim==1, 'trial_id'].idxmax()
    else:
        stop = three_aud_misses.loc[three_aud_misses.index>two_hits_left].idxmax()
        # Stop three trials before the three auditory misses.
        position = three_aud_misses.index.get_loc(stop)
        stop = three_aud_misses.index[position - 3]
        
    # Exceptions.
    if session_id == 'AR115_20231116_142507':
        stop = 35
    stop_flags[session_id] = (0, int(stop))

    # Get the indices of those trials.
    trial_ids = table.loc[(table.index<=stop), 'trial_id'].to_list()
    trial_indices[session_id] = trial_ids

# Save yaml files.
yaml_save = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\stop_flags\stop_flags_end_session.yaml'
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(stop_flags, stream)
yaml_save = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\stop_flags\trial_indices_end_session.yaml'
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(trial_indices, stream)