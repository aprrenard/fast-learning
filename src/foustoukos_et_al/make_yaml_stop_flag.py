"""This script generate yaml files associating a stop trial index ti each session to know when to cut the session.

"""
import os

import pandas as pd
import yaml

from nwb_wrappers import nwb_reader_functions as nwb_read


# Read NWB directory and list all NWB files.
nwb_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\NWB'
nwb_list = os.listdir(nwb_path)
nwb_list = [os.path.join(nwb_path,f) for f in nwb_list]

# Generate stop flags for the whisker sensory mapping block presented at the end of the session
# aka the non-motivated whisker trials.
# In that case the stop flag is a beginning flag corresponding to the first trial of this block.

stop_flag_non_motiv = {'stop_flag': {},
                     'trial_indices': {}
                     }

for nwb_file in nwb_list:
    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]
    table = nwb_read.get_trial_table(nwb_file)

    # End of motivated block i.e. trial just before start of the 50 non-motivated trials.
    non_mot_begin = (table.whisker_stim != table.whisker_stim.shift()).cumsum().idxmax()
    stop_flag_non_motiv['stop_flag'][session_id] = non_mot_begin
    # Exceptions. Forgot to remove the auditory trials.
    if session_id == 'AR132_20240427_122605':
        stop_flag_non_motiv['stop_flag'][session_id] = 571

    # Get the indices of those trials.
    non_mot_idx = table.loc[(table.trial_id>=non_mot_begin), 'trial_id'].to_list()
    # Select only the 50 last non-motivated trials (in case more than 50 were presented).
    non_mot_idx = non_mot_idx[-50:]
    stop_flag_non_motiv['trial_indices'][session_id] = non_mot_idx

# Save yaml file.
yaml_save = r'C:\Users\aprenard\recherches\repos\fast-learning\docs\stop_flags\stop_flags_non_motiv.yaml'
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(stop_flag_non_motiv, stream)




# Check selected trials with various stop criteria.

temp = []
trial_idx_table_GF = []
trial_idx_table_AR = []

for nwb_file in nwb_list:

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