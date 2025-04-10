import os
import sys

import yaml
import pandas as pd

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from nwb_wrappers import nwb_reader_functions as nwb_read


# Path to the directory containing the processed data.
processed_dir = io.solve_common_paths('processed_data')
nwb_dir = io.solve_common_paths('nwb')
db_path = io.solve_common_paths('db')


# #############################################################################
# Checking the yaml config files reward group match with the db.
# #############################################################################

sessions, nwb_files, mice, db = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            reward_group='R+'
                                            )

# Read groups from db.
db_reward_groups = db[['subject_id', 'session_id', 'reward_group']].drop_duplicates()

# Read groups from nwb files.
yaml_reward_groups = []

for session, nwb_file in zip(sessions, nwb_files):
    metadata = nwb_read.get_session_metadata(nwb_file)
    g = 'R+' if metadata['wh_reward'] == 1 else 'R-'
    yaml_reward_groups.append([session[:5], session, g])

yaml_reward_groups = pd.DataFrame(yaml_reward_groups, columns=['subject_id', 'session_id', 'reward_group'])

df = pd.merge(db_reward_groups, yaml_reward_groups, on=['subject_id', 'session_id'], suffixes=('_db', '_yaml'))

df.loc[df.reward_group_db != df.reward_group_yaml, :]