from datetime import datetime

import pandas as pd
import yaml


def read_excel_db(db_path):
    database = pd.read_excel(db_path, converters={'day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    return database


def read_group_yaml(group_yaml_path):
    with open(group_yaml_path, 'r') as file:
        group_yaml = yaml.safe_load(file)

    return group_yaml


def read_stop_flags_and_indices_yaml(stop_flag_yaml_path, trial_indices_path):
    with open(stop_flag_yaml_path, 'r') as file:
        stop_flags = yaml.safe_load(file)
    with open(trial_indices_path, 'r') as file:
        trial_indices = yaml.safe_load(file)
    trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
    
    return stop_flags, trial_indices