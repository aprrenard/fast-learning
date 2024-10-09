import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import warnings


def read_excel_db(db_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        database = pd.read_excel(db_path, converters={'day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    return database


def select_sessions_from_db(db_path, nwb_path, experimenters=None,
                            exclude_cols=['exclude', 'two_p_exclude'],
                            **filters):
    """Select sessions from excel database on filters.
    Return a list of session ids or nwb file paths.

    Args:
        db_path (_type_): _description_
        experimenters (_type_): _description_
        exclude_cols (_type_): _description_
        nwb_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    db = read_excel_db(db_path)
    for key, val in filters.items():
        if isinstance(val, list):
            db = db.loc[(db[key].isin(val))]
        else:
            db = db.loc[(db[key]==val)]
    # Remove excluded sessions.
    for col in exclude_cols:
        db = db.loc[(db[col]!='exclude')]
        
    mice_list = list(db.subject_id.unique())
    session_list = list(db.session_id)
    if experimenters:
        session_list = [session for session in session_list
                        if session[:2] in experimenters]
        mice_list = [mouse for mouse in mice_list
                     if mouse[:2] in experimenters]
    nwb_paths = [os.path.join(nwb_path, f + '.nwb') for f in session_list]
    
    return session_list, nwb_paths, mice_list, db


def get_reward_group_from_db(db_path, session_id):
    db = read_excel_db(db_path)
    reward_group = db.loc[db['session_id']==session_id, 'reward_group'].values[0]

    return reward_group


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

