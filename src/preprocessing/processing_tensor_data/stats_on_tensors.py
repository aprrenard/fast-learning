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
from src.utils.utils_behavior import *
from src.utils.utils_imaging import compute_roc
from joblib import Parallel, delayed


def test_response(data, trial_selection, response_win, baseline_win, method='mannwhitney'):
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
        if method == 'mannwhitney':
            # Use Mann-Whitney U test.
            _, p = stats.mannwhitneyu(response[cell], baseline[cell], alternative='greater')
            pval[cell] = p
            if np.isnan(p):
                print(f'Cell {cell} has NaN values.')  
        elif method == 'wilcoxon':
            _, p = stats.wilcoxon(response[cell], baseline[cell])
            pval[cell] = p
            if np.isnan(p):
                print(f'Cell {cell} has NaN values.')
        else:
            raise ValueError(f'Unknown method: {method}')

    return pval


# =============================================================================
# Test cell responsiveness and selectivity.
# =============================================================================


# Test responsiveness each day independantly.
# ===========================================

# Parameters.
response_win = (0, 0.180)
response_length = 180  # for file name.
baseline_win = (-1, 0)
days = ['-3', '-2', '-1', '0', '+1', '+2']

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')

# Get mice list.
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

df_aud = []
df_wh = []
df_map = []
for mouse_id in mice_list:

    print(f'Testing responses of {mouse_id}')
    data_learning = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_learning_data.nc'))
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))
    sessions = data_learning.attrs['session_ids']
    # # Substract baseline -- no need to do that for a Wilcoxon test.
    # data_learning = data_learning - np.nanmean(data_learning.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    # data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    days = list(np.sort(np.unique(data_learning.day)))

    # Test auditory responses for each day.
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'auditory_stim': 1, 'day': day}
        rois = data_learning.roi.values
        cell_types = data_learning.cell_type.values
        pval_aud = test_response(data_learning, trial_selection, response_win, baseline_win)
        df_aud.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                                'day': day, 'roi':rois, 'cell_type': cell_types,
                                'pval_aud': pval_aud}))

    # Test whisker responses for each day.
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'whisker_stim': 1, 'day': day}
        pval_wh = test_response(data_learning, trial_selection, response_win, baseline_win)
        df_wh.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                            'day': day, 'roi':rois, 'cell_type': cell_types,
                            'pval_wh': pval_wh}))

    # Test whisker responses during baseline mapping trials.
    for day in days:
        session_id = sessions[days.index(day)]
        trial_selection = {'day': day}
        pval_map = test_response(data_mapping, trial_selection, response_win, baseline_win)
        df_map.append(pd.DataFrame({'mouse_id': mouse_id, 'session_id':session_id,
                            'day': day, 'roi':rois, 'cell_type': cell_types,
                            'pval_mapping': pval_map}))

# Save test results in dataframe.
df_aud = pd.concat(df_aud)
df_wh = pd.concat(df_wh)
df_map = pd.concat(df_map)
df = pd.merge(df_aud, df_wh, on=['mouse_id', 'session_id', 'day', 'roi', 'cell_type'])
df = pd.merge(df, df_map, on=['mouse_id', 'session_id', 'day', 'roi', 'cell_type'])
df['pval_aud'] = df['pval_aud'].astype(float)
df['pval_wh'] = df['pval_wh'].astype(float)
df['pval_mapping'] = df['pval_mapping'].astype(float)

df = df.reset_index(drop=True)
df.to_csv(os.path.join(processed_data_folder, f'response_test_results_win_{response_length}ms.csv'))


# Test responsiveness pulling all days together.
# ==============================================

# Parameters.
response_win = (0, 0.180)
response_length = 180  # for file name.
baseline_win = (-1, 0)
days = ['-2', '-1', '0', '+1', '+2']

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')

# Get mice list.
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

df_aud = []
df_wh = []
df_map = []
for mouse_id in mice_list:

    print(f'Testing responses of {mouse_id}')
    data_learning = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_learning_data.nc'))
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))
    sessions = data_learning.attrs['session_ids']
    # # Substract baseline -- no need to do that for a Wilcoxon test.
    # data_learning = data_learning - np.nanmean(data_learning.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    # data_mapping = data_mapping - np.nanmean(data_mapping.sel(time=slice(*baseline_win)), axis=2, keepdims=True)
    
    days = list(np.sort(np.unique(data_learning.day)))

    # Test auditory responses for each day.
    trial_selection = {'auditory_stim': 1}
    rois = data_learning.roi.values
    cell_types = data_learning.cell_type.values
    pval_aud = test_response(data_learning, trial_selection, response_win, baseline_win)
    df_aud.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi':rois, 'cell_type': cell_types,
                            'pval_aud': pval_aud}))

    # Test whisker responses for each day.
    trial_selection = {'whisker_stim': 1}
    pval_wh = test_response(data_learning, trial_selection, response_win, baseline_win)
    df_wh.append(pd.DataFrame({'mouse_id': mouse_id,
                        'roi':rois, 'cell_type': cell_types,
                        'pval_wh': pval_wh}))

    # Test whisker responses during baseline mapping trials.
    trial_selection = {}
    pval_map = test_response(data_mapping, trial_selection, response_win, baseline_win)
    df_map.append(pd.DataFrame({'mouse_id': mouse_id,
                        'roi':rois, 'cell_type': cell_types,
                        'pval_mapping': pval_map}))

# Save test results in dataframe.
df_aud = pd.concat(df_aud)
df_wh = pd.concat(df_wh)
df_map = pd.concat(df_map)
df = pd.merge(df_aud, df_wh, on=['mouse_id', 'roi', 'cell_type'])
df = pd.merge(df, df_map, on=['mouse_id', 'roi', 'cell_type'])
df['pval_aud'] = df['pval_aud'].astype(float)
df['pval_wh'] = df['pval_wh'].astype(float)
df['pval_mapping'] = df['pval_mapping'].astype(float)

df = df.reset_index(drop=True)
df.to_csv(os.path.join(processed_data_folder, f'response_test_results_alldaystogether_win_{response_length}ms.csv'))


# Test responsiveness of mapping trials using ROC analysis.
# =========================================================

# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-1, 0)
nshuffles = 1000

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')
result_file = os.path.join(processed_data_folder, 'response_test_results_mapping_ROC.csv')

# Get mice list.
days = ['-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# Load results if already computed.
if not os.path.exists(result_file):
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'roc', 'roc_p'])
else:
    df_results = pd.read_csv(result_file)
if not append_results:
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'roc', 'roc_p'])

df = []

for mouse_id in mice_list:
    if df_results.loc[df_results.mouse_id==mouse_id].shape[0] > 0:
        print(f'Mouse {mouse_id} already done. Skipping.')
        continue
    print(f'Processing {mouse_id}')
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))

    # Select time window.
    response = data_mapping.sel(time=slice(*response_win)).mean(dim='time')    
    baseline = data_mapping.sel(time=slice(*baseline_win)).mean(dim='time')
    
    roc, roc_p = compute_roc(baseline, response, nshuffles=nshuffles)
    df.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi': data_mapping.roi.values,
                            'cell_type': data_mapping.cell_type.values,
                            'roc': roc, 'roc_p': roc_p}))
if len(df)>0:
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    df_results = pd.concat([df_results, df])
    df_results.to_csv(result_file)
else:
    print('No new data to process.')
    

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




# =============================================================================
# Compute day 0 LMI.
# =============================================================================

# This perfornms ROC analysis on each cell with n first and m last trials
# of day 0 during learning.


# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-1, 0)
nshuffles = 100
n_first = 5
n_last = 20

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')
result_file = os.path.join(processed_data_folder, 'lmi_day0_results.csv')

# Get mice list.
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
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
    data = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_learning_data.nc'))
    data = data - np.nanmean(data.sel(time=slice(*baseline_win)), axis=2, keepdims=True) 
    
    # Select days.
    data = data.sel(trial=data['day'].isin([0]))    
    # Select whisker trials.
    data = data.sel(trial=data.coords['whisker_stim']==1)

    data_pre = data.sel(trial=data['trial_w']<=n_first)
    data_pre = data_pre.sel(time=slice(*response_win)).mean(dim='time')
    data_post = data.sel(trial=data['trial_w']>=data['trial_w'].max()-n_last)
    data_post = data_post.sel(time=slice(*response_win)).mean(dim='time')

    lmi, lmi_p = compute_roc(data_pre, data_post, nshuffles=nshuffles)
    df.append(pd.DataFrame({'mouse_id': mouse_id,
                            'roi': data.roi.values,
                            'cell_type': data.cell_type.values,
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



# =============================================================================
# Baseline VS stim for mapping trials.
# =============================================================================

# This perfornms ROC analysis on each cell with mapping trials, testing
# the difference between baseline and stimulation periods.
# The rational is to assess whether cells with high LMI values would
# develop a response after learning or whether they already had a response
# a strong response before learning.


# Parameters.
append_results = False
response_win = (0, 0.300)
baseline_win = (-300, 0)
nshuffles = 100

# Get directories and files.
db_path = io.solve_common_paths('db')
nwb_path = io.solve_common_paths('nwb')
processed_data_folder = io.solve_common_paths('processed_data')
result_file = os.path.join(processed_data_folder, 'roc_stimvsbaseline_results.csv')

# Get mice list.
days = ['-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = io.select_sessions_from_db(db_path, nwb_path,
                                                exclude_cols=['exclude', 'two_p_exclude'],
                                                experimenters=['AR', 'GF', 'MI'],
                                                day=days,
                                                two_p_imaging='yes',)

# Load results if already computed.
if not os.path.exists(result_file):
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'day', 'roc', 'roc_p'])
else:
    df_results = pd.read_csv(result_file)
if not append_results:
    df_results = pd.DataFrame(columns=['mouse_id', 'roi', 'cell_type', 'day', 'roc', 'roc_p'])

df = []
for mouse_id in mice_list:
    if df_results.loc[df_results.mouse_id==mouse_id].shape[0] > 0:
        print(f'Mouse {mouse_id} already done. Skipping.')
        continue
    print(f'Processing {mouse_id}')
    data_mapping = xr.open_dataarray(os.path.join(processed_data_folder, 'mice', mouse_id, 'tensor_xarray_mapping_data.nc'))
    
    # ROC analysis of baseline vs stim for each day
    for day in [-2, -1, 0, 1, 2]:
        data_day = data_mapping.sel(trial=data_mapping.coords['day'] == day)
        response = data_day.sel(time=slice(*response_win)).mean(dim='time')
        baseline = data_day.sel(time=slice(*baseline_win)).mean(dim='time')
        roc, roc_p = compute_roc(baseline, response, nshuffles=nshuffles, n_jobs=20)
        df.append(pd.DataFrame({
            'mouse_id': data_mapping.attrs.get('mouse_id', mouse_id),
            'roi': data_mapping.roi.values,
            'cell_type': data_mapping.cell_type.values,
            'day': day,
            'roc': roc,
            'roc_p': roc_p
        }))
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