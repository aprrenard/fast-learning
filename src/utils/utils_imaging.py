import os
import pickle

import numpy as np
import warnings
from sklearn.metrics import auc, roc_curve
from sklearn.utils import shuffle
from scipy.stats import percentileofscore
import xarray as xr



def load_session_2p_imaging(mouse_id, session_id, dir_path):
    array_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d.npy")
    tensor_metadata_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d_metadata.pickle")
    data = np.load(array_path)
    with open(tensor_metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return np.float32(data), metadata


def load_mouse_xarray(mouse_id, dir_path, file_name):
    
    array_path = os.path.join(dir_path, mouse_id, file_name)
    with xr.open_dataarray(array_path) as ds:
        data = ds.copy()
    return data


def substract_baseline(arr, time_axis, baseline_win):
    """Substract the mean of the baseline window from the array.

    Args:
        arr (numpy.array): Array of shape (n_neurons, n_trials, n_timepoints).
        time_axis (int): Axis of the timepoints.
        baseline_window (tuple): Tuple of two integers indicating the start and
        end of the baseline window.

    Returns:
        numpy.array: Array of shape (n_neurons, n_trials, n_timepoints).
    """
    ndims = arr.ndim
    slices = [slice(None),] * ndims
    slices[time_axis] = slice(baseline_win[0], baseline_win[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline = np.nanmean(arr[*slices], axis=time_axis, keepdims=True)
    arr = arr - baseline
    return arr


def pad_arrays(arrays, side='end', dims=None, pad_value=np.nan):
    ndims = arrays[0].ndim
    if dims is None:
        dims = range(ndims)
    for idim in dims:
        max_shape = max(arr.shape[idim] for arr in arrays)
        for i, arr in enumerate(arrays):
            if arr.shape[idim] < max_shape:
                if side == 'beginning':
                    pad_width = [(0, 0)] * ndims
                    pad_width[idim] = (max_shape - arr.shape[idim], 0)
                elif side == 'end':
                    pad_width = [(0, 0)] * ndims
                    pad_width[idim] = (0, max_shape - arr.shape[idim])
                arrays[i] = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    return arrays


def stack_sessions(arrays, axis=1):
    # Pad trial type dim for those sessions without WH and WM trials.
    # These two trials types are the first two.
    arrays = pad_arrays(arrays, side='beginning', dims=[1], pad_value=np.nan)
    # Pad the number of trials for each session.
    arrays = pad_arrays(arrays, side='end', dims=[2], pad_value=np.nan)
    return np.stack(arrays, axis=axis)


def extract_trials(arr, metadata, trial_type, n_trials=None, repeat_last_trial=False):
    """Extract an array of shape (n_neurons, n_trials, n_timepoints)
    for a given trial type.
    As nan values are not desirable to run dimensionailty reduction or
    decoding, to solve inconsistencies in the number of trials between
    sessions, we repeat the last trial until we have n_trials trials.

    Args:
        arr (numpy.array): Single session imaging data of shape
        (n_neurons, n_trial_type, n_trials, n_timepoints).
        metadata (dict): Associated metadata.
        trial_type (str): Trial type label to extract.
        n_trials (int): Number of trials to extract. If less than n_trials,
        the last trial is repeated until we have n_trials.

    Returns:
        numpy.array: Array of shape (n_neurons, n_trials, n_timepoints).
    """

    if trial_type not in metadata['trial_types']:
        print(f"Trial type '{trial_type}' is not present in the metadata.")
        return None
    data = arr[:, metadata['trial_types'].index(trial_type)]
    
    if n_trials is not None:
        # Keep only the first n_trials.
        data = data[:, :n_trials]
        print(f'here {data.shape}')
    # Remove nan values at the end of the trial dimension.
    data = data[:, ~np.isnan(data).all(axis=(0,2))]
    if n_trials is not None and repeat_last_trial:
        # Repeat last trial if less than required number of trials.
        if data.shape[1] < n_trials:
            data = np.concatenate([data, np.tile(data[:,[-1]], (1, n_trials - data.shape[1], 1))], axis=1)
    return data


def shape_features_matrix(mouse_list, session_list, data_dir, trial_type, n_trials=None):
    """Return a 4d array of shape (n_neurons, n_trials, n_timepoints)
    for a given trial type and number of trials. This array can then be used
    to create X for dimensionality reduction and decoding. For each mouse,
    different days with the same cells are concatenated along the trial
    dimension.

    Args:
        mice (_type_): _description_
        session_list (_type_): _description_
        data_dir (_type_): _description_
        trial_type (_type_): _description_
        n_trials (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    mice_dataset = []
    for mouse_id in mouse_list:
        # Get sessions for that mouse.
        sessions = [session_id for session_id in session_list if mouse_id in session_id]
        # Load the datasets and metadata for each session.
        session_data = []
        sessions_metadata = []
        for session_id in sessions:
            data, metadata = load_session_2p_imaging(mouse_id, session_id, data_dir)
            session_data.append(data)
            sessions_metadata.append(metadata)

        days = []
        for i, arr in enumerate(session_data):
            data = extract_trials(arr, sessions_metadata[i], trial_type, n_trials)
            if data is not None:
                days.append(data)
        days = np.concatenate(days, axis=1)
        mice_dataset.append(days)

    print('dataset shapes')
    for arr in mice_dataset:
        print(arr.shape)
    
    X = np.concatenate(mice_dataset, axis=0)

    return X


def compute_roc(data_pre, data_post, nshuffles=1000):
    '''
    Compute ROC analysis and Learning modulation index for each cell in data.
    data_pre: np array of shape (cell, trial).
    data_post: np array of shape (cell, trial).

    LMI are computed on days D-2, D-1 together VERSUS D+1, D+2 together.
    '''
    
    ncell = data_pre.shape[0]

    lmi = np.full(ncell, np.nan)
    lmi_p = np.full(ncell, np.nan)

    for icell in range(ncell):
        print(f'ROC computation: {icell+1}/{ncell} cells', end='\r')
        
        X_pre = data_pre[icell]
        X_post = data_post[icell]
        X = np.r_[X_pre, X_post]
        y = np.r_[[0 for _ in range(X_pre.shape[0])], [1 for _ in range(X_post.shape[0])]]
        
        fpr, tpr, _ = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)
        lmi[icell] = (roc_auc - 0.5) * 2
        
        # Test significativity of LMI values with shuffles.
        if nshuffles:
            roc_auc_shuffle = np.zeros(nshuffles)
            for ishuffle in range(nshuffles):
                y_shuffle = shuffle(y, random_state=ishuffle)
                fpr, tpr, _ = roc_curve(y_shuffle, X)
                roc_auc_shuffle[ishuffle] = auc(fpr, tpr)
            lmi_p[icell] = percentileofscore(roc_auc_shuffle, roc_auc) / 100
        else:
            lmi_p[icell] = np.nan
    print('')
    return lmi, lmi_p