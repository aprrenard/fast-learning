import os
import pickle

import numpy as np
import warnings


def load_session_2p_imaging(mouse_id, session_id, dir_path):
    array_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d.npy")
    tensor_metadata_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d_metadata.pickle")
    data = np.load(array_path)
    with open(tensor_metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return np.float32(data), metadata


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

# TODO pad beginning or end
# give list of axis to pad 

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


def extract_trials(arr, metadata, trial_type, n_trials=None):
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
    # Remove nan values at the end of the trial dimension.
    data = data[:, ~np.isnan(data).all(axis=(0,2))]
    if n_trials is not None:
        # Repeat last trial if less than required number of trials.
        if data.shape[1] < n_trials:
            data = np.concatenate([data, np.tile(data[:,[-1]], (1, n_trials - data.shape[1], 1))], axis=1)
    return data


def shape_features_matrix(mouse_list, session_list, data_dir, trial_type, n_trials):
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
