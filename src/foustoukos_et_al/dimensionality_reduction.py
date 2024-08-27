import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import src.utils.utils_io as io


def load_session_2p_imaging(nwb_file, dir_path):
    mouse_id = nwb_file[-25:-20]
    session_id = nwb_file[-25:-4]
    array_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d.npy")
    tensor_metadata_path = os.path.join(dir_path, mouse_id, session_id, "tensor_4d_metadata.pickle")
    data = np.load(array_path)
    with open(tensor_metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return data, metadata


def extract_trials(arr, metadata, trial_type, n_trials):
    """Extract an arrya of shape (n_neurons, n_trials, n_timepoints)
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
        return np.empty((0, 0, 0))
    data = arr[:, metadata['trial_types'].index(trial_type)]
    # Keep only the first n_trials.
    data = data[:, :n_trials]
    # Remove nan values at the end of the trial dimension.
    data = data[:, ~np.isnan(data).all(axis=(0,2))]
    # Repeat last trial if less than required number of trials.
    if data.shape[1] < n_trials:
        data = np.concatenate([data, np.tile(data[:,[-1]], (1, n_trials - data.shape[1], 1))], axis=1)
    return data


def shape_features_matrix(mice, nwb_list, data_dir, trial_type, n_trials):
    """Return a 4d array of shape (n_neurons, n_trials, n_days, n_timepoints)
    for a given trial type and number of trials. This array can then be used
    to create X for dimensionality reduction and decoding.

    Args:
        mice (_type_): _description_
        nwb_list (_type_): _description_
        data_dir (_type_): _description_
        trial_type (_type_): _description_
        n_trials (_type_): _description_

    Returns:
        _type_: _description_
    """    
    mice_dataset = []
    for mouse_id in mice:
        # Get sessions for that mouse.
        # This assumes the order of the session days is preserved.
        nwbs = [file for file in nwb_list if mouse_id in file]
        # Load the datasets and metadata for each session.
        sessions = []
        sessions_metadata = []
        for nwb_file in nwbs:
            data, metadata = load_session_2p_imaging(nwb_file, data_dir)
            sessions.append(data)
            sessions_metadata.append(metadata)

        days = []
        for i, arr in enumerate(sessions):
            data = extract_trials(arr, sessions_metadata[i], trial_type, n_trials)
            days.append(data)
        days = np.stack(days, axis=1)
        mice_dataset.append(days)

    X = np.concatenate(mice_dataset, axis=0)

    return X


# =============================================================================
# Load the data.
# =============================================================================

# Path to the directory containing the processed data.
processed_dir = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed\mice"

# Rewarded and non-rewarded NWB files.
group_yaml_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_rewarded.yaml"
group_yaml_non_rew = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/mice_info/groups/imaging_non_rewarded.yaml"
nwb_list_rew = io.read_group_yaml(group_yaml_rew)
nwb_list_non_rew = io.read_group_yaml(group_yaml_non_rew)
nwb_list = nwb_list_rew + nwb_list_non_rew


# =============================================================================
# PCA on non-motivated whisker trials.
# =============================================================================

sampling_rate = 30
trial_type = 'WH'
n_trials = 50
n_days = 5
win = (1, 2)  # from stimulus onset to 300 ms after.
win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline = (0, 1)
baseline = (int(baseline[0] * sampling_rate), int(baseline[1] * sampling_rate))

# TODO: improve session section
# parametrise number of days.
mice = [file[-25:-20] for file in nwb_list_rew]
mice = list(set(mice))
mice = [m for m in mice if 'AR' in m]
mice = mice[1:2]

activity = shape_features_matrix(mice, nwb_list_rew, processed_dir, 'UM', 50)
# Subtract baselines.
activity = activity - np.nanmean(activity[:, :, :, baseline[0]:baseline[1]], axis=3, keepdims=True)

# Shape feature matrix.
# One average PSTH per day to fit the model and transform the full data.
# X = np.mean(activity[:, :, :, win[0]:win[1]], axis=3)
# X = np.mean(activity, axis=2)
X = np.mean(activity[:, :, :, win[0]:win[1]], axis=(2,3))
X = np.reshape(X, (X.shape[0], -1))

# Transpose to (n_samples, n_features).
X = X.T
# z-score the data.
X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
# Perform PCA.
pca = PCA(n_components=None)
model = pca.fit(X)


# Project response time series on the PCs.
# -----------------------------------------
activity.shape
reduced_act.shape
# Reshape activity to (n_neurons, n_trials x n_timepoints) and transform.
reduced_act = model.transform(activity.reshape(activity.shape[0], -1).T)
s = activity.shape
reduced_act = reduced_act.T.reshape(s[0], s[1], s[2], s[3])
# # Add extra dimension for consecutive days.
# reduced_act = reduced_act.reshape(s[0], n_days, n_trials, s[2])
# Make 
pc_psth = np.mean(reduced_act, axis=2)
# pc_psth = pc_psth - np.mean(pc_psth[:, :, baseline[0]:baseline[1]],
#                             axis=2, keepdims=True)

# Save PC PSTH to pdf.
pdf_path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\analysis_output\pca\pca_exploration.pdf"
with PdfPages(pdf_path) as pdf:
    for pc in range(30):
        f, axes = plt.subplots(1, 5, sharey=True, figsize=(15, 5))
        for i in range(n_days):
            axes[i].plot(pc_psth[pc, i, :])
            axes[i].axvline(30, color='orange')
        f.suptitle(f"PC {pc+1}")
        pdf.savefig(f)
        plt.close()


# Look at loadings and explained variance.
# ----------------------------------------

plt.figure()
plt.plot(model.explained_variance_ratio_.cumsum())

# Plot loading of first 10 PCs.
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i+1)
    plt.plot(model.components_[i])
    plt.title(f"PC {i+1}")
plt.tight_layout()
plt.show()








# Check that the psth of the non reduced data looks good.
X_global.shape
activity = X_global.reshape(X_global.shape[0], 5, 50, X_global.shape[2])
activity = activity - np.mean(activity[:, :, :, :30], axis=3, keepdims=True)
activity = np.mean(reduced_act, axis=(0,2))

activity.shape
f, axes = plt.subplots(1,5, sharey=True)
for i in range(5):
    axes[i].plot(activity[i,:])
    axes[i].axvline(30, color='orange')


mouse_id = 'AR135'
session_id = 'AR135_20240424_160805'
tensor_path = os.path.join(processed_dir, mouse_id, session_id, "tensor_4d.npy")
tensor = np.load(tensor_path)
tensor.shape

tensor = tensor - np.nanmean(tensor[:,:,:,:30], axis=3, keepdims=True)
plt.plot(np.nanmean(tensor[:,4,:,:30*4], axis=(0,1)))


# 6d tensor. Looks okay.
PROCESSED_DATA_PATH = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data_processed'
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_AR.npy')
traces_rew = np.load(read_path, allow_pickle=True)
read_path = os.path.join(PROCESSED_DATA_PATH, 'traces_non_motivated_trials_rew_AR_metadata.pickle')
with open(read_path, 'rb') as fid:
    metadata_rew = pickle.load(fid)

# Substract baseline.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

traces_rew.shape
metadata_rew['mice']

data = np.nanmean(traces_rew[2,], axis=(1,2,3))
data.shape
f, axes = plt.subplots(1,5, sharey=True)
for i in range(5):
    axes[i].plot(data[i,:])
    axes[i].axvline(30, color='orange')








# # ===========================================================================
# # Merge sessions of a single mouse.
# # ===========================================================================

# mouse_id = "AR143"
# # Get sessions for that mouse.
# sessions = [file for file in nwb_list if mouse_id in file]

# # Load the datasets and metadata for each session.
# stack = []
# stack_metadata = []
# for nwb_file in sessions:
#     session_id = nwb_file[-25:-4]
#     tensor_path = os.path.join(processed_dir, mouse_id, session_id, "tensor_4d.npy")
#     tensor_metadata_path = os.path.join(processed_dir, mouse_id, session_id, "tensor_4d_metadata.pickle")
#     dataset = np.load(tensor_path)
#     stack.append(dataset)
#     with open(tensor_metadata_path, 'rb') as f:
#         metadata = pickle.load(f)
#     stack_metadata.append(metadata)



# # Some sessions have different number of trial type (second dimension).
# # We need to pad the dataset to have the same shape.
# # Find the missing trial type and place nan values at the right location.
# # Trial types are ordered according to the following list.
# trial_type_labels = ['WH', 'WM', 'AH', 'AM', 'FA', 'CR', 'UM']
# for i, arr in enumerate(stack):
#     metadata = stack_metadata[i]
#     trial_types = metadata['trial_types']
#     missing_trial_type = list(set(trial_type_labels) - set(trial_types))
#     for trial_type in missing_trial_type:
#         missing_index = trial_type_labels.index(trial_type)
#         stack[i] = np.insert(arr, missing_index, np.nan, axis=1)
#         stack_metadata[i]['trial_types'] = trial_type_labels

# # Stack the datasets along the trial dimension.
# # Stack independently for each trial type to remove nan's padded at the end
# # of the trial dimension.

# tensor = []
# for itype, trial_type in enumerate(trial_type_labels):
#     arrays = [arr[:,i] for arr in stack]
#     arrays = np.concatenate(arrays, axis=1)
#     # Remove nan values at the end of the trial dimension
#     for i, arr in enumerate(arrays):
#         if np.isnan(arr).all():

#     tensor.append(arrays)

# tensor = np.concatenate(stack, axis=2)
# tensor_metadata = stack_metadata[0]
# tensor_metadata['trial_types'] = trial_type_labels
# tensor_metadata['trials'] = np.concatenate([metadata['trials'] for metadata in stack_metadata])

# stack_metadata[0]['trials']
# stack_metadata[0]['trial_types']

# # Combine the datasets if needed
# # combined_dataset = np.concatenate((dataset1, dataset2), axis=0)
