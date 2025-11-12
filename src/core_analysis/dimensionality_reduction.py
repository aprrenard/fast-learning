import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import src.utils.utils_io as io
import src.utils.utils_imaging as imaging_utils 


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
win = (1, 1.300)  # from stimulus onset to 300 ms after.
win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline = (0, 1)
baseline = (int(baseline[0] * sampling_rate), int(baseline[1] * sampling_rate))

# TODO: improve session section
# parametrise number of days.
mice = [file[-25:-20] for file in nwb_list_rew]
mice = list(set(mice))
mice = [m for m in mice if 'AR' in m]
mice = mice[:1]

activity = utils_imaging.shape_features_matrix(mice, nwb_list_rew, processed_dir, trial_type, 50)
# Subtract baselines.
activity = activity - np.nanmean(activity[:, :, baseline[0]:baseline[1]],
                                 axis=2, keepdims=True)

# Shape feature matrix.
# Fit PCA by either:
# - for each neuron, keep all trial and compute the mean response over time.
# - for each neuron, average trials and keep the mean response over time
# (keep time dimension).

# X = np.mean(activity[:, :, win[0]:win[1]], axis=2)
X = np.mean(activity, axis=1)
X = np.reshape(X, (X.shape[0], -1))

# Transpose to (n_samples, n_features).
X = X.T
# z-score the data.
X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
# Perform PCA.
pca = PCA(n_components=None)
model = pca.fit(X)


# Project response time series on the PCs.
# ----------------------------------------

# Apply the model.
# Reshape activity to (n_neurons, n_trials x n_timepoints) and transform.
reduced_act = model.transform(activity.reshape(activity.shape[0], -1).T)
reduced_act = reduced_act.T
n_pc = model.n_components_
s = activity.shape
# First dim length is min(n_features, n_samples).
# Add session dimension.
reduced_act = reduced_act.reshape(n_pc, n_days, n_trials, s[2])

# Create PC PSTH's. 
pc_psth = reduced_act - np.mean(reduced_act[:, :, :, baseline[0]:baseline[1]],
                                axis=3, keepdims=True)
pc_psth = np.mean(reduced_act, axis=2)

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


# =============================================================================
# Projection on learning dimension.
# =============================================================================

sampling_rate = 30
n_trials = 50
n_days = 5
win = (1, 2)  # from stimulus onset to 300 ms after.
win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline = (0, 1)
baseline = (int(baseline[0] * sampling_rate), int(baseline[1] * sampling_rate))

mice = [file[-25:-20] for file in nwb_list_rew]
mice = list(set(mice))
mice = [m for m in mice if 'AR' in m]
mice = mice[:1]

activity_UM = shape_features_matrix(mice, nwb_list_rew, processed_dir, 'UM', 50)
# Subtract baselines.
activity_UM = activity_UM - np.nanmean(activity_UM[:, :, :, baseline[0]:baseline[1]],
                                 axis=3, keepdims=True)

activity_WH = shape_features_matrix(mice, nwb_list_rew, processed_dir, 'WH', 30)
# Subtract baselines.
activity_WH = activity_WH - np.nanmean(activity_WH[:, :, :, baseline[0]:baseline[1]],
                                 axis=3, keepdims=True)




# Shape feature matrix.
# Fit PCA by either:
# - for each neuron, keep all trial and compute the mean response over time.
# - for each neuron, average trials and keep the mean response over time
# (keep time dimension).

# X = np.mean(activity[:, :, :, win[0]:win[1]], axis=3)
X = np.mean(activity, axis=2)
X = np.reshape(X, (X.shape[0], -1))

# Transpose to (n_samples, n_features).
X = X.T
# z-score the data.
X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
# Perform PCA.
pca = PCA(n_components=None)
model = pca.fit(X)






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
