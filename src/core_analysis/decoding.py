import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.stats import bootstrap

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
            rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})


# ##################################################
# Decoding before and after learning.
# ##################################################

# Similar to the previous section, but compute a correlation matrix for
# each mouse and then average across mice.

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data.
vectors_rew = []
vectors_nonrew = []
mice_rew = []
mice_nonrew = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]


for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins.
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # # Equalize the mean activity of each cell across days by shifting (additive constant).
    # # For each cell, compute its mean activity on each day.
    # # Use the mean across all days as the reference mean.
    # # For each day, add a constant so that its mean matches the reference mean.
    # cell_axis = 0  # axis for cells
    # trial_axis = 1  # axis for trials
    # if 'cell' in d.dims:
    #     cell_dim = 'cell'
    # else:
    #     cell_dim = 'roi'
    # trial_dim = 'trial'
    # day_per_trial = d['day'].values
    # unique_days = np.unique(day_per_trial)
    # arr = d.values  # shape: (n_cells, n_trials)
    # arr_eq = arr.copy()
    # for icell in range(arr.shape[cell_axis]):
    #     # For each cell, get trial indices for each day
    #     cell_vals = arr[icell, :]
    #     # Compute mean across all days (reference)
    #     ref_mean = np.nanmean(cell_vals)
    #     for day in unique_days:
    #         day_mask = (day_per_trial == day)
    #         if np.any(day_mask):
    #             day_mean = np.nanmean(cell_vals[day_mask])
    #             shift = ref_mean - day_mean
    #             arr_eq[icell, day_mask] = cell_vals[day_mask] + shift
    # # Replace d.values with the shifted array
    # d.values[:] = arr_eq

    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)
    
    if rew_gp == 'R-':
        vectors_nonrew.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew.append(d)
        mice_rew.append(mouse)


# Decoding accuracy between reward groups.
# ----------------------------------------
# Train a single classifier per mouse and plot average cross-validated accuracy
# Convoluted function because I test mean equalization without leaks to test sets.

def per_mouse_cv_accuracy(vectors, label_encoder, seed=42, n_shuffles=100, return_weights=False, equalize=False, debug=False, n_jobs=20):
    accuracies = []
    chance_accuracies = []
    weights_per_mouse = []
    rng = np.random.default_rng(seed)

    for d in vectors:
        days_per_trial = d['day'].values
        mask = np.isin(days_per_trial, [-2, -1, 1, 2])
        trials = d.values[:, mask].T
        labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
        X = trials
        y = labels

        y_enc = label_encoder.transform(y)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_scores = []
        all_true = []
        all_pred = []
        for train_idx, test_idx in cv.split(X, y_enc):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            # Equalization
            if equalize:
                ref_means = np.nanmean(X_train, axis=0)
                pre_label = label_encoder.transform(['pre'])[0]
                post_label = label_encoder.transform(['post'])[0]
                pre_mask_train = y_train == pre_label
                post_mask_train = y_train == post_label
                for icell in range(X_train.shape[1]):
                    pre_mean = np.nanmean(X_train[pre_mask_train, icell])
                    post_mean = np.nanmean(X_train[post_mask_train, icell])
                    X_train[pre_mask_train, icell] += ref_means[icell] - pre_mean
                    X_train[post_mask_train, icell] += ref_means[icell] - post_mean
                pre_mask_test = y_test == pre_label
                post_mask_test = y_test == post_label
                for icell in range(X_test.shape[1]):
                    pre_mean = np.nanmean(X_test[pre_mask_test, icell])
                    post_mean = np.nanmean(X_test[post_mask_test, icell])
                    X_test[pre_mask_test, icell] += ref_means[icell] - pre_mean
                    X_test[post_mask_test, icell] += ref_means[icell] - post_mean
            # Scaling (always applied)
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_test_proc = scaler.transform(X_test)
            clf.fit(X_train_proc, y_train)
            y_pred = clf.predict(X_test_proc)
            fold_scores.append(np.mean(y_pred == y_test))
            all_true.extend(y_test)
            all_pred.extend(y_pred)
        acc = np.mean(fold_scores)
        accuracies.append(acc)

        if debug:
            print("Accuracy:", acc)
            print("True labels:", all_true)
            print("Predicted labels:", all_pred)
            print("Label counts (true):", np.bincount(all_true))
            print("Label counts (pred):", np.bincount(all_pred))

        # Estimate chance level by shuffling labels n_shuffles times (parallelized)
        def shuffle_score(clf, X, y_enc, cv):
            y_shuff = rng.permutation(y_enc)
            fold_scores = []
            for train_idx, test_idx in cv.split(X, y_shuff):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_shuff[train_idx], y_shuff[test_idx]
                # Equalization
                if equalize:
                    ref_means = np.nanmean(X_train, axis=0)
                    pre_label = label_encoder.transform(['pre'])[0]
                    post_label = label_encoder.transform(['post'])[0]
                    pre_mask_train = y_train == pre_label
                    post_mask_train = y_train == post_label
                    for icell in range(X_train.shape[1]):
                        pre_mean = np.nanmean(X_train[pre_mask_train, icell])
                        post_mean = np.nanmean(X_train[post_mask_train, icell])
                        X_train[pre_mask_train, icell] += ref_means[icell] - pre_mean
                        X_train[post_mask_train, icell] += ref_means[icell] - post_mean
                    pre_mask_test = y_test == pre_label
                    post_mask_test = y_test == post_label
                    for icell in range(X_test.shape[1]):
                        pre_mean = np.nanmean(X_test[pre_mask_test, icell])
                        post_mean = np.nanmean(X_test[post_mask_test, icell])
                        X_test[pre_mask_test, icell] += ref_means[icell] - pre_mean
                        X_test[post_mask_test, icell] += ref_means[icell] - post_mean
                # Scaling (always applied)
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_test_proc = scaler.transform(X_test)
                clf.fit(X_train_proc, y_train)
                y_pred = clf.predict(X_test_proc)
                fold_scores.append(np.mean(y_pred == y_test))
            return np.mean(fold_scores)

        shuffle_scores = Parallel(n_jobs=n_jobs)(
            delayed(shuffle_score)(clf, X, y_enc, cv) for _ in range(n_shuffles)
        )
        chance_accuracies.append(np.mean(shuffle_scores))

        if return_weights:
            # Train classifier on full dataset and return weights
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(X)
            clf_full = LogisticRegression(max_iter=5000, random_state=seed)
            clf_full.fit(X_full, y_enc)
            weights_per_mouse.append(clf_full.coef_.flatten())

    if return_weights:
        return np.array(accuracies), np.array(chance_accuracies), weights_per_mouse
    else:
        return np.array(accuracies), np.array(chance_accuracies)


le = LabelEncoder()
le.fit(['pre', 'post'])

accs_rew, chance_rew, weights_rew = per_mouse_cv_accuracy(vectors_rew, le, n_shuffles=1, return_weights=True, equalize=True)
accs_nonrew, chance_nonrew, weights_nonrew = per_mouse_cv_accuracy(vectors_nonrew, le, n_shuffles=1, return_weights=True, equalize=True)

for w in weights_rew:
    plt.plot(w, label='R+')

print(f"Mean accuracy R+: {np.nanmean(accs_rew):.3f} +/- {np.nanstd(accs_rew):.3f}")
print(f"Mean accuracy R-: {np.nanmean(accs_nonrew):.3f} +/- {np.nanstd(accs_nonrew):.3f}")

# Plot
plt.figure(figsize=(4, 5))
# Plot chance levels in grey
sns.pointplot(data=[chance_rew, chance_nonrew], color='grey', estimator=np.nanmean, errorbar='ci', linestyles="none")
# Plot actual accuracies
sns.stripplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], alpha=0.7)
sns.pointplot(data=[accs_rew, accs_nonrew], palette=reward_palette[::-1], linestyle=None, estimator=np.nanmean, errorbar='ci')
plt.xticks([0, 1], ['R+', 'R-'])
plt.ylabel('Cross-validated accuracy')
plt.title('Pre vs post learning classification accuracy across mice')
plt.ylim(0, 1)
sns.despine()

# Statistical test: Mann-Whitney U test between R+ and R- accuracies
stat, p_value = mannwhitneyu(accs_rew, accs_nonrew, alternative='two-sided')
print(f"Mann-Whitney U test: stat={stat:.3f}, p-value={p_value:.4f}")

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = 'decoding_accuracy.svg'
if projection_type is not None:
    svg_file = f'decoding_accuracy_{projection_type}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save results.
results_df = pd.DataFrame({
    'accuracy': np.concatenate([accs_rew, accs_nonrew]),
    'reward_group': ['R+'] * len(accs_rew) + ['R-'] * len(accs_nonrew),
    'chance_accuracy': np.concatenate([chance_rew, chance_nonrew]),
})
results_csv_file = 'decoding_accuracy_data.csv'
if projection_type is not None:
    results_csv_file = f'decoding_accuracy_data_{projection_type}.csv'
results_df.to_csv(os.path.join(output_dir, results_csv_file), index=False)
# Save stats.

stat_file = 'decoding_accuracy_stats.csv'
if projection_type is not None:
    stat_file = f'decoding_accuracy_stats_{projection_type}.csv'
pd.DataFrame({'stat': [stat], 'p_value': [p_value]}).to_csv(os.path.join(output_dir, stat_file), index=False)


# Decoding accuracy using a decoder trained on day -2 vs +2
# ---------------------------------------------------------
# For each mouse, train a classifier to distinguish activity patterns on day -2 vs +2.
# Then, use this trained classifier to decode every pair of days (including each day against itself).
# Plot accuracy as a matrix for each reward group.
 
all_days = [-2, -1, 0, 1, 2]
day_labels = [str(d) for d in all_days]

def pairwise_day_decoding_fixed_decoder_cv(vectors, days, seed=42, n_splits=10):
    # Returns: list of accuracy matrices (n_mice x n_days x n_days)
    acc_matrices = []
    rng = np.random.default_rng(seed)
    for d in vectors:
        acc_matrix = np.full((len(days), len(days)), np.nan)
        day_per_trial = d['day'].values
        # For each fold
        fold_accs = np.zeros((n_splits, len(days), len(days)))
        # Get indices for each day
        day_indices = {day: np.where(day_per_trial == day)[0] for day in days}
        # For each fold
        for k in range(n_splits):
            # Split indices for each day
            train_idx = []
            test_idx = []
            for day in days:
                idx = day_indices[day]
                if len(idx) < n_splits:
                    continue  # skip if not enough trials
                idx_shuff = rng.permutation(idx)
                fold_size = len(idx) // n_splits
                test_fold = idx_shuff[k*fold_size:(k+1)*fold_size] if k < n_splits-1 else idx_shuff[k*fold_size:]
                train_fold = np.setdiff1d(idx, test_fold)
                train_idx.append((day, train_fold))
                test_idx.append((day, test_fold))
            # Train decoder on -2 vs +2 (using train data only)
            train_days = [-2, +2]
            train_mask = np.concatenate([fold for day, fold in train_idx if day in train_days])
            train_labels = np.concatenate([[day]*len(fold) for day, fold in train_idx if day in train_days])
            if len(train_mask) < 2 or len(np.unique(train_labels)) < 2:
                continue
            X_train = d.values[:, train_mask].T
            y_train = np.array([0 if day == -2 else 1 for day in train_labels])
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train_proc, y_train)
            # Test decoder on all pairs of days (using test data only)
            for i, day_i in enumerate(days):
                for j, day_j in enumerate(days):
                    test_mask_i = [fold for day, fold in test_idx if day == day_i]
                    test_mask_j = [fold for day, fold in test_idx if day == day_j]
                    if not test_mask_i or not test_mask_j:
                        continue
                    test_mask_i = test_mask_i[0]
                    test_mask_j = test_mask_j[0]
                    if len(test_mask_i) < 1 or len(test_mask_j) < 1:
                        continue
                    X_test = np.concatenate([d.values[:, test_mask_i].T, d.values[:, test_mask_j].T], axis=0)
                    y_test = np.array([0] * len(test_mask_i) + [1] * len(test_mask_j))
                    X_test_proc = scaler.transform(X_test)
                    y_pred = clf.predict(X_test_proc)
                    acc = np.mean(y_pred == y_test)
                    fold_accs[k, i, j] = acc
        # Average over folds
        with np.errstate(invalid='ignore'):
            acc_matrix = np.nanmean(fold_accs, axis=0)
        acc_matrices.append(acc_matrix)
    return np.array(acc_matrices)

# Compute accuracy matrices for each group
accs_rew_matrix = pairwise_day_decoding_fixed_decoder_cv(vectors_rew, all_days)
accs_nonrew_matrix = pairwise_day_decoding_fixed_decoder_cv(vectors_nonrew, all_days)
# Average across mice
mean_accs_rew = np.nanmean(accs_rew_matrix, axis=0)
mean_accs_nonrew = np.nanmean(accs_nonrew_matrix, axis=0)

# Make matrices symmetric with filled diagonal for aesthetics
def make_symmetric_with_diag(mat):
    sym_mat = np.full_like(mat, np.nan)
    iu = np.triu_indices_from(mat, k=1)
    sym_mat[iu] = mat[iu]
    il = np.tril_indices_from(mat, k=-1)
    sym_mat[il] = mat.T[il]
    diag = np.diag_indices_from(mat)
    sym_mat[diag] = np.diag(mat)
    return sym_mat

mean_accs_rew_sym = make_symmetric_with_diag(mean_accs_rew)
mean_accs_nonrew_sym = make_symmetric_with_diag(mean_accs_nonrew)

# Plot accuracy matrices and shared colormap
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
vmax = max(np.nanmax(mean_accs_rew_sym), np.nanmax(mean_accs_nonrew_sym))
vmin = 0.5

sns.heatmap(mean_accs_rew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax0, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_accs_rew_sym), cbar=False)
ax0.set_title("R+ group: Fixed decoder (-2 vs +2) pairwise day accuracy (CV)")
ax0.set_xlabel("Day")
ax0.set_ylabel("Day")

sns.heatmap(mean_accs_nonrew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax1, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_accs_nonrew_sym), cbar=False)
ax1.set_title("R- group: Fixed decoder (-2 vs +2) pairwise day accuracy (CV)")
ax1.set_xlabel("Day")
ax1.set_ylabel("Day")

# Shared colorbar
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=ax2)

plt.tight_layout()
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix.png'), format='png', dpi=300)

# Save data
np.save(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix_Rplus.npy'), mean_accs_rew)
np.save(os.path.join(output_dir, 'pairwise_day_decoding_fixed_decoder_accuracy_matrix_Rminus.npy'), mean_accs_nonrew)


# Compute chance accuracy matrices by shuffling labels
def pairwise_day_decoding_chance(vectors, days, n_splits=10, seed=42, n_shuffles=5):
    acc_matrices = []
    rng = np.random.default_rng(seed)
    for d in vectors:
        acc_matrix = np.full((len(days), len(days)), np.nan)
        for i, day_i in enumerate(days):
            for j, day_j in enumerate(days):
                if i >= j:
                    continue
                day_per_trial = d['day'].values
                mask_i = day_per_trial == day_i
                mask_j = day_per_trial == day_j
                if np.sum(mask_i) < 2 or np.sum(mask_j) < 2:
                    continue
                X = np.concatenate([d.values[:, mask_i].T, d.values[:, mask_j].T], axis=0)
                y = np.array([day_i] * np.sum(mask_i) + [day_j] * np.sum(mask_j))
                le_pair = LabelEncoder()
                y_enc = le_pair.fit_transform(y)
                scores = []
                for _ in range(n_shuffles):
                    y_shuff = rng.permutation(y_enc)
                    clf = LogisticRegression(max_iter=5000, random_state=seed)
                    cv = StratifiedKFold(n_splits=min(n_splits, len(y_enc)), shuffle=True, random_state=seed)
                    fold_scores = []
                    for train_idx, test_idx in cv.split(X, y_shuff):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y_shuff[train_idx], y_shuff[test_idx]
                        scaler = StandardScaler()
                        X_train_proc = scaler.fit_transform(X_train)
                        X_test_proc = scaler.transform(X_test)
                        clf.fit(X_train_proc, y_train)
                        y_pred = clf.predict(X_test_proc)
                        fold_scores.append(np.mean(y_pred == y_test))
                    scores.append(np.mean(fold_scores))
                acc_matrix[i, j] = np.mean(scores)
        acc_matrices.append(acc_matrix)
    return np.array(acc_matrices)

# Compute chance matrices for each group
chance_accs_rew_matrix = pairwise_day_decoding_chance(vectors_rew, all_days, n_shuffles=100)
chance_accs_nonrew_matrix = pairwise_day_decoding_chance(vectors_nonrew, all_days, n_shuffles=100)
mean_chance_accs_rew = np.nanmean(chance_accs_rew_matrix, axis=0)
mean_chance_accs_nonrew = np.nanmean(chance_accs_nonrew_matrix, axis=0)
mean_chance_accs_rew_sym = make_symmetric_with_empty_diag(mean_chance_accs_rew)
mean_chance_accs_nonrew_sym = make_symmetric_with_empty_diag(mean_chance_accs_nonrew)

# Plot chance accuracy matrices
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
vmax = max(np.nanmax(mean_chance_accs_rew_sym), np.nanmax(mean_chance_accs_nonrew_sym))
vmin = 0.5

sns.heatmap(mean_chance_accs_rew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax0, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_chance_accs_rew_sym), cbar=False)
ax0.set_title("R+ group: Pairwise day chance accuracy")
ax0.set_xlabel("Day")
ax0.set_ylabel("Day")

sns.heatmap(mean_chance_accs_nonrew_sym, annot=True, fmt=".2f", xticklabels=day_labels, yticklabels=day_labels,
            ax=ax1, cmap="viridis", vmin=vmin, vmax=vmax, mask=np.isnan(mean_chance_accs_nonrew_sym), cbar=False)
ax1.set_title("R- group: Pairwise day chance accuracy")
ax1.set_xlabel("Day")
ax1.set_ylabel("Day")

norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=ax2)

plt.tight_layout()
sns.despine()

# Save figure
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_chance_accuracy_matrix.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'pairwise_day_decoding_chance_accuracy_matrix.png'), format='png', dpi=300)

# Save data
np.save(os.path.join(output_dir, 'pairwise_day_decoding_chance_accuracy_matrix_Rplus.npy'), mean_chance_accs_rew)
np.save(os.path.join(output_dir, 'pairwise_day_decoding_chance_accuracy_matrix_Rminus.npy'), mean_chance_accs_nonrew)



# Relationship between classifier weights and learning modualtion index.
# ----------------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

# Merge classifier weights and LMI for each mouse
weights_all = []
lmi_all = []
mouse_ids = []

for i, mouse in enumerate(mice_rew + mice_nonrew):
    # Get classifier weights for this mouse
    if mouse in mice_rew:
        w = weights_rew[mice_rew.index(mouse)]
    else:
        w = weights_nonrew[mice_nonrew.index(mouse)]
    # Get cell IDs
    d = vectors_rew[mice_rew.index(mouse)] if mouse in mice_rew else vectors_nonrew[mice_nonrew.index(mouse)]
    cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
    # Get LMI for this mouse
    lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
    lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
    lmi_vals = lmi_mouse['lmi'].values
    # Only keep cells with non-nan LMI and weights
    mask = ~np.isnan(lmi_vals) & ~np.isnan(w)
    weights_all.append(w[mask])
    lmi_all.append(lmi_vals[mask])
    mouse_ids.extend([mouse] * np.sum(mask))
    # Flatten lists
    weights_flat = np.concatenate(weights_all)
    lmi_flat = np.concatenate(lmi_all)
    mouse_ids_flat = np.array(mouse_ids)

# Plot scatter and regression for each mouse
plt.figure(figsize=(4, 4))

for i, mouse in enumerate(np.unique(mouse_ids_flat)):
    mask = mouse_ids_flat == mouse
    sns.scatterplot(x=lmi_flat[mask], y=weights_flat[mask], alpha=0.5)
    # # Regression line for each mouse
    # if np.sum(mask) > 1:
    #     reg = LinearRegression().fit(lmi_flat[mask].reshape(-1, 1), weights_flat[mask])
    #     x_vals = np.linspace(np.nanmin(lmi_flat[mask]), np.nanmax(lmi_flat[mask]), 100)
    #     plt.plot(x_vals, reg.predict(x_vals.reshape(-1, 1)), color='grey', alpha=0.5)

# Main regression line for all mice
reg_all = LinearRegression().fit(lmi_flat.reshape(-1, 1), weights_flat)
x_vals = np.linspace(np.nanmin(lmi_flat), np.nanmax(lmi_flat), 100)
y_pred = reg_all.predict(x_vals.reshape(-1, 1))
plt.plot(x_vals, y_pred, color='#2d2d2d', linewidth=2)

# Bootstrap confidence interval for regression line
n_boot = 1000
y_boot = np.zeros((n_boot, len(x_vals)))
for i in range(n_boot):
    Xb, yb = resample(lmi_flat, weights_flat)
    regb = LinearRegression().fit(Xb.reshape(-1, 1), yb)
    y_boot[i] = regb.predict(x_vals.reshape(-1, 1))
ci_low = np.percentile(y_boot, 2.5, axis=0)
ci_high = np.percentile(y_boot, 97.5, axis=0)
plt.fill_between(x_vals, ci_low, ci_high, color='black', alpha=0.2, label='95% CI')

plt.xlabel('Learning Modulation Index (LMI)')
plt.ylabel('Classifier Weight')
plt.title('Classifier Weight vs LMI')
plt.tight_layout()
plt.ylim(-2.5, 2)
sns.despine()

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_mouse.png'), format='png', dpi=300)


# Fit linear regression
reg = LinearRegression()
reg.fit(X, y)
r2 = reg.score(X, y)
print(f"Linear regression R^2: {r2:.3f}")

# Bootstrapped confidence interval for R^2
n_boot = 1000
r2_boot = []
for _ in range(n_boot):
    Xb, yb = resample(X, y)
    regb = LinearRegression().fit(Xb, yb)
    r2_boot.append(regb.score(Xb, yb))
r2_ci = np.percentile(r2_boot, [2.5, 97.5])
print(f"Bootstrapped R^2 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# Save plot
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'classifier_weights_vs_lmi_by_group.svg'), format='svg', dpi=300)


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

# This is to show that without the cells modulated on average, no information
# can be decoded. The non-modulated cells could still carry some information
# similarly to non-place cells in the hippocampus.


# Accuracy as a function of percent most modulated cells removed.
# ---------------------------------------------------------------

lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)

le = LabelEncoder()
le.fit(['pre', 'post'])
percentiles = np.arange(0, 91, 5)  # 0% to 60% in steps of 5%
accs_rew_curve = []
accs_nonrew_curve = []

for perc in percentiles:
    accs_rew_perc = []
    accs_nonrew_perc = []
    for group, vectors, mice_group in zip(
        ['R+', 'R-'],
        [vectors_rew, vectors_nonrew],
        [mice_rew, mice_nonrew]
    ):
        for i, mouse in enumerate(mice_group):
            # Get vector and cell LMI for this mouse
            d = vectors[i]
            cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
            lmi_mouse = lmi_df[(lmi_df['mouse_id'] == mouse) & (lmi_df['roi'].isin(cell_ids))]
            lmi_mouse = lmi_mouse.set_index('roi').reindex(cell_ids)
            abs_lmi = np.abs(lmi_mouse['lmi'].values)
            # Sort cells by abs(LMI)
            sorted_idx = np.argsort(-abs_lmi)  # descending
            n_cells = len(cell_ids)
            n_remove = int(np.round(n_cells * perc / 100))
            keep_idx = sorted_idx[n_remove:]
            # If less than 2 cells remain, skip
            if len(keep_idx) < 2:
                continue
            # Subset vector
            d_sub = d.isel({d.dims[0]: keep_idx})
            # Classification
            days_per_trial = d_sub['day'].values
            mask = np.isin(days_per_trial, [-2, -1, 1, 2])
            trials = d_sub.values[:, mask].T
            labels = np.array(['pre' if day in [-2, -1] else 'post' for day in days_per_trial[mask]])
            y_enc = le.transform(labels)
            clf = LogisticRegression(max_iter=50000)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scaler = StandardScaler()
            X = scaler.fit_transform(trials)
            scores = cross_val_score(clf, X, y_enc, cv=cv, n_jobs=1)
            acc = np.mean(scores)
            if group == 'R+':
                accs_rew_perc.append(acc)
            else:
                accs_nonrew_perc.append(acc)
    accs_rew_curve.append(accs_rew_perc)
    accs_nonrew_curve.append(accs_nonrew_perc)
# Plot with bootstrap confidence intervals
mean_rew = []
ci_rew = []
mean_nonrew = []
ci_nonrew = []
x_vals = 100 - percentiles  # 100% to 40% retained

for a in accs_rew_curve:
    mean_rew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_rew.append(res.confidence_interval)
    else:
        ci_rew.append((np.nan, np.nan))
for a in accs_nonrew_curve:
    mean_nonrew.append(np.nanmean(a))
    if len(a) > 1:
        res = bootstrap((np.array(a),), np.nanmean, confidence_level=0.95, n_resamples=1000, method='basic')
        ci_nonrew.append(res.confidence_interval)
    else:
        ci_nonrew.append((np.nan, np.nan))

# Prepare DataFrame for seaborn
df_plot = pd.DataFrame({
    'percent_cells_retained': np.tile(x_vals, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'ci_low': np.concatenate([np.array([ci.low for ci in ci_rew]), np.array([ci.low for ci in ci_nonrew])]),
    'ci_high': np.concatenate([np.array([ci.high for ci in ci_rew]), np.array([ci.high for ci in ci_nonrew])]),
    'reward_group': ['R+'] * len(x_vals) + ['R-'] * len(x_vals)
})

plt.figure(figsize=(6, 5))
sns.lineplot(data=df_plot, x='percent_cells_retained', y='accuracy', hue='reward_group', palette=reward_palette[::-1])
for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_plot[df_plot['reward_group'] == group]
    plt.fill_between(sub['percent_cells_retained'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.3)
plt.xlabel('Percent of cells retained')
plt.ylabel('Classification accuracy')
plt.title('Accuracy vs percent modulated cells retained')
plt.legend()
plt.ylim(0, 1)
plt.xlim(100, min(x_vals))  # Flip x-axis: start at 100% and go down
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/decoding'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.svg'), format='svg', dpi=300)

# Save data
curve_df = pd.DataFrame({
    'percent_cells_retained': np.tile(100 - percentiles, 2),
    'accuracy': np.concatenate([mean_rew, mean_nonrew]),
    'reward_group': ['R+'] * len(percentiles) + ['R-'] * len(percentiles)
})
curve_df.to_csv(os.path.join(output_dir, 'accuracy_vs_percent_modulated_cells.csv'), index=False)


# Inflexion/progressive learning during Day 0.
# #############################################

# The idea is to see if decoding accuracy improves progressively during Day 0 learning trials,
# indicating progressive learning, or if it jumps suddenly at some point,
# indicating an inflexion point, or if it remains flat.
# Use a decoder trained on Day -2 vs Day +2 as before, possibly using a
# sliding window. Look both at the classification of each trial/group of trials
# and at the decision values (distance to the hyperplane) as a continuous measure.
# We want to correlate that with performance across mice.

sampling_rate = 30
win = (0, 0.180)  # from stimulus onset to 180 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40 
substract_baseline = True
select_responsive_cells = False
select_lmi = False
projection_type = None  # 'wS2', 'wM1' or None

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# mice = [m for m in mice if m in mice_groups['good_day0']]

# Load data.
vectors_rew_mapping = []
vectors_nonrew_mapping = []
mice_rew = []
mice_nonrew = []
vectors_nonrew_day0_learning = []
vectors_rew_day0_learning = []

# Load responsive cells.
# Responsiveness df.
# test_df = os.path.join(io.processed_dir, f'response_test_results_alldaystogether_win_180ms.csv')
# test_df = pd.read_csv(test_df)
# test_df = test_df.loc[test_df['mouse_id'].isin(mice)]
# selected_cells = test_df.loc[test_df['pval_mapping'] <= 0.05]

if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_180ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]
    
# Load behaviour table with learning trials.
path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)
# Select day 0 performance for whisker trials.
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1), ['mouse_id', 'trial_w', 'learning_curve_w']]

for mouse in mice:
    
    # Load mapping data.
    # ------------------
    
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name, substracted=True)
    # xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Check that each day has at least n_map_trials mapping trials
    # and select the first n_map_trials mapping trials for each day.
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
    # Average bins.
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(d.values).sum(), 'NaN values in the data.')
    d = d.fillna(0)
    
    if rew_gp == 'R-':
        vectors_nonrew_mapping.append(d)
        mice_nonrew.append(mouse)
    elif rew_gp == 'R+':
        vectors_rew_mapping.append(d)
        mice_rew.append(mouse)

    # Load learning data for day 0.
    # -----------------------------
    
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging_utils.load_mouse_xarray(mouse, folder, file_name)
    # xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin([0]))
    # Select whisker trials.
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] == 0:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Average bins.
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Remove artefacts by setting them at 0. To avoid NaN values and
    # mismatches (that concerns a single cell).
    print(np.isnan(xarray.values).sum(), 'NaN values in the data.')
    xarray = xarray.fillna(0)
        
    if rew_gp == 'R-':
        vectors_nonrew_day0_learning.append(xarray)
    elif rew_gp == 'R+':
        vectors_rew_day0_learning.append(xarray)
    


def progressive_learning_analysis(vectors_mapping, vectors_learning, mice_list, pre_days=[-2, -1], post_days=[1, 2], window_size=10, step_size=5, seed=42):
    """
    Analyze progressive learning during Day 0 using a sliding window approach.
    Train decoder on Day -2 vs +2 mapping trials, then apply to Day 0 learning trials.
    """
    results = []
    
    for i, (d_mapping, d_learning, mouse) in enumerate(zip(vectors_mapping, vectors_learning, mice_list)):
        print(d_mapping.shape, d_learning.shape)
        day_per_trial = d_mapping['day'].values
        
        # Get Day -2 and +2 trials for training from mapping data
        train_mask = np.isin(day_per_trial, pre_days + post_days)
            
        X_train = d_mapping.values[:, train_mask].T
        y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

        # Train classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        # clf = SVC(kernel='linear', random_state=seed)
        clf.fit(X_train_scaled, y_train)
        
        # Get Day 0 learning trials
        n_learning_trials = d_learning.sizes['trial']
            
        # Sliding window analysis on learning trials
        window_results = []
        for start_idx in range(0, n_learning_trials - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Get window data from learning trials
            X_window = d_learning.values[:, start_idx:end_idx].T
            X_window_scaled = scaler.transform(X_window)
            
            # Get decision values (distance to hyperplane)
            decision_values = clf.decision_function(X_window_scaled)
            mean_decision_value = np.mean(decision_values)
            
            # Get prediction probabilities
            # proba = clf.predict_proba(X_window_scaled)
            # mean_proba_post = np.mean(proba[:, 1])  # Probability of being "post"
            
            window_results.append({
                'window_start': start_idx,
                'window_center': start_idx + window_size // 2,
                'window_end': end_idx,
                'trial_start': start_idx,
                'trial_center': start_idx + window_size // 2,
                'trial_end': end_idx,
                'mean_decision_value': mean_decision_value,
                'mouse_idx': i,
                'mouse_id': mouse
            })
            
        results.extend(window_results)
    
    return pd.DataFrame(results)


# Run analysis for both groups
window_size = 10
step_size = 1

mice_good = [m for m in mice_rew if m in mice_groups['good_day0']]
mice_good_mask = np.isin(mice_rew, mice_good)
vectors_good_mice = [vectors_rew_mapping[i] for i in range(len(vectors_rew_mapping)) if mice_good_mask[i]]
vectors_learning_good_mice = [vectors_rew_day0_learning[i] for i in range(len(vectors_rew_day0_learning)) if mice_good_mask[i]]

mice_bad = [m for m in mice_rew if m in mice_groups['bad_day0']+mice_groups['meh_day0']]
mice_bad_mask = np.isin(mice_rew, mice_bad)
vectors_bad_mice = [vectors_rew_mapping[i] for i in range(len(vectors_rew_mapping)) if mice_bad_mask[i]]
vectors_learning_bad_mice = [vectors_rew_day0_learning[i] for i in range(len(vectors_rew_day0_learning)) if mice_bad_mask[i]]

results_rew = progressive_learning_analysis(vectors_rew_mapping, vectors_rew_day0_learning, mice_rew, window_size=window_size, step_size=step_size)
results_nonrew = progressive_learning_analysis(vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew, window_size=window_size, step_size=step_size)
results_good = progressive_learning_analysis(vectors_good_mice, vectors_learning_good_mice, mice_good, window_size=window_size, step_size=step_size)
results_bad = progressive_learning_analysis(vectors_bad_mice, vectors_learning_bad_mice, mice_bad, window_size=window_size, step_size=step_size)

results_rew['reward_group'] = 'R+'
results_nonrew['reward_group'] = 'R-'
results_good['group'] = 'good'
results_bad['group'] = 'bad'
results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)

cut_n_trials = 100

def plot_behavior(ax, data, color, title, cut_n_trials=cut_n_trials):
    sns.lineplot(data=data, x='trial_w', y='learning_curve_w', color=color, errorbar='ci', ax=ax)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Learning curve (w)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    if cut_n_trials is not None:
        ax.set_xlim(0, cut_n_trials)

def plot_decision(ax, data, color, title, cut_n_trials=cut_n_trials):
    sns.lineplot(data=data, x='trial_center', y='mean_decision_value', estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Mean decision value')
    ax.set_title(title)
    ax.set_ylim(-3, 5)
    if cut_n_trials is not None:
        ax.set_xlim(0, cut_n_trials)

plt.figure(figsize=(16, 10))

# Top row: Behavioral learning curves
ax1 = plt.subplot(2, 4, 1)
data_rew = bh_df.loc[bh_df['mouse_id'].isin(mice_rew)]
plot_behavior(ax1, data_rew, reward_palette[1], 'R+ mice behavior')

ax2 = plt.subplot(2, 4, 2)
data_nonrew = bh_df.loc[bh_df['mouse_id'].isin(mice_nonrew)]
plot_behavior(ax2, data_nonrew, reward_palette[0], 'R- mice behavior')

ax3 = plt.subplot(2, 4, 3)
data_good = bh_df.loc[bh_df['mouse_id'].isin(mice_good)]
plot_behavior(ax3, data_good, reward_palette[1], 'Good day0 mice behavior')

ax4 = plt.subplot(2, 4, 4)
data_bad = bh_df.loc[bh_df['mouse_id'].isin(mice_bad)]
plot_behavior(ax4, data_bad, reward_palette[1], 'Bad day0 mice behavior')

# Bottom row: Decision values
ax5 = plt.subplot(2, 4, 5)
plot_decision(ax5, results_rew, reward_palette[1], 'R+ mice')

ax6 = plt.subplot(2, 4, 6)
plot_decision(ax6, results_nonrew, reward_palette[0], 'R- mice')

ax7 = plt.subplot(2, 4, 7)
plot_decision(ax7, results_good, reward_palette[1], 'Good day0 mice')

ax8 = plt.subplot(2, 4, 8)
plot_decision(ax8, results_bad, reward_palette[1], 'Bad day0 mice')

plt.tight_layout()
sns.despine()

# Save figure
plt.savefig(os.path.join(output_dir, 'progressive_learning_day0.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'progressive_learning_day0.png'), format='png', dpi=300)




# Statistical analysis: Test for trend using linear regression
print("\nStatistical analysis of progressive learning:")
for group in ['R+', 'R-']:
    group_data = results_combined[results_combined['reward_group'] == group]
    if len(group_data) > 0:
        # Linear regression on decision values
        X_reg = group_data['trial_center'].values.reshape(-1, 1)
        y_reg_decision = group_data['mean_decision_value'].values
        y_reg_proba = group_data['mean_proba_post'].values
        
        reg_decision = LinearRegression().fit(X_reg, y_reg_decision)
        reg_proba = LinearRegression().fit(X_reg, y_reg_proba)
        
        # Calculate correlation coefficients
        r_decision, p_decision = spearmanr(group_data['trial_center'], group_data['mean_decision_value'])
        r_proba, p_proba = spearmanr(group_data['trial_center'], group_data['mean_proba_post'])
        
        print(f"{group} group:")
        print(f"  Decision values - slope: {reg_decision.coef_[0]:.4f}, R²: {reg_decision.score(X_reg, y_reg_decision):.3f}")
        print(f"  Decision values - Spearman r: {r_decision:.3f}, p: {p_decision:.4f}")
        print(f"  Probability - slope: {reg_proba.coef_[0]:.4f}, R²: {reg_proba.score(X_reg, y_reg_proba):.3f}")
        print(f"  Probability - Spearman r: {r_proba:.3f}, p: {p_proba:.4f}")

# Save results
results_combined.to_csv(os.path.join(output_dir, 'progressive_learning_day0_data.csv'), index=False)

# Additional analysis: Trial-by-trial classification for individual mice
def single_trial_classification_day0(vectors_mapping, vectors_learning, mice_list, seed=42):
    """
    Classify each individual trial in Day 0 learning using decoder trained on Day -2 vs +2 mapping.
    """
    trial_results = []
    
    for i, (d_mapping, d_learning, mouse) in enumerate(zip(vectors_mapping, vectors_learning, mice_list)):
        day_per_trial = d_mapping['day'].values
        
        # Train on Day -2 vs +2 mapping trials
        train_mask = np.isin(day_per_trial, [-2, 2])
        if np.sum(train_mask) < 10:
            continue
            
        X_train = d_mapping.values[:, train_mask].T
        y_train = np.array([0 if day == -2 else 1 for day in day_per_trial[train_mask]])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train_scaled, y_train)
        
        # Test on Day 0 learning trials
        n_learning_trials = d_learning.sizes['trial']
        
        for j in range(n_learning_trials):
            X_trial = d_learning.values[:, j:j+1].T
            X_trial_scaled = scaler.transform(X_trial)
            
            decision_value = clf.decision_function(X_trial_scaled)[0]
            proba_post = clf.predict_proba(X_trial_scaled)[0, 1]
            prediction = clf.predict(X_trial_scaled)[0]
            
            trial_results.append({
                'mouse_idx': i,
                'mouse_id': mouse,
                'trial_in_day0': j,
                'decision_value': decision_value,
                'proba_post': proba_post,
                'prediction': prediction
            })
    
    return pd.DataFrame(trial_results)

# Run single trial analysis
print("\nAnalyzing single trial classifications...")
trial_results_rew = single_trial_classification_day0(vectors_rew, vectors_rew_day0_learning, mice_rew)
trial_results_nonrew = single_trial_classification_day0(vectors_nonrew, vectors_nonrew_day0_learning, mice_nonrew)

trial_results_rew['reward_group'] = 'R+'
trial_results_nonrew['reward_group'] = 'R-'
trial_results_all = pd.concat([trial_results_rew, trial_results_nonrew], ignore_index=True)

# Plot single trial results
plt.figure(figsize=(12, 8))

# Plot decision values for individual trials
plt.subplot(2, 2, 1)
sns.scatterplot(data=trial_results_all, x='trial_in_day0', y='decision_value', 
                hue='reward_group', palette=reward_palette[::-1], alpha=0.6)
sns.lineplot(data=trial_results_all, x='trial_in_day0', y='decision_value', 
                hue='reward_group', palette=reward_palette[::-1], estimator=np.mean, errorbar='ci')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Trial in Day 0')
plt.ylabel('Decision value')
plt.title('Single trial decision values')

# Plot probability for individual trials
plt.subplot(2, 2, 2)
sns.scatterplot(data=trial_results_all, x='trial_in_day0', y='proba_post', 
                hue='reward_group', palette=reward_palette[::-1], alpha=0.6)
sns.lineplot(data=trial_results_all, x='trial_in_day0', y='proba_post', 
                hue='reward_group', palette=reward_palette[::-1], estimator=np.mean, errorbar='ci')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Trial in Day 0')
plt.ylabel('Probability "post"')
plt.title('Single trial probabilities')

# Cumulative accuracy plot
plt.subplot(2, 2, 3)
for group in ['R+', 'R-']:
    group_data = trial_results_all[trial_results_all['reward_group'] == group]
    if len(group_data) > 0:
        # Calculate cumulative accuracy for each mouse
        mouse_cumulative = []
        for mouse_idx in group_data['mouse_idx'].unique():
            mouse_data = group_data[group_data['mouse_idx'] == mouse_idx].sort_values('trial_in_day0')
            cumulative_acc = np.cumsum(mouse_data['prediction']) / (np.arange(len(mouse_data)) + 1)
            mouse_cumulative.append(cumulative_acc)
        
        if mouse_cumulative:
            # Pad sequences to same length and average
            max_len = max(len(seq) for seq in mouse_cumulative)
            padded = np.full((len(mouse_cumulative), max_len), np.nan)
            for i, seq in enumerate(mouse_cumulative):
                padded[i, :len(seq)] = seq
            
            mean_cumulative = np.nanmean(padded, axis=0)
            sem_cumulative = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))
            
            x_vals = np.arange(max_len)
            color = reward_palette[::-1][0] if group == 'R+' else reward_palette[::-1][1]
            plt.plot(x_vals, mean_cumulative, color=color, label=group)
            plt.fill_between(x_vals, mean_cumulative - sem_cumulative, 
                            mean_cumulative + sem_cumulative, color=color, alpha=0.3)

plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Trial in Day 0')
plt.ylabel('Cumulative fraction "post"')
plt.title('Cumulative classification')
plt.legend()

# Distribution of decision values across trials
plt.subplot(2, 2, 4)
sns.boxplot(data=trial_results_all, x='reward_group', y='decision_value', palette=reward_palette[::-1])
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.ylabel('Decision value')
plt.title('Distribution of decision values')

plt.tight_layout()
sns.despine()

# Save figure
plt.savefig(os.path.join(output_dir, 'single_trial_classification_day0.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'single_trial_classification_day0.png'), format='png', dpi=300)

# Save trial results
trial_results_all.to_csv(os.path.join(output_dir, 'single_trial_classification_day0_data.csv'), index=False)

print(f"\nAnalysis complete. Results saved to {output_dir}")


