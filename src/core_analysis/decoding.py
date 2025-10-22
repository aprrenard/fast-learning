import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp
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
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

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


#  ############################################
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

# Load behaviour table with learning trials.
path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)
# Select day 0 performance for whisker trials.
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1), ['mouse_id', 'trial_w', 'learning_curve_w', 'learning_trial']]

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
    

def progressive_learning_analysis(vectors_mapping, vectors_learning, mice_list, bh_df=None,
                                  pre_days=[-2, -1], post_days=[1, 2], window_size=10, step_size=5, 
                                  align_to_learning=False, trials_before=50, trials_after=100, seed=42):
    """
    Analyze progressive learning during Day 0 using a sliding window approach.
    Train decoder on Day -2 vs +2 mapping trials, then apply to Day 0 learning trials.

    Parameters:
    -----------
    vectors_mapping : list
        List of xarrays with mapping data for each mouse
    vectors_learning : list
        List of xarrays with Day 0 learning data for each mouse
    mice_list : list
        List of mouse IDs
    bh_df : pd.DataFrame, optional
        Behavioral dataframe with 'learning_trial' column for alignment
    pre_days : list
        Days to use as "pre-learning" for training (default: [-2, -1])
    post_days : list
        Days to use as "post-learning" for training (default: [1, 2])
    window_size : int
        Number of trials in each sliding window
    step_size : int
        Step size for sliding window
    align_to_learning : bool
        If True, align trials to learning onset (trial 0 = learning_trial)
    trials_before : int
        Number of trials before learning onset to include (only if align_to_learning=True)
    trials_after : int
        Number of trials after learning onset to include (only if align_to_learning=True)
    seed : int
        Random seed for classifier

    Returns:
    --------
    pd.DataFrame with window results including trial indices (absolute and relative to learning)
    """
    results = []

    for i, (d_mapping, d_learning, mouse) in enumerate(zip(vectors_mapping, vectors_learning, mice_list)):
        print(d_mapping.shape, d_learning.shape)
        day_per_trial = d_mapping['day'].values

        # Get Day -2/-1 and +1/+2 trials for training from mapping data
        train_mask = np.isin(day_per_trial, pre_days + post_days)
        if np.sum(train_mask) < 4:
            print(f"Not enough training trials for {mouse}, skipping.")
            continue

        X_train = d_mapping.values[:, train_mask].T
        y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

        # Train classifier
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train_scaled, y_train)

        # Sanity check: Verify sign convention is correct (decision_function sign)
        pre_mask = np.isin(day_per_trial, pre_days)
        post_mask = np.isin(day_per_trial, post_days)
        if np.sum(pre_mask) > 0 and np.sum(post_mask) > 0:
            X_pre = scaler.transform(d_mapping.values[:, pre_mask].T)
            X_post = scaler.transform(d_mapping.values[:, post_mask].T)
            mean_dec_pre = np.mean(clf.decision_function(X_pre))
            mean_dec_post = np.mean(clf.decision_function(X_post))
        else:
            mean_dec_pre, mean_dec_post = 0.0, 0.0

        if mean_dec_pre > mean_dec_post:
            print(f"WARNING: {mouse} has flipped decision values! Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")
            print(f"  Flipping sign of decision values for plotting consistency.")
            sign_flip = -1
        else:
            sign_flip = 1
            print(f"{mouse}: Decision values oriented. Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")

        # Get learning trial for this mouse if aligning
        learning_trial_idx = None
        if align_to_learning and bh_df is not None:
            mouse_bh = bh_df[bh_df['mouse_id'] == mouse]
            if not mouse_bh.empty and 'learning_trial' in mouse_bh.columns:
                # Get the learning_trial value (should be same for all rows of this mouse)
                learning_trial_val = mouse_bh['learning_trial'].iloc[0]
                if not np.isnan(learning_trial_val):
                    learning_trial_idx = int(learning_trial_val)
                    print(f"{mouse}: Learning onset at trial_w = {learning_trial_idx}")
                else:
                    print(f"{mouse}: No learning trial defined, using absolute indexing")
            else:
                print(f"{mouse}: No behavioral data found, using absolute indexing")

        # Get Day 0 learning trials
        n_learning_trials = d_learning.sizes['trial']

        # Determine trial range to analyze
        if align_to_learning and learning_trial_idx is not None:
            # Align to learning onset: trial 0 = learning_trial
            trial_start = max(0, learning_trial_idx - trials_before)
            trial_end = min(n_learning_trials, learning_trial_idx + trials_after)
            trial_offset = learning_trial_idx  # For converting to relative indices
        else:
            # Use absolute trial indices
            trial_start = 0
            trial_end = n_learning_trials
            trial_offset = 0

        # Sliding window analysis on learning trials
        window_results = []
        for start_idx in range(trial_start, max(trial_start, trial_end - window_size + 1), step_size):
            end_idx = start_idx + window_size
            if end_idx > trial_end:
                break

            # Get window data from learning trials
            X_window = d_learning.values[:, start_idx:end_idx].T
            if X_window.shape[0] == 0:
                continue
            X_window_scaled = scaler.transform(X_window)

            # Decision values (distance to hyperplane)
            decision_values = clf.decision_function(X_window_scaled)
            mean_decision_value = np.mean(decision_values) * sign_flip

            # Probability of being "post"
            if hasattr(clf, "predict_proba"):
                # Use predicted labels to get the proportion of trials classified as "post"
                # (we trained with labels 0=pre, 1=post), not the classifier's confidence.
                preds = clf.predict(X_window_scaled)
                mean_proba_post = np.mean(preds == 1)
            else:
                # If classifier has no predict_proba (e.g., some SVMs), approximate with sigmoid on decision
                dv = decision_values * sign_flip
                mean_proba_post = np.mean(1 / (1 + np.exp(-dv)))

            # Store both absolute and aligned trial indices
            trial_center_abs = start_idx + window_size // 2
            trial_center_aligned = trial_center_abs - trial_offset  # Relative to learning onset
            
            window_results.append({
                'window_start': start_idx,
                'window_center': start_idx + window_size // 2,
                'window_end': end_idx,
                'trial_start': start_idx,
                'trial_center': trial_center_abs,
                'trial_center_aligned': trial_center_aligned,
                'trial_end': end_idx,
                'mean_decision_value': mean_decision_value,
                'mean_proba_post': mean_proba_post,
                'mouse_idx': i,
                'mouse_id': mouse
            })

        results.extend(window_results)

    return pd.DataFrame(results)

# Run analysis for both groups
window_size = 10
step_size = 1

# Set alignment parameters
align_to_learning = True  # Set to True to align to individual learning onset
trials_before = 50  # Number of trials before learning onset to include
trials_after = 100  # Number of trials after learning onset to include

# If mice_groups exist, try to plot good/bad subsets, otherwise plot empty panels
try:
    mice_good = [m for m in mice_rew if m in mice_groups.get('good_day0', [])]
    mice_bad = [m for m in mice_rew if m in (mice_groups.get('bad_day0', []) + mice_groups.get('meh_day0', []))]
except Exception:
    mice_good, mice_bad = [], []

# Build results for R+ and R- (and optionally good/bad subsets)
results_rew = progressive_learning_analysis(
    vectors_rew_mapping, vectors_rew_day0_learning, mice_rew, 
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_nonrew = progressive_learning_analysis(
    vectors_nonrew_mapping, vectors_nonrew_day0_learning, mice_nonrew,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_good = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_good],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_good],
    mice_good,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_bad = progressive_learning_analysis(
    [vectors_rew_mapping[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    [vectors_rew_day0_learning[i] for i, m in enumerate(mice_rew) if m in mice_bad],
    mice_bad,
    bh_df=bh_df, window_size=window_size, step_size=step_size,
    align_to_learning=align_to_learning, trials_before=trials_before, trials_after=trials_after)
results_rew['reward_group'] = 'R+'
results_nonrew['reward_group'] = 'R-'
results_combined = pd.concat([results_rew, results_nonrew], ignore_index=True)

cut_n_trials = 100

def plot_behavior(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return
    
    if align_to_learning and 'learning_trial' in data.columns:
        # Create aligned trial index
        data_plot = data.copy()
        data_plot['trial_w_aligned'] = data_plot.groupby('mouse_id').apply(
            lambda x: x['trial_w'] - x['learning_trial'].iloc[0] if not pd.isna(x['learning_trial'].iloc[0]) else x['trial_w']
        ).reset_index(level=0, drop=True)
        x_col = 'trial_w_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        data_plot = data
        x_col = 'trial_w'
        xlabel = 'Trial within Day 0'
    
    sns.lineplot(data=data_plot, x=x_col, y='learning_curve_w', color=color, errorbar='ci', ax=ax)
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Learning curve (w)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)

def plot_decision(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        return
    
    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'
    
    sns.lineplot(data=data, x=x_col, y='mean_decision_value', estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean decision value')
    ax.set_title(title)
    ax.set_ylim(-3, 6)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)

def plot_proba(ax, data, color, title, cut_n_trials=cut_n_trials, align_to_learning=False):
    if data is None or data.empty:
        ax.set_title(title + " (no data)")
        ax.set_xlim(-cut_n_trials//2 if align_to_learning else 0, cut_n_trials//2 if align_to_learning else cut_n_trials)
        ax.set_ylim(0, 1)
        return
    
    if align_to_learning and 'trial_center_aligned' in data.columns:
        x_col = 'trial_center_aligned'
        xlabel = 'Trial relative to learning onset'
    else:
        x_col = 'trial_center'
        xlabel = 'Trial within Day 0'
    
    sns.lineplot(data=data, x=x_col, y='mean_proba_post', estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0.5, color='black', alpha=0.5, linestyle='--')
    if align_to_learning:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Learning onset')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('P(post)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    if cut_n_trials is not None:
        if align_to_learning:
            ax.set_xlim(-cut_n_trials//2, cut_n_trials//2)
        else:
            ax.set_xlim(0, cut_n_trials)

# Create a 3-row x 4-col figure:
plt.figure(figsize=(16, 12))

# Top row: Behavioral learning curves (4 panels)
ax1 = plt.subplot(3, 4, 1)
data_rew = bh_df.loc[bh_df['mouse_id'].isin(mice_rew)]
plot_behavior(ax1, data_rew, reward_palette[1], 'R+ mice behavior', align_to_learning=align_to_learning)

ax2 = plt.subplot(3, 4, 2)
data_nonrew = bh_df.loc[bh_df['mouse_id'].isin(mice_nonrew)]
plot_behavior(ax2, data_nonrew, reward_palette[0], 'R- mice behavior', align_to_learning=align_to_learning)

ax3 = plt.subplot(3, 4, 3)
data_good = bh_df.loc[bh_df['mouse_id'].isin(mice_good)]
plot_behavior(ax3, data_good, reward_palette[1], 'Good day0 mice behavior', align_to_learning=align_to_learning)

ax4 = plt.subplot(3, 4, 4)
data_bad = bh_df.loc[bh_df['mouse_id'].isin(mice_bad)]
plot_behavior(ax4, data_bad, reward_palette[1], 'Bad day0 mice behavior', align_to_learning=align_to_learning)

# Middle row: Decision values (4 panels)
ax5 = plt.subplot(3, 4, 5)
plot_decision(ax5, results_rew, reward_palette[1], 'R+ mean decision value', align_to_learning=align_to_learning)

ax6 = plt.subplot(3, 4, 6)
plot_decision(ax6, results_nonrew, reward_palette[0], 'R- mean decision value', align_to_learning=align_to_learning)

ax7 = plt.subplot(3, 4, 7)
plot_decision(ax7, results_good, reward_palette[1], 'Good day0 mean decision value', align_to_learning=align_to_learning)

ax8 = plt.subplot(3, 4, 8)
plot_decision(ax8, results_bad, reward_palette[1], 'Bad day0 mean decision value', align_to_learning=align_to_learning)

# Bottom row: Probability P(post) (4 panels) - separated from decision values
ax9 = plt.subplot(3, 4, 9)
plot_proba(ax9, results_rew, reward_palette[1], 'R+ P(post)', align_to_learning=align_to_learning)

ax10 = plt.subplot(3, 4, 10)
plot_proba(ax10, results_nonrew, reward_palette[0], 'R- P(post)', align_to_learning=align_to_learning)

ax11 = plt.subplot(3, 4, 11)
plot_proba(ax11, results_good, reward_palette[1], 'Good day0 P(post)', align_to_learning=align_to_learning)

ax12 = plt.subplot(3, 4, 12)
plot_proba(ax12, results_bad, reward_palette[1], 'Bad day0 P(post)', align_to_learning=align_to_learning)

plt.tight_layout()
sns.despine()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'
output_dir = io.adjust_path_to_host(output_dir)
plt.savefig(os.path.join(output_dir, 'decoder_decision_value_day0_learning_with_alignment_to_learning.svg'), format='svg', dpi=300)


# Correlation between behavioral performance and decision value during Day 0 learning
# ----------------------------------------------------------------------------------

# For each mouse, correlate the behavioral learning curve with the decision value curve
# during Day 0. Test if the population-level correlation is significantly different from zero
# using one-sample tests (Wilcoxon and t-test).

mice = results_combined['mouse_id'].unique()
# mice = [m for m in mice if m in mice_groups['good_day0']]

corr_real = []
reward_groups = []

for mouse in mice:
    # Get reward group for this mouse
    group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
    reward_groups.append(group)
    # Get decision values for this mouse
    dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
    # Get behavioral curve for this mouse
    bh_mouse = bh_df[bh_df['mouse_id'] == mouse]
    
    # Align trials appropriately
    if align_to_learning:
        # Use aligned trial indices (trial_center_aligned for decision, trial_w - learning_trial for behavior)
        learning_trial = bh_mouse['learning_trial'].iloc[0] if 'learning_trial' in bh_mouse.columns else 0
        # Align behavior trials to learning onset
        bh_mouse_aligned = bh_mouse.copy()
        bh_mouse_aligned['trial_w_aligned'] = bh_mouse_aligned['trial_w'] - learning_trial
        # Match on aligned indices
        common_trials_aligned = np.intersect1d(
            dec_mouse['trial_center_aligned'].dropna(), 
            bh_mouse_aligned['trial_w_aligned']
        )
        if len(common_trials_aligned) < 10:
            corr_real.append(np.nan)
            continue
        dec_vals = dec_mouse.set_index('trial_center_aligned').loc[common_trials_aligned]['mean_decision_value'].values
        perf_vals = bh_mouse_aligned.set_index('trial_w_aligned').loc[common_trials_aligned]['learning_curve_w'].values
    else:
        # Use absolute trial indices (trial_start for decision, trial_w for behavior)
        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        if len(common_trials) < 10:
            corr_real.append(np.nan)
            continue
        dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values

    # Compute Pearson correlation
    corr = pearsonr(perf_vals, dec_vals)[0]
    corr_real.append(corr)

# Prepare DataFrame for analysis
df_corr = pd.DataFrame({
    'mouse_id': mice,
    'reward_group': reward_groups,
    'real_corr': corr_real
})

# Plot correlation results using seaborn
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, group in enumerate(['R+', 'R-']):
    ax = axes[i]
    sub = df_corr[df_corr['reward_group'] == group].copy()
    correlations = sub['real_corr'].dropna().values

    if len(correlations) == 0:
        continue

    # Prepare DataFrame for seaborn
    df_plot = pd.DataFrame({
        'correlation': correlations,
        'group': [group] * len(correlations)
    })

    # Violin/strip plot for individual data points
    sns.swarmplot(data=df_plot, y='correlation', color='grey', ax=ax,edgecolor=None)

    # Plot mean and CI using seaborn pointplot
    sns.pointplot(data=df_plot, y='correlation', color=reward_palette_r[i], ax=ax, errorbar='ci',)

    # Add horizontal line at 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_ylim(-1, 1)
    ax.set_ylabel('Correlation\n(Decision value vs Performance)', fontsize=11)
    ax.set_title(f'{group}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_xlabel('')
    sns.despine(ax=ax)

plt.tight_layout()

# Statistical tests

for group in ['R+', 'R-']:
    sub = df_corr[df_corr['reward_group'] == group]
    correlations = sub['real_corr'].dropna().values
    
    if len(correlations) == 0:
        print(f"\n{group} Group: No data available")
        continue

    
    # Test if mean correlation is significantly different from 0 (one-sample test)
    if len(correlations) >= 3:
        # Wilcoxon signed-rank test (non-parametric, preferred for small N)
        stat_w, p_wilcoxon = wilcoxon(correlations, alternative='greater')

        
        print(f"  ---")
        print(f"  One-sample Wilcoxon test (H0: median ≤ 0): p = {p_wilcoxon:.4f}")

        
        # Interpretation
        if p_wilcoxon < 0.05:
            print(f"  ✓ Significant positive correlation (p < 0.05)")
        else:
            print(f"  ✗ Not significant (p ≥ 0.05)")
    else:
        print(f"  Insufficient data for statistical test (N < 3)")


plt.savefig(os.path.join(output_dir, 'correlation_behavior_decision_day0.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'correlation_behavior_decision_day0.pdf'), format='pdf', dpi=300)



# Cross-correlation analysis
# --------------------------

def crosscorr_peak_aligned(results_combined, bh_df, mice_list, min_trials=10):
    crosscorr_curves = []
    peak_lags = []
    mice_xcorr = []

    for mouse in mice_list:
        # Get decision values and behavioral curve for this mouse
        dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
        bh_mouse = bh_df[bh_df['mouse_id'] == mouse]
        # Align by trial index (trial_center for decision, trial_w for behavior)
        common_trials = np.intersect1d(dec_mouse['trial_center'], bh_mouse['trial_w'])
        if len(common_trials) < min_trials:
            continue
        dec_vals = dec_mouse.set_index('trial_center').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values
        # Remove nan
        mask = ~np.isnan(dec_vals) & ~np.isnan(perf_vals)
        if np.sum(mask) < min_trials:
            continue
        dec_vals = dec_vals[mask]
        perf_vals = perf_vals[mask]
        # Subtract mean for cross-correlation
        dec_vals -= np.nanmean(dec_vals)
        perf_vals -= np.nanmean(perf_vals)
        # Compute cross-correlation (full mode)
        xcorr = correlate(dec_vals, perf_vals, mode='full', method='auto')
        lags = np.arange(-len(perf_vals)+1, len(dec_vals))
        # Normalize
        xcorr /= (np.std(dec_vals) * np.std(perf_vals) * len(dec_vals))
        # Find peak
        peak_idx = np.argmax(xcorr)
        peak_lag = lags[peak_idx]
        # Center curve at peak
        center_idx = np.where(lags == 0)[0][0]
        shift = peak_idx - center_idx
        # Store curve and lag
        crosscorr_curves.append(np.roll(xcorr, -shift))
        peak_lags.append(peak_lag)
        mice_xcorr.append(mouse)

    # Pad curves to same length
    if not crosscorr_curves:
        return None, None, None, None
    max_len = max(len(c) for c in crosscorr_curves)
    xcorr_mat = np.full((len(crosscorr_curves), max_len), np.nan)
    for i, c in enumerate(crosscorr_curves):
        pad_left = (max_len - len(c)) // 2
        pad_right = max_len - len(c) - pad_left
        xcorr_mat[i, pad_left:pad_left+len(c)] = c

    mean_xcorr = np.nanmean(xcorr_mat, axis=0)
    sem_xcorr = np.nanstd(xcorr_mat, axis=0) / np.sqrt(np.sum(~np.isnan(xcorr_mat), axis=0))
    lags_common = np.arange(-max_len//2+1, max_len//2+1)
    return mean_xcorr, sem_xcorr, lags_common, mice_xcorr

# Separate mice by reward group
mice_rplus = results_combined[results_combined['reward_group'] == 'R+']['mouse_id'].unique()
mice_rminus = results_combined[results_combined['reward_group'] == 'R-']['mouse_id'].unique()

mean_xcorr_rplus, sem_xcorr_rplus, lags_common_rplus, mice_xcorr_rplus = crosscorr_peak_aligned(
    results_combined, bh_df, mice_rplus)
mean_xcorr_rminus, sem_xcorr_rminus, lags_common_rminus, mice_xcorr_rminus = crosscorr_peak_aligned(
    results_combined, bh_df, mice_rminus)

plt.figure(figsize=(7, 4))
if mean_xcorr_rplus is not None:
    plt.plot(lags_common_rplus, mean_xcorr_rplus, color=reward_palette[1], label='R+ mean cross-corr (centered)')
    plt.fill_between(lags_common_rplus, mean_xcorr_rplus - sem_xcorr_rplus, mean_xcorr_rplus + sem_xcorr_rplus, color=reward_palette[1], alpha=0.3)
if mean_xcorr_rminus is not None:
    plt.plot(lags_common_rminus, mean_xcorr_rminus, color=reward_palette[0], label='R- mean cross-corr (centered)')
    plt.fill_between(lags_common_rminus, mean_xcorr_rminus - sem_xcorr_rminus, mean_xcorr_rminus + sem_xcorr_rminus, color=reward_palette[0], alpha=0.3)
plt.axvline(0, color='red', linestyle='--', label='Peak aligned')
plt.xlabel('Lag (trials)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation (decision vs behavior)\nAligned at peak for each mouse')
plt.legend()
sns.despine()
plt.tight_layout()

# Plot distributions of lags to peak for both groups
plt.figure(figsize=(6, 4))
all_peak_lags = []
all_groups = []

# Compute peak lags for R+ and R- mice
def get_peak_lags(results_combined, bh_df, mice_list, min_trials=10):
    peak_lags = []
    for mouse in mice_list:
        dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
        bh_mouse = bh_df[bh_df['mouse_id'] == mouse]
        common_trials = np.intersect1d(dec_mouse['trial_center'], bh_mouse['trial_w'])
        if len(common_trials) < min_trials:
            continue
        dec_vals = dec_mouse.set_index('trial_center').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['learning_curve_w'].values
        dec_vals_smooth = gaussian_filter1d(dec_vals, sigma=2)
        perf_vals_smooth = gaussian_filter1d(perf_vals, sigma=2)
        mask = ~np.isnan(dec_vals_smooth) & ~np.isnan(perf_vals_smooth)
        if np.sum(mask) < min_trials:
            continue
        dec_vals_smooth = dec_vals_smooth[mask]
        perf_vals_smooth = perf_vals_smooth[mask]
        dec_vals_smooth -= np.nanmean(dec_vals_smooth)
        perf_vals_smooth -= np.nanmean(perf_vals_smooth)
        xcorr = correlate(dec_vals_smooth, perf_vals_smooth, mode='full', method='auto')
        lags = np.arange(-len(perf_vals_smooth)+1, len(dec_vals_smooth))
        xcorr /= (np.std(dec_vals_smooth) * np.std(perf_vals_smooth) * len(dec_vals_smooth))
        peak_idx = np.argmax(xcorr)
        peak_lag = lags[peak_idx]
        peak_lags.append(peak_lag)
    return peak_lags

peak_lags_rplus = get_peak_lags(results_combined, bh_df, mice_rplus)
peak_lags_rminus = get_peak_lags(results_combined, bh_df, mice_rminus)

# Prepare DataFrame for plotting
df_lags = pd.DataFrame({
    'peak_lag': np.concatenate([peak_lags_rplus, peak_lags_rminus]),
    'reward_group': ['R+'] * len(peak_lags_rplus) + ['R-'] * len(peak_lags_rminus)
})

sns.violinplot(data=df_lags, x='reward_group', y='peak_lag', palette=reward_palette[::-1], inner=None)
sns.stripplot(data=df_lags, x='reward_group', y='peak_lag', color='black', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', label='Zero lag')
plt.ylabel('Lag to peak (trials)')
plt.title('Distribution of lags to peak cross-correlation')
plt.tight_layout()
sns.despine()

# Save lag distribution plot
plt.savefig(os.path.join(output_dir, 'crosscorr_peak_lag_distribution.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'crosscorr_peak_lag_distribution.png'), format='png', dpi=300)

plt.savefig(os.path.join(output_dir, 'crosscorr_decision_behavior_peak_aligned.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, 'crosscorr_decision_behavior_peak_aligned.png'), format='png', dpi=300)



















# Additional analysis: Trial-by-trial classification for individual mice
def single_trial_classification_day0(vectors_mapping, vectors_learning, mice_list, seed=42):
    """
    Classify each individual trial in Day 0 learning using decoder trained on Day -2 vs +2 mapping.
    
    Sign convention: pre-learning (label 0) → negative decision values
                     post-learning (label 1) → positive decision values
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
        
        # Sanity check: Verify sign convention
        pre_mask = day_per_trial == -2
        post_mask = day_per_trial == 2
        X_pre = scaler.transform(d_mapping.values[:, pre_mask].T)
        X_post = scaler.transform(d_mapping.values[:, post_mask].T)
        mean_dec_pre = np.mean(clf.decision_function(X_pre))
        mean_dec_post = np.mean(clf.decision_function(X_post))
        
        if mean_dec_pre > mean_dec_post:
            print(f"WARNING: {mouse} has flipped decision values! Pre={mean_dec_pre:.3f}, Post={mean_dec_post:.3f}")
            sign_flip = -1
        else:
            sign_flip = 1
        
        # Test on Day 0 learning trials
        n_learning_trials = d_learning.sizes['trial']
        
        for j in range(n_learning_trials):
            X_trial = d_learning.values[:, j:j+1].T
            X_trial_scaled = scaler.transform(X_trial)
            
            decision_value = clf.decision_function(X_trial_scaled)[0] * sign_flip  # Apply sign correction
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


