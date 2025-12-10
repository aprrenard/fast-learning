import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pymc as pm 
import scipy as sp

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from nwb_wrappers import nwb_reader_functions as nwb_read
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
from scipy.signal import find_peaks, hilbert
import matplotlib.cm as cm


# ############################################
# Licking raster plot to illustrates the task.
# ############################################

# NWB files don't have the piezo lick trace, so we need to work with
# the raw data.
# Read raw trace, get timestamps of whisker trials from behavior table
# and plot licking raster around whisker stimulus.


def detect_piezo_lick_times(lick_data, ni_session_sr=5000, sigma=100, height=None, distance=None, prominence=None, width=None, do_plot=False, t_start=0, t_stop=800,):
    """
        Detect lick times from the lick data envelope using Hilbert transform.
        The envelope is extracted using the Hilbert transform and then smoothed.
        This method preserves peak amplitudes better than Gaussian filtering alone.
    Args:
        lick_data: Raw piezo lick trace
        ni_session_sr: Sampling rate of session
        sigma: Standard deviation of gaussian filter used to smooth the envelope
        height: Minimum peak height for find_peaks
        distance: Minimum distance between peaks for find_peaks
        prominence: Minimum prominence for find_peaks
        width: Minimum width for find_peaks
        do_plot: Whether to plot the detection for debugging
        t_start: Start time for plotting (seconds)
        t_stop: Stop time for plotting (seconds)

    Returns:
        lick_times: Array of detected lick times in seconds
    """

    # Get envelope using Hilbert transform
    analytic_signal = hilbert(lick_data)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope with a gaussian filter
    envelope = gaussian_filter1d(envelope, sigma=sigma)

    # Find peaks in the smoothed envelope
    peaks, _ = find_peaks(envelope, height=height, distance=distance, prominence=prominence, width=width)
    lick_times = peaks / ni_session_sr

    # Debugging: optional plotting
    if do_plot:
        ni_session_sr = int(float(ni_session_sr))
        plt.plot(lick_data[int(ni_session_sr*t_start):int(ni_session_sr*t_stop)], c='k', label="lick_data", lw=1)
        plt.plot(envelope[int(ni_session_sr*t_start):int(ni_session_sr*t_stop)], c='green', label="lick_envelope", lw=1)
        for lick_time in lick_times:
            if lick_time > t_start and lick_time < t_stop:
                plt.axvline(x=ni_session_sr*lick_time - ni_session_sr*t_start, color='red', lw=1, alpha=0.6)
        # plt.xticks(ticks=[t_start * ni_session_sr, t_stop * ni_session_sr], labels=[t_start, t_stop])
        plt.legend(loc='upper right', frameon=False)
        plt.show()

    return lick_times

# analytic_signal = hilbert(lick_trace)
# envelope = np.abs(analytic_signal)
# plt.plot(lick_trace[:1000], c='k', label="lick_data", lw=1, )
# plt.plot(envelope[:1000], c='green', label="lick_envelope", lw=1, )


mouse_id = 'GF305'

result_file = "/mnt/lsens-data/GF305/Recordings/BehaviourFiles/GF305_29112020_103331/Results.txt"

df_results = pd.read_csv(result_file, sep=r'\s+', engine='python')
# print(df_results.columns)
# print(df_results.tail())

# Test lick detection on a single trial.


# for i in range(50, 400):
#     if df_results.loc[i, 'Stim/NoStim'] == 1:
#         continue
#     trace_file = f"/mnt/lsens-data/GF305/Recordings/BehaviourFiles/GF305_29112020_103331/LickTrace{i}.bin"
#     lick_trace = np.fromfile(trace_file)[1::2]

#     sr = 100000
#     lick_times = detect_piezo_lick_times(
#         lick_trace,
#         ni_session_sr=sr,
#         sigma=200,
#         # height=0.1,
#         distance=sr*0.05,
#         prominence=1,
#         # width=sr*0.01,
#         width=None,
#         do_plot=True,
#         t_start=0,
#         t_stop=7
#     )

# session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
#     io.db_path, io.nwb_dir, experimenters=None,
#     exclude_cols = ['exclude',  'two_p_exclude'],
#     day = ['0'],
#     mouse_id = ['GF305'])
# table = make_behavior_table(nwb_list, session_list, io.db_path, cut_session=False, stop_flag_yaml=None, trial_indices_yaml=None)
# table.early_lick.sum()


sr = 100000  # Sampling rate of piezo lick trace
df_lick_raster = pd.DataFrame(columns=['trialnumber', 'trial_type', 'lick_times'])
trial_counter = 1
for _, trial in df_results.iterrows():
    if trial['EarlyLick'] == 1:
        continue
    if trial['Whisker/NoWhisker'] == 1:
        trial_type = 'whisker'
    elif trial['Auditory/NoAuditory'] == 1:
        trial_type = 'auditory'
    elif trial['Stim/NoStim'] == 0:
        trial_type = 'no_stim'

    lick_file = f"/mnt/lsens-data/GF305/Recordings/BehaviourFiles/GF305_29112020_103331/LickTrace{int(trial['trialnumber'])}.bin"
    lick_trace = np.fromfile(lick_file)[1::2]
    lick_times = detect_piezo_lick_times(
        lick_trace,
        ni_session_sr=sr,
        sigma=200,
        height=0.04,
        # prominence=1,
        distance=sr*0.05,
        width=None,
        do_plot=False,
    )

    # # Visualize lick trace for trials with a lick before stimulus onset (t < 2 sec)
    # stim_onset_time = 2  # Stimulus onset time in seconds
    # if np.any(lick_times < stim_onset_time):
    #     plt.figure(figsize=(10, 4))

    #     # Compute envelope for plotting
    #     from scipy.signal import hilbert
    #     analytic_signal = hilbert(lick_trace)
    #     envelope = np.abs(analytic_signal)
    #     envelope_smooth = gaussian_filter1d(envelope, sigma=200)

    #     # Time vector
    #     time_vec = np.arange(len(lick_trace)) / sr

    #     # Plot raw trace
    #     plt.plot(time_vec, lick_trace, color='black', lw=0.5, alpha=0.7, label='Raw trace')
    #     # Plot envelope
    #     plt.plot(time_vec, envelope_smooth, color='green', lw=1.5, label='Envelope (smoothed)')

    #     # Mark stimulus onset
    #     plt.axvline(stim_onset_time, color='blue', linestyle='--', lw=2, label='Stimulus onset')

    #     # Mark detected lick times
    #     for lt in lick_times:
    #         plt.axvline(lt, color='red', linestyle='-', lw=1, alpha=0.7)

    #     # Add custom legend entry for detected licks
    #     plt.axvline(np.nan, color='red', linestyle='-', lw=1, alpha=0.7, label='Detected licks')

    #     plt.title(f'Trial {int(trial.trialnumber)}: Lick before stimulus onset', fontsize=12)
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Piezo signal')
    #     plt.legend(loc='upper right', frameon=False)
    #     plt.tight_layout()
    #     plt.show()

    # Use trial_counter to avoid blanks due to early licks
    df_lick_raster = pd.concat([
        df_lick_raster,
        pd.DataFrame({
            'trialnumber': [trial_counter],
            'trial_type': [trial_type],
            'lick_times': [lick_times.tolist()]
        })
    ], ignore_index=True)
    trial_counter += 1


# Remove mapping trials.
df_lick_raster = df_lick_raster[df_lick_raster.trialnumber <= 320]


# Plot licking raster around stimulus events for all trials (non-sorted)
interval_start = -1.0  # seconds
interval_stop = 6.0    # seconds
# Adjust figure size to fit as a panel in an A4 paper figure (A4: 210mm x 297mm)
# Typical panel width: ~80-100mm, height: ~60-120mm
fig_width_mm = 45
fig_height_mm = 90
fig_width_in = fig_width_mm / 25.4
fig_height_in = fig_height_mm / 25.4

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
ax = fig.add_subplot(111)

colors = {'whisker': stim_palette[1], 'auditory': stim_palette[0], 'no_stim': stim_palette[2]}

for i, row in df_lick_raster.iterrows():
    trial_type = row['trial_type']
    lick_times = np.array(row['lick_times']) -2  # GF has 2 sec extra baseline
    # lick_window = lick_times[(lick_times >= interval_start) & (lick_times <= interval_stop)]
    ax.scatter(lick_times, np.full_like(lick_times, i + 0.5), color=colors.get(trial_type, 'grey'),
               s=1, alpha=1, marker='o', linewidths=0)
ax.axvspan(0, 1, color='lightgrey', alpha=0.5, zorder=0)
ax.set_xlabel('Lick time from stim onset (secs)')
ax.set_ylabel('Trial')
ax.set_xlim([-1, 4])
# ax.set_xticklabels(range(-1, 5))
sns.despine()


# plt.show()  # Not needed in Interactive Window - figure displays automatically

# Save non-sorted plot
save_path = r"/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/illustrations/lick_raster"
save_path = io.adjust_path_to_host(save_path)
plt.savefig(os.path.join(save_path, 'lick_raster_GF305_nosorting.svg'), format='svg', dpi=300)
plt.close()

# Plot licking raster with sorted trials (no_stim first, then whisker, then auditory)
trial_order = ['no_stim', 'whisker', 'auditory']
df_sorted = pd.concat([df_lick_raster[df_lick_raster['trial_type'] == t] for t in trial_order], ignore_index=True)

fig = plt.figure(figsize=(3, 8))
ax = fig.add_subplot(111)

for i, row in df_sorted.iterrows():
    trial_type = row['trial_type']
    lick_times = np.array(row['lick_times'])
    lick_window = lick_times[(lick_times >= interval_start) & (lick_times <= interval_stop)]
    ax.scatter(lick_window, np.full_like(lick_window, i + 0.5), color=colors.get(trial_type, 'grey'),
               s=1.5, alpha=1, marker='|', linewidths=1.5)
    ax.axvspan(2, 3, color='lightgrey', alpha=0.5, zorder=0)
    
ax.set_xlabel('Lick time from stim onset (secs)')
ax.set_ylabel('Trial')
ax.set_xlim([interval_start, interval_stop])
ax.set_xticklabels(range(-1, 5))
sns.despine()

# Save sorted plot
plt.savefig(os.path.join(save_path, 'lick_raster_GF305_sorted.svg'), format='svg', dpi=300)
plt.close()
