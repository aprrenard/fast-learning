import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import server_path
from analysis.psth_analysis import return_events_aligned_data_table


nwb_list =  [
            # 'AR103_20230823_102029.nwb',
            #  'AR103_20230826_173720.nwb',
            #  'AR103_20230825_190303.nwb',
            #  'AR103_20230824_100910.nwb',
            #  'AR103_20230827_180738.nwb',
             'GF333_21012021_125450.nwb'
             ]


rrs_keys = ['ophys', 'fluorescence_all_cells', 'dcnv']
time_range = (1,5)
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
epoch_name = 'unmotivated'

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]

table = return_events_aligned_data_table(nwb_list, rrs_keys, time_range, trial_selection, epoch_name)

temp = table.groupby(['mouse_id', 'session_id', 'roi', 'time', 'cell_type', 'behavior_type', 'behavior_day'], as_index=False).agg(np.nanmean)
temp = temp.astype({'roi': str})
f = plt.figure()
sns.lineplot(data=temp.loc[temp.roi.isin(['10','38','44'])], x='time', y='activity', hue='roi', style='session_id', n_boot=100)

temp = table.groupby(['mouse_id', 'session_id', 'roi', 'time', 'behavior_type', 'behavior_day'], as_index=False).agg(np.nanmean)
temp = temp.astype({'roi': str})
f = plt.figure()
sns.lineplot(data=temp.loc[temp.roi.isin(['10','38','44'])], x='time', y='activity', hue='roi', style='session_id', n_boot=100)

temp.roi.unique()

temp.loc[temp.roi==15, 'trace'].plot()

table.loc[table.roi==0].plot(y='activity', use_index=False)
table.loc[table.roi==0].plot(y='activity', use_index=True)

table.roi.unique()

table.iloc[:190]

a = np.concatenate((np.ones(3)*0, np.ones(3)*1, np.ones(3)*2)).reshape((3,3))

a.flatten()

for icell in temp.loc[temp.cell_type=='wM1', 'roi']:
    temp.loc[temp.cell_type=='wM1', 'roi'].unique()


data = table.loc[table.roi==10]

plt.figure()
for ievent in range(50):
    plt.plot(data.loc[data.event==ievent, 'activity'].to_numpy()+ievent*10)
plt.axvline(90, color='k')
plt.axvline(271, color='k')








# ###################



nwb_list = [
            # 'AR103_20230823_102029.nwb',
            #  'AR103_20230826_173720.nwb',
            #  'AR103_20230825_190303.nwb',
            #  'AR103_20230824_100910.nwb',
            #  'AR103_20230827_180738.nwb',
             'GF333_21012021_125450.nwb',
             'GF333_26012021_142304.nwb'
             ]

rrs_keys = ['ophys', 'fluorescence_all_cells']
nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]

F = []
time_stamp = []
Fneu = []
F0 = []
dff = []
dcnv = []
F_fissa = []
dff_fissa = []
events = []

for nwb_file in nwb_list:
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'F')
    F.append(data)
    ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, rrs_keys + 'F')
    time_stamp.append(ts)
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'Fneu')
    Fneu.append(data)
    # data = nwb_read.get_roi_response_serie_data(nwb_file, 'F0')
    # F0.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'dff')
    dff.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'dcnv')
    dcnv.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'F_fissa')
    F_fissa.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, rrs_keys + 'dff_fissa')
    dff_fissa.append(data)
    ev = nwb_read.get_trial_timestamps_from_table(nwb_file, {'whisker_stim': [1], 'lick_flag':[0]})[0]
    events.append(ev)

F = np.concatenate(F, axis=1)
time_stamp = np.concatenate(time_stamp)
Fneu = np.concatenate(Fneu, axis=1)
# F0 = np.concatenate(F0, axis=1)
dff = np.concatenate(dff, axis=1)
dcnv = np.concatenate(dcnv, axis=1)
F_fissa = np.concatenate(F_fissa, axis=1)
dff_fissa = np.concatenate(dff_fissa, axis=1)
events = [e for e in ev for ev in events]








10, 44, 38

icell = 44

f, axes = plt.subplots(4,1, sharex=True)

axes[0].plot(F[icell])
axes[0].plot(Fneu[icell])
# axes[0].plot(F0[icell])
axes[0].axhline(1, linestyle='--', color='k')

axes[1].plot(F[icell]-0.7*Fneu[icell])
axes[1].plot(Fneu[icell])
# axes[1].plot(F0[icell])
axes[1].axhline(1, linestyle='--', color='k')

axes[2].plot(dff[icell])

axes[3].plot(dcnv[icell])

plt.suptitle(icell)



icell = 1

f, axes = plt.subplots(2,1, sharex=True)

axes[0].plot(F[icell])
axes[0].plot(Fneu[icell])
axes[1].plot(F_fissa[icell])

