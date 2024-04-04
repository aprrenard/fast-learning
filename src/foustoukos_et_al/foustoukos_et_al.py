import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

sys.path.append('C:\\Users\\aprenard\\recherches\\repos\\NWB_analysis')
from analysis.psth_analysis import make_events_aligned_data_table, make_events_aligned_array
from nwb_wrappers import nwb_reader_functions as nwb_read
import nwb_utils.server_path as server_path


def read_excel_db(excel_path):
    database = pd.read_excel(excel_path, converters={'session_day': str})

    # Remove empty lines.
    database = database.loc[~database.isna().all(axis=1)]

    # Change yes/no columns to booleans.
    database = database.replace('yes', True)
    database = database.replace('no', False)
    database = database.astype({'2P_calcium_imaging': bool,
                                'optogenetic': bool,'pharmacology': bool})

    return database


# Read excel database.
excel_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\session_metadata.xlsx'
db = read_excel_db(excel_path)
db.session_day
mice = db.loc[db['2P_calcium_imaging']==True, 'subject_id'].unique()

rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (1,3)
epoch_name = 'unmotivated'

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = []
for mouse_id in mice:
    nwb_list.extend([nwb for nwb in os.listdir(nwb_path) if mouse_id in nwb])
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])

behavior_types = ['auditory', 'whisker']
days = [-2, -1, 0, 1, 2]
nwb_list = [nwb for nwb in nwb_list if nwb_read.get_bhv_type_and_training_day_index(nwb)[1] in days]
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
table = make_events_aligned_array(nwb_list, rrs_keys, time_range, trial_selection, epoch_name)


# PSTH over cells for different days and cell types.
palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])
temp = table.groupby(['mouse_id', 'session_id', 'roi', 'time', 'cell_type', 'behavior_type', 'behavior_day'],
                     as_index=False).agg(np.nanmean)
temp = temp.astype({'roi': str})

temp = temp.loc[~((temp.mouse_id=='AR131') & (temp.cell_type=='wS2'))]

for mouse_id in mice:
    d = temp.loc[temp.mouse_id==mouse_id]
    sns.relplot(data=d, x='time', y='activity', hue='cell_type', hue_order=['na', 'wM1', 'wS2'], n_boot=100, kind='line',
                col='behavior_day', palette=palette)
    if mouse_id == 'AR131':
        plt.suptitle(f'{mouse_id} R-')
    else:
        plt.suptitle(f'{mouse_id} R+')
    plt.subplots_adjust(top=0.85)



# For GF


rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (0,5)
epoch_name = 'unmotivated'

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = os.listdir(nwb_path)
nwb_list = [nwb for nwb in nwb_list if 'GF' in nwb]
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])
nwb_list = [nwb for nwb in nwb_list
            if 'twophoton' in nwb_read.get_session_metadata(nwb)['session_type']]

nwb_list_rew = [nwb for nwb in nwb_list
                if nwb_read.get_session_metadata(nwb)['wh_reward']==1]
nwb_list_non_rew = [nwb for nwb in nwb_list
                if nwb_read.get_session_metadata(nwb)['wh_reward']==0]

trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
table_rew = make_events_aligned_data_table(nwb_list_rew, rrs_keys, time_range, trial_selection, epoch_name)
table_rew['wh_reward'] = 'R+'
table_non_rew = make_events_aligned_data_table(nwb_list_non_rew, rrs_keys, time_range, trial_selection, epoch_name)
table_non_rew['wh_reward'] = 'R-'
del table
import gc
gc.collect()
del table_rew



# PSTH over cells for different days and cell types.
palette = sns.color_palette(['#212120', '#3351ff', '#c959af'])
temp = table_rew.groupby(['mouse_id', 'session_id', 'roi', 'time', 'cell_type', 'behavior_type', 'behavior_day', 'wh_reward'],
                     as_index=False).agg(np.nanmean)
temp = table_rew.astype({'roi': str})

sns.relplot(data=temp, x='time', y='activity', hue='cell_type', hue_order=['na', 'wM1', 'wS2'], n_boot=100, kind='line',
            col='behavior_day', palette=palette)
if table['wh_reward'].unique()[0] == 'R+':
    plt.suptitle(f'{mouse_id} R+')
else:
    plt.suptitle(f'{mouse_id} R-')
plt.subplots_adjust(top=0.85)






    
    

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

d = temp.loc[(temp.cell_type=='wM1') & (temp.behavior_day.isin([-1, 1]))]
sns.lineplot(x='time', y='activity', data=d, hue='cell_type', style='behavior_day', n_boot=100)



for cell in table.loc[table.cell_type=='wM1', 'roi'].unique():
    d = table.loc[(table.roi==cell) & (table.behavior_day.isin([-1,+1]))]
    plt.figure()
    sns.lineplot(x='time', y='activity', data=d, style='behavior_day', n_boot=100)
    plt.title(cell)

event = 0
cell = 166
bday = 1
for event in range(50):
    d = table.loc[(table.roi==cell) & (table.event==event) & (table.behavior_day==bday)]
    sns.lineplot(x='time', y='activity', data=d)








path = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\AR127\\suite2p\\plane0\\F_cor.npy"
F_cor = np.load(path, allow_pickle=True)
plt.plot(F_cor[166])

sep = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\data\\AR127\\suite2p\\plane0\\prepared.npz"
sep = np.load(sep, allow_pickle=True)

# fissa = sep['result']
fissa = sep['raw']


cell = 166
tif=0
fissa[cell,tif][0]

for i in range(100):
    plt.plot(np.arange(0+500*i,500*(i+1)), fissa[cell,i][0])



# Extract and reshape corrected traces to (ncells, nt).
ncells, ntifs = fissa.shape
F_cor = []
for icell in range(ncells):
    tmp = []
    for itif in range(ntifs):
        tmp.append(fissa[icell,itif][0])
    F_cor.append(np.concatenate(tmp))


a = np.stack(F_cor, axis=0)
a.shape

plt.plot(F_cor[166])
plt.plot(np.concatenate(tmp))
len(F_cor)






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


# Using numpy rather than pandas.
# ###############################

# Read excel database.
excel_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\mice_info\\session_metadata.xlsx'
db = read_excel_db(excel_path)
mice = db.loc[db['2P_calcium_imaging']==True, 'subject_id'].unique()

rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
time_range = (0,5)
epoch_name = 'unmotivated'

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = []
for mouse_id in mice:
    nwb_list.extend([nwb for nwb in os.listdir(nwb_path) if mouse_id in nwb])
nwb_list = sorted([os.path.join(nwb_path,nwb) for nwb in nwb_list])

behavior_types = ['auditory', 'whisker']
days = [-2, -1, 0, 1, 2]
behavior_days = list(itertools.product(behavior_types, days))
nwb_list = [nwb for nwb in nwb_list if nwb_read.get_bhv_type_and_training_day_index(nwb) in behavior_days]
trial_selection = {'whisker_stim': [1], 'lick_flag':[0]}
traces, metadata = make_events_aligned_array(nwb_list, rrs_keys, time_range, trial_selection, epoch_name)

# Save activity dict.
save_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials.npy')
np.save(save_path, {'data': traces, 'metadata':metadata})


# PSTH's day -1 VS day +1 full population.
# ----------------------------------------

read_path = ('C:\\Users\\aprenard\\recherches\\data\\traces_non_motivated_trials.npy')
traces = np.load(read_path, allow_pickle=True)

# Just look at GF data.
traces = traces[3:]

# Substract baseline.
traces = traces - np.nanmean(traces[:,:,:,:,:,:30], axis=5, keepdims=True)

pop_resp_all_cells = np.nanmean(traces, axis=(2,3,4))

day_m1 = np.nanmean(pop_resp_all_cells[:,1], axis=(0))
day_1 = np.nanmean(pop_resp_all_cells[:,3], axis=(0))
plt.plot(day_m1)
plt.plot(day_1)
