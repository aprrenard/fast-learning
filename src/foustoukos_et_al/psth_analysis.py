"""This script generates PSTH numpy arrays from lists of NWB files.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# PSTH's day -1 VS day +1 full population.
# ----------------------------------------

read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_rew_GF.npy')
traces_rew = np.load(read_path, allow_pickle=True)
# read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
#              'data_processed\\UM_rewarded_metadata.npy')
# metadata = np.load(read_path, allow_pickle=True).item()

read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
             'data_processed\\traces_non_motivated_trials_non_rew_GF.npy')
traces_non_rew = np.load(read_path, allow_pickle=True)
# read_path = ('\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\'
#              'data_processed\\UM_non_rewarded_metadata.npy')
# metadata_non_rew = np.load(read_path, allow_pickle=True).item()

# traces_non_rew = traces_non_rew[[1,3,4,5,6,7,8,9]]

# Substract baseline.
traces_rew = traces_rew - np.nanmean(traces_rew[:,:,:,:,:,:30], axis=5, keepdims=True)
traces_non_rew = traces_non_rew - np.nanmean(traces_non_rew[:,:,:,:,:,:30], axis=5, keepdims=True)

# Plot responses across cells.
day_m1 = np.nanmean(traces_rew[:1], axis=(4))
day_m1 = np.nanmean(day_m1, axis=(0,2,3))[1]
day_1 = np.nanmean(traces_rew[:1], axis=(4))
day_1 = np.nanmean(day_1, axis=(0,2,3))[3]

day_m1_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_m1_non_rew = np.nanmean(day_m1_non_rew, axis=(0,2,3))[1]
day_1_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_1_non_rew = np.nanmean(day_1_non_rew, axis=(0,2,3))[3]

f, axes = plt.subplots(1,2, sharey=True)
axes[0].plot(day_m1)
axes[0].plot(day_1)

axes[1].plot(day_m1_non_rew)
axes[1].plot(day_1_non_rew)

# Projection neurons.


# Plot responses across cells.
day_m1_S2 = np.nanmean(traces_rew, axis=(4))
day_m1_S2 = np.nanmean(day_m1_S2, axis=(0,3))[1,2]
day_1_S2 = np.nanmean(traces_rew, axis=(4))
day_1_S2 = np.nanmean(day_1_S2, axis=(0,3))[3,2]

day_m1_S2_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_m1_S2_non_rew = np.nanmean(day_m1_S2_non_rew, axis=(0,3))[1,2]
day_1_S2_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_1_S2_non_rew = np.nanmean(day_1_S2_non_rew, axis=(0,3))[3,2]

# Plot responses across cells.
day_m1_M1 = np.nanmean(traces_rew, axis=(4))
day_m1_M1 = np.nanmean(day_m1_M1, axis=(0,3))[1,1]
day_1_M1 = np.nanmean(traces_rew, axis=(4))
day_1_M1 = np.nanmean(day_1_M1, axis=(0,3))[3,1]

day_m1_M1_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_m1_M1_non_rew = np.nanmean(day_m1_M1_non_rew, axis=(0,3))[1,1]
day_1_M1_non_rew = np.nanmean(traces_non_rew, axis=(4))
day_1_M1_non_rew = np.nanmean(day_1_M1_non_rew, axis=(0,3))[3,1]

f, axes = plt.subplots(2,2, sharey=True)

axes[0,0].plot(day_m1_M1, color='k')
axes[0,0].plot(day_1_M1, color='b')

axes[0,1].plot(day_m1_S2, color='k')
axes[0,1].plot(day_1_S2, color='r')

axes[1,0].plot(day_m1_M1_non_rew, color='k')
axes[1,0].plot(day_1_M1_non_rew, color='b')

axes[1,1].plot(day_m1_S2_non_rew, color='k')
axes[1,1].plot(day_1_S2_non_rew, color='r')



# Coun  t cells.

np.sum(np.nansum(~np.isnan(traces_rew[:,0,:,:,0,0]), axis=(1,2)))
np.sum(np.nansum(~np.isnan(traces_non_rew[:,0,:,:,0,0]), axis=(1,2)))

np.nansum(~np.isnan(traces_rew[:,0,2,:,0,0]), axis=(1))
np.nansum(~np.isnan(traces_non_rew[:,0,1,:,0,0]), axis=(1))

# Count trials per session.
np.nansum(~np.isnan(traces_rew[:,:,0,0,:,0]), axis=(2))
np.nansum(~np.isnan(traces_non_rew[:,:,0,0,:,0]), axis=(2))

traces_rew.shape