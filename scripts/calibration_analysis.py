"""Analysis script for coil and whisker displacement calibration.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from src.utils import ci_timeseries_bootstrap

sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)


# Saving data as np arrays.
# #########################

# Sensor file.
# ------------

FOLDER_READ = os.path.join('K:','calibration')
FOLDER_WRITE = 'C:\\Users\\aprenard\\recherches\\fast-learning\\calibration\\data'
FILE_NAME = '220731_calibration_sensor_fakemouse.npy'
NTRIALS = 5
TRIAL_LENGTH = 400000

calibration_sensor_files = [
    '220731_calibration_sensor_fakemouse_blank_300um.m',
    '220731_calibration_sensor_fakemouse_blank_350um.m',
    # '220731_calibration_sensor_fakemouse_blank_375um.m',
    '220731_calibration_sensor_fakemouse_blank_400um.m',
    # '220731_calibration_sensor_fakemouse_blank_425um.m',
    '220731_calibration_sensor_fakemouse_blank_450um.m',
    # '220731_calibration_sensor_fakemouse_blank_475um.m',
    '220731_calibration_sensor_fakemouse_blank_500um.m',
    # '220731_calibration_sensor_fakemouse_blank_525um.m',
    '220731_calibration_sensor_fakemouse_blank_550um.m',
    '220731_calibration_sensor_fakemouse_blank_600um.m',
    # '220819_calibration_sensor_fakemouse_blank_650um.m',
    # '220819_calibration_sensor_fakemouse_blank_700um.m',
    # '220819_calibration_sensor_fakemouse_blank_750um.m',
    # '220819_calibration_sensor_fakemouse_blank_800um.m',
]
calibration_sensor_files = [os.path.join(FOLDER_READ,f) for f in calibration_sensor_files]
data = np.zeros((len(calibration_sensor_files),NTRIALS,TRIAL_LENGTH))
for i, ifile in enumerate(calibration_sensor_files):
    data[i] = loadmat(ifile)['data'].astype('float32')[:NTRIALS]
path_write = os.path.join(FOLDER_WRITE,FILE_NAME)
np.save(path_write,data)


# Coil calibration and impulse files.
# -----------------------------------

FOLDER_READ = os.path.join('K:','calibration')
FOLDER_WRITE = 'C:\\Users\\aprenard\\recherches\\fast-learning\\calibration\\data'
NTRIALS = 5
TRIAL_LENGTH = 400000
FILE_NAME_MAGNETIC = '220731_calibration_coil_fakemouse.npy'
FILE_NAME_VOLTAGE = '220731_impulse_coil_fakemouse.npy'

calibration_coil_files = [
    '220731_calibration_coil_fakemouse_biphasic_square_1ms_35mT.m',
    '220731_calibration_coil_fakemouse_biphasic_square_1ms_30mT.m',

    '220731_calibration_coil_fakemouse_biphasic_hann_1.6ms_35mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_1.6ms_30mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_1.6ms_25mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_1.6ms_20mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_1.6ms_15mT.m',

    '220731_calibration_coil_fakemouse_biphasic_hann_3ms_35mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_3ms_30mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_3ms_25mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_3ms_20mT.m',
    '220731_calibration_coil_fakemouse_biphasic_hann_3ms_15mT.m',

    # '220819_calibration_coil_fakemouse_monophasic_hann_1ms_35mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_1ms_30mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_1ms_25mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_1ms_20mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_1ms_15mT.m',

    # '220819_calibration_coil_fakemouse_monophasic_hann_3ms_35mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_3ms_30mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_3ms_25mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_3ms_20mT.m',
    # '220819_calibration_coil_fakemouse_monophasic_hann_3ms_15mT.m',
]
calibration_coil_files = [os.path.join(FOLDER_READ,f) for f in calibration_coil_files]

# Magntic strength response.
data = np.zeros((len(calibration_coil_files),NTRIALS,TRIAL_LENGTH)) * np.nan
for i, ifile in enumerate(calibration_coil_files):
    data[i] = loadmat(ifile)['data'].astype('float32')[:NTRIALS]

path_write = os.path.join(FOLDER_WRITE,FILE_NAME_MAGNETIC)
np.save(path_write,data)

# Voltage impulse.
data = np.zeros((len(calibration_coil_files),TRIAL_LENGTH))
for i, ifile in enumerate(calibration_coil_files):
    data[i] = loadmat(ifile)['stim'].astype('float32')

path_write = os.path.join(FOLDER_WRITE,FILE_NAME_VOLTAGE)
np.save(path_write,data)


# Displacement file.
# ------------------

FOLDER_READ = os.path.join('K:','calibration')
FOLDER_WRITE = 'C:\\Users\\aprenard\\recherches\\fast-learning\\calibration\\data'
NTRIALS = 5
TRIAL_LENGTH = 400000
FILE_NAME = '220731_calibration_displacement_fakemouse.npy'

displacement_files = [
    '220731_calibration_displacement_fakemouse_biphasic_square_1ms_35mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_square_1ms_30mT.m',

    '220731_calibration_displacement_fakemouse_biphasic_hann_1.6ms_35mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_1.6ms_30mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_1.6ms_25mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_1.6ms_20mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_1.6ms_15mT.m',

    '220731_calibration_displacement_fakemouse_biphasic_hann_3ms_35mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_3ms_30mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_3ms_25mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_3ms_20mT.m',
    '220731_calibration_displacement_fakemouse_biphasic_hann_3ms_15mT.m',

    # '220819_calibration_displacement_fakemouse_monophasic_hann_1ms_35mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_1ms_30mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_1ms_25mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_1ms_20mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_1ms_15mT.m',

    # '220819_calibration_displacement_fakemouse_monophasic_hann_3ms_35mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_3ms_30mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_3ms_25mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_3ms_20mT.m',
    # '220819_calibration_displacement_fakemouse_monophasic_hann_3ms_15mT.m',
]
displacement_files = [os.path.join(FOLDER_READ,f) for f in displacement_files]

data = np.zeros((len(displacement_files),NTRIALS,TRIAL_LENGTH))
for i, ifile in enumerate(displacement_files):
    data[i] = loadmat(ifile)['data'].astype('float32')[:NTRIALS]

path_write = os.path.join(FOLDER_WRITE,FILE_NAME)
np.save(path_write,data)


# Displacement analysis.
# ######################

# Read data.
# ----------

FOLDER_READ = 'C:\\Users\\aprenard\\recherches\\fast-learning\\calibration\\data'

FILE_NAME = '220819_calibration_displacement_fakemouse.npy'
path = os.path.join(FOLDER_READ,FILE_NAME)
data_disp = np.load(path)
data_disp = data_disp[:,:,180000:240000]

FILE_NAME = '220819_calibration_coil_fakemouse.npy'
path = os.path.join(FOLDER_READ,FILE_NAME)
data_coil = np.load(path)
data_coil = data_coil[:,:,180000:240000]
# Ajusting scale.
if '220819' in FILE_NAME:
    data_coil[:12] *= 100
    data_coil[12:] *= 1000
else:
    data_coil *= 100

FILE_NAME = '220819_impulse_coil_fakemouse.npy'
path = os.path.join(FOLDER_READ,FILE_NAME)
data_impulse = np.load(path)
data_impulse = data_impulse[:,180000:240000]

FILE_NAME = '220819_calibration_sensor_fakemouse.npy'
path = os.path.join(FOLDER_READ,FILE_NAME)
data_sensor = np.load(path)
data_sensor = data_sensor[:,:,180000:240000]


# Check linearity and sensitivity of the sensor.
# ----------------------------------------------


DISTANCE = [300,350,400,450,500,550,600,650,700,750,800]
# DISTANCE = [300,350,400,450,500,550,600]
NTRIALS = 5

trials = np.repeat(np.arange(NTRIALS).reshape(1,NTRIALS),len(DISTANCE),axis=0).flatten()
distance = np.repeat(DISTANCE,NTRIALS)
voltage = data_sensor[:,:,20000:40000].mean(2).flatten()

df_sensor = pd.DataFrame(np.stack([trials,distance,voltage]).T,
                         columns=['trial','distance','voltage'],)

X = df_sensor['distance'].to_numpy().reshape(-1,1)
y = df_sensor['voltage'].to_numpy().reshape(-1,1)
reg = LinearRegression().fit(X,y)
coef = reg.coef_[0,0]
intercept = reg.intercept_[0]

plt.figure()
sns.regplot(data=df_sensor,x='distance',y='voltage',marker='+')
plt.xticks(DISTANCE)
plt.title('Coef {:.10} intercept {:.10}'.format(coef,intercept))
sns.despine()
plt.show()


# Convert angular displacement, speed and acceleration.
# -----------------------------------------------------

# Note that the higher the voltage the smaller the distance from the sensor,
# so the whisker is moving up (in the dorsal direction) as distance decreases
# since the sensor is positioned above the whisker.

DISTANCE_FROM_BASE = 2300  # Particle position from whisker base in µm.
BASELINE_LENGTH = 20000  # Baseline length for normalization in samples.

# Flip sign so positive angle means whisker moves up.
data_disp = (data_disp) / coef * -1
data_disp = np.rad2deg(np.arctan(data_disp/DISTANCE_FROM_BASE))
data_disp = data_disp - np.mean(data_disp[:,:,:BASELINE_LENGTH],
                                axis=2,keepdims=True)
# Smooth out sensor noise with Savitzky-Golay.
data_disp_speed = savgol_filter(data_disp,window_length=31,polyorder=2,
                                deriv=1,delta=1/100000,axis=2)
data_disp_acc = savgol_filter(data_disp,window_length=81,polyorder=2,
                              deriv=2,delta=1/100000,axis=2)
data_disp_sg = savgol_filter(data_disp,window_length=31,polyorder=2,axis=2)


# Illustrate the different signal types.

sns.set_theme(context='poster', style='ticks', palette='deep')

stimulus = 0
time = np.linspace(-20,40,6000)
start = 18000
end = 24000

f, axes = plt.subplots(3,1, sharex=True)
axes[0].plot(time,data_impulse[stimulus,start:end])
m = data_coil[stimulus,:,start:end].mean(0)
axes[1].plot(time,m)
m = data_disp[stimulus,:,start:end].mean(0)
axes[2].plot(time,m)
sns.despine()

# Illustrate Savitzky-Golay smooting.
time = np.linspace(-5,10,1500)
start = 19500
end = 21000
plt.plot(time, data_disp[0,0,start:end])
plt.plot(time, data_disp_sg[0,0,start:end])
plt.ylabel('Voltage (V)')
plt.xlabel('Time (ms)')
sns.despine()


# Plot voltage pulses, magnetic pulses and resulting displacement.

sns.set_theme(context='poster', style='ticks', palette='Spectral')

stimulus = 0
time = np.linspace(-5,10,1500)
start = 19500
end = 21000

f, axes = plt.subplots(5,3, sharex=True)
for iax in axes[:,1]: iax.axhline(35, linestyle='--',alpha=0.5,color='grey')

for istim in range(2):
    axes[0,0].plot(time,data_impulse[istim,start:end])

for istim in range(2,7):
    axes[1,0].plot(time,data_impulse[istim,start:end])

for istim in range(7,12):
    axes[2,0].plot(time,data_impulse[istim,start:end])

for istim in range(12,17):
    axes[3,0].plot(time,data_impulse[istim,start:end])

for istim in range(17,22):
    axes[4,0].plot(time,data_impulse[istim,start:end])

for istim in range(2):
    axes[0,1].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in range(2,7):
    axes[1,1].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in range(7,12):
    axes[2,1].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in range(12,17):
    axes[3,1].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in range(17,22):
    axes[4,1].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in range(2):
    axes[0,2].plot(time,data_disp[istim,:,start:end].mean(0))

for istim in range(2,7):
    axes[1,2].plot(time,data_disp[istim,:,start:end].mean(0))

for istim in range(7,12):
    axes[2,2].plot(time,data_disp[istim,:,start:end].mean(0))

for istim in range(12,17):
    axes[3,2].plot(time,data_disp[istim,:,start:end].mean(0))

for istim in range(17,22):
    axes[4,2].plot(time,data_disp[istim,:,start:end].mean(0))
    

for iax in axes[:,1]: iax.set_ylim([-5,40])
for iax in axes[:,1]: iax.set_yticks([0,10,20,30,40])
for iax in axes[:,2]: iax.set_ylim([-8,8])
for iax in axes[:,2]: iax.set_yticks(range(-8,9,4))
axes[2,1].set_ylabel('Mag. strength (mT)')
axes[2,2].set_ylabel('Deflection (°)')
axes[2,0].set_ylabel('Voltage (V)')

axes[0,0].set_ylim([-4,4])
axes[0,0].set_yticks([-4,0,4])
axes[1,0].set_ylim([-8,5])
axes[1,0].set_yticks([-8,0,4])
axes[2,0].set_ylim([-3,3])
axes[2,0].set_yticks([-3,0,3])
axes[3,0].set_ylim([-.1,5])
axes[3,0].set_yticks([0,5])
axes[4,0].set_ylim([-.1,2])
axes[4,0].set_yticks([0,2])

# for iax in axes.flatten(): iax.set_xticks([0,10,20,30,40,50])
for iax in axes.flatten(): iax.set_xticks(range(-5,11))
sns.despine()
plt.tight_layout()


# Matching displacement amplitude.

stimulus = 0
time = np.linspace(-5,10,1500)
start = 19500
end = 21000
stimuli = [0,3,8]

f, axes = plt.subplots(3,2, sharex=True)
# for iax in axes[:,0]: iax.axhline(30, linestyle='--',alpha=0.5,color='grey')

for istim in stimuli:
    axes[1,0].plot(time,data_coil[istim,:,start:end].mean(0))

for istim in stimuli:
    axes[0,1].plot(time,data_disp[istim,:,start:end].mean(0))
for istim in stimuli:
    axes[1,1].plot(time,data_disp_speed[istim,:,start:end].mean(0)/1000)
for istim in stimuli:
    axes[2,1].plot(time,data_disp_acc[istim,:,start:end].mean(0))

axes[0,1].set_yticks(range(-6,7,2))
axes[0,1].set_ylim(-6,6)
axes[1,1].set_yticks(range(-16,17,4))
axes[1,1].set_ylim(-16,16)
axes[2,1].set_ylim(-70000000,70000000)
for iax in axes[:,0]: iax.set_yticks([0,10,20,30,40])
for iax in axes.flatten(): iax.set_xticks(range(-5,11))

axes[1,0].set_ylabel('Mag. strength (mT)')
axes[0,1].set_ylabel('Deflection (°)')
axes[1,1].set_ylabel('Speed (°/s)')
axes[2,1].set_ylabel('Acceleration (°/s^2)')
sns.despine()   



# Comparing artefacts.

stimulus = 0
time = np.linspace(-5,10,10500)
start = 19500
end = 30000
stimuli = [0,3,8]

f, axes = plt.subplots(3,2, sharex=True)
# for iax in axes[:,0]: iax.axhline(30, linestyle='--',alpha=0.5,color='grey')

axes[0,0].plot(time,data_coil[0,:,start:end].mean(0))
axes[0,1].plot(time,data_disp[0,:,start:end].mean(0))

axes[1,1].plot(time,data_disp_speed[istim,:,start:end].mean(0)/1000)
axes[2,1].plot(time,data_disp_acc[istim,:,start:end].mean(0))

axes[0,1].set_yticks(range(-6,7,2))
axes[0,1].set_ylim(-6,6)
axes[1,1].set_yticks(range(-16,17,4))
axes[1,1].set_ylim(-16,16)
axes[2,1].set_ylim(-70000000,70000000)
for iax in axes[:,0]: iax.set_yticks([0,10,20,30,40])
for iax in axes.flatten(): iax.set_xticks(range(-5,11))

axes[1,0].set_ylabel('Mag. strength (mT)')
axes[0,1].set_ylabel('Deflection (°)')
axes[1,1].set_ylabel('Speed (°/s)')
axes[2,1].set_ylabel('Acceleration (°/s^2)')
sns.despine()   



# Find peaks
d = data_disp - data_disp[:,:,:100000].mean(2,keepdims=True)

d = d.mean(1)

t = np.linspace(-20,200,22000)
plt.figure()
plt.plot(t, d[[0,3,9],198000:220000].T,alpha=.8)
plt.show()

trace = data_disp[11,0,198000:220000]
trace = gaussian_filter1d(trace, sigma=10)
peaks, _ = find_peaks(trace, width=100,height=0,distance=500,prominence=0.0005)
plt.figure()
plt.plot(data_disp[11,0,198000:220000])
plt.plot(trace)
plt.scatter(peaks, trace[peaks], marker='x', color=sns.color_palette()[1],zorder=2)
plt.show()



d = data.mean((1,2))
plt.plot(DISTANCE,d)
plt.show()




d = data - data[:,:,:100000].mean(2,keepdims=True)

d = d.mean(1)

t = np.linspace(-20,200,22000)
plt.figure()
plt.plot(t, d[[0,1],198000:220000].T,alpha=.8)
plt.show()

# d = data[:,180000:220000]
# s = stim[180000:220000]
# time = np.linspace(-200,200,d.shape[1])
# f, axes = plt.subplots(2,1)
# axes[0].plot(time,s)
# axes[0].set_ylim(-8,6)
# for i in range(d.shape[0]):
#     axes[1].plot(time,d[i])
# axes[1].set_ylim(-10,40)
# axes[1].axhline(35, c='grey', alpha=.6, linestyle='--')
# sns.despine()
# plt.suptitle(title)

# d = data[:,198000:202000]
# s = stim[198000:202000]
# time = np.linspace(-20,20,d.shape[1])
# f, axes = plt.subplots(2,1)
# axes[0].plot(time,s)
# axes[0].set_ylim(-8,6)
# for i in range(d.shape[0]):
#     axes[1].plot(time,d[i])
# axes[1].set_ylim(-10,40)
# axes[1].axhline(35, c='grey', alpha=.6, linestyle='--')
# sns.despine()
# plt.suptitle(title)

# d = data[:,199600:200400]
# s = stim[199600:200400]
# time = np.linspace(-4,4,d.shape[1])
# f, axes = plt.subplots(2,1)
# axes[0].plot(time,s)
# axes[0].set_ylim(-8,6)
# for i in range(d.shape[0]):
#     axes[1].plot(time,d[i])
# axes[1].set_ylim(-10,40)
# sns.despine()
# plt.suptitle(title)
# axes[1].axhline(35, c='grey', alpha=.6, linestyle='--')
