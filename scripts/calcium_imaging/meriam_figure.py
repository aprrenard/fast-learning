import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = 'M:\\z_LSENS\\Share\\Meriam_Malekzadeh\\MM021\\suite2p\\plane0'

F = np.load(os.path.join(FOLDER,'F.npy'))
Fneu = np.load(os.path.join(FOLDER,'Fneu.npy'))
stat = np.load(os.path.join(FOLDER,'stat.npy'), allow_pickle=True)
ops = np.load(os.path.join(FOLDER,'ops.npy'), allow_pickle=True).item()

ops.keys()

# Look for nice cells.
# shift = 1000
# for icell in range(F.shape[0]):
#     plt.plot(F[icell]-F[icell].mean() + shift*icell)

rois = [3,2,1,6,7]
rois_values = range(len(rois))
ncells = stat.shape[0]
meanImg = ops['meanImg']

plt.imshow(ops['meanImg'],vmax=meanImg.max()*.5,cmap='gray')
im = np.zeros((ops['Ly'], ops['Lx'])) * np.nan
for icell,iroi in enumerate(rois):
    ypix = stat[iroi]['ypix'][~stat[iroi]['overlap']]
    xpix = stat[iroi]['xpix'][~stat[iroi]['overlap']]
    im[ypix,xpix] = rois_values[icell]
plt.imshow(im, interpolation='none',alpha=1)


_, axes = plt.subplots(1,2)

shift = 2000
for icell, iroi in enumerate(rois):
    axes[0].axhline(np.percentile(F[iroi],8) + shift*icell, linestyle='--', c='grey')
    axes[0].plot(F[iroi] + shift*icell)
    axes[0].plot(Fneu[iroi] + shift*icell, c='grey')


shift = 5
for icell, iroi in enumerate(rois):
    axes[1].plot( (F[iroi]-Fneu[iroi]*0.7) / np.percentile(F[iroi]-Fneu[iroi]*0.7,8) + shift*icell)
