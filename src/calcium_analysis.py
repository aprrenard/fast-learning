import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)
palette = sns.color_palette()
path = 'D:\\analysis\\Anthony_Renard\\data\\PB124_20230331_173237\\suite2p\\plane0'
F = np.load(os.path.join(path,'F.npy'))
Fneu = np.load(os.path.join(path,'Fneu.npy'))


icell = 0
Fc = F - 0.7 * Fneu
F0 = np.percentile(Fc, 8, axis=1)

time = np.arange(0,Fc.shape[1]) / 30

f, axes = plt.subplots(3,1, sharex=True)

axes[0].plot(time, F[icell,:])
axes[0].plot(time, Fneu[icell], c=palette[1])
axes[0].set_ylabel('Fraw and Fneu')

axes[1].axhline(F0[icell], c=palette[1])
axes[1].plot(time, Fc[icell,:])
axes[1].set_ylabel('F corrected\n(Fcor = F - 0.7 * Fneu)')

axes[2].plot(time, (Fc[icell,:]-F0[icell]) / F0[icell])
axes[2].set_ylabel('(F - F0) / F0')
axes[2].set_xlabel('Time (sec)')
sns.despine()


plt.figure()
plt.plot(time, F[icell,:])
plt.ylabel('Raw Fluorescence (ROI_id = 0)')
plt.xlabel('Time (sec)')
sns.despine()