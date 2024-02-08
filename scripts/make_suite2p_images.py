import os
import numpy as np
import matplotlib.pyplot as plt


ops_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\Suite2PRois\\AR106\\suite2p\\plane0\\ops.npy'
ops = np.load(ops_path, allow_pickle=True).item()

mean_img = ops['meanImg']
max_img = ops['max_proj']

plt.imshow(mean_img, vmin=10, vmax=100)
plt.imshow(max_img, vmin=10, vmax=50)


mean_img.shape