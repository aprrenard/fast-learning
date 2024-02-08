"""Improvement of GF baseline computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter
from scipy.signal import filtfilt, firwin

F_raw = np.load('C:\\Users\\aprenard\\recherches\\fast-learning\\data\\F.npy')

fw_base = 1
nfilt = 30
fs = 30
nyq_rate = fs / 2.0
cutoff = min(1.0, fw_base / nyq_rate)
b = firwin(nfilt, cutoff=cutoff, window='hamming')
padlen = min(3 * nfilt, F_raw.shape[1] - 1)

F_filt = filtfilt(b, [1.0], F_raw, axis=1,
                                    padlen=padlen)

F0 = percentile_filter(F_filt, 5, size=(1,fs*120))

icell = 0
plt.plot(F_raw[icell])
plt.plot(F_filt[icell])
plt.plot(F0[icell])
