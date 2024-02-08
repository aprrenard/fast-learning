import os
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:/Users/aprenard/timestamp_continuous.bin"

a = np.fromfile(file_path,dtype='double')
a.shape
plt.plot(np.abs(a[0:10000:4]))
plt.plot(a[np.arange(1, 1000, 4)])

# a = a.reshape((-1,5))
# a.shape

plt.plot(a[:100,1])
