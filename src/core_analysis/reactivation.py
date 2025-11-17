
"""
This scritps looks at reactivationo of the response to the whisker stimulus.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
# from src.utils.utils_behavior import *  # Not needed for this script





sampling_rate = 30
win = (0, 0.180)  # from stimulus onset to 180 ms after.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40 

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)

# mice = [m for m in mice if m in mice_groups['good_day0']]

mouse = ["GF305"]

# Load mapping data.
# ------------------

print(f"Processing mouse: {mouse}")

folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
file_name = 'tensor_xarray_mapping_data.nc'
xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)
# xarray = xarray - np.nanmean(xarray.sel(time=slice(-1, 0)).values, axis=2, keepdims=True)
rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

# Select days.
xarray = xarray.sel(trial=xarray['day'].isin(days))

# Check that each day has at least n_map_trials mapping trials
# and select the first n_map_trials mapping trials for each day.
n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
if np.any(n_trials < n_map_trials):
    print(f'Not enough mapping trials for {mouse}.')
    continue

# Select last n_map_trials mapping trials for each day.
d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))
# Average bins.
d = d.sel(time=slice(win[0], win[1])).mean(dim='time')

# Remove artefacts by setting them at 0. To avoid NaN values and
# mismatches (that concerns a single cell).
print(np.isnan(d.values).sum(), 'NaN values in the data.')
d = d.fillna(0)