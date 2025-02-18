import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as imaging_utils
import src.utils.utils_io as io
from src.behavior import compute_performance, plot_single_session
import warnings

# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
%matplotlib inline

# Path to the directory containing the processed data.
processed_dir = io.solve_common_paths('processed_data')
nwb_dir = io.solve_common_paths('nwb')
db_path = io.solve_common_paths('db')



# #############################################################################
# 1. PSTH's.
# #############################################################################

# Parameters.
# -----------

sampling_rate = 30
win_sec = (0.8, 3)  
win = (int(win_sec[0] * sampling_rate), int(win_sec[1] * sampling_rate))
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = ['-2', '-1', '0', '+1', '+2']
cell_type = None
variance_across = 'cells'

_, _, mice, _ = io.select_sessions_from_db(db_path,
                                            nwb_dir,
                                            two_p_imaging='yes')
# mice = [m for m in mice if m not in ['AR163']]
print(mice)
len(mice)


# Load the data.
# --------------

psth = []

for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = io.get_mouse_reward_group_from_db(db_path, mouse_id)

    file_name = 'tensor_xarray_mapping_data.nc'
    data = imaging_utils.load_mouse_xarray(mouse_id, processed_dir, file_name)
    data = imaging_utils.substract_baseline(data, 2, baseline_win)
        
    avg_response = data.groupby('day').mean(dim='trial')
    avg_response.name = 'psth'
    avg_response = avg_response.to_dataframe().reset_index()
    avg_response['mouse_id'] = mouse_id
    avg_response['reward_group'] = reward_group
    
    psth.append(avg_response)
psth = pd.concat(psth)


# Grand average psth's for all cells and projection neurons.
# ----------------------------------------------------------

data = psth.loc[psth.time<1.5]
data = data.loc[data['day'].isin([-2, -1, 0, 1, 2])]
data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type'])['psth'].agg('mean').reset_index()

# # GF305 has baseline artefact on day -1 at auditory trials.
# data = data.loc[~data.mouse_id.isin(['GF305'])]

# data = data.loc[data['cell_type']=='wM1']
# data = data.loc[~data.mouse_id.isin(['GF305', 'GF306', 'GF307'])]
fig = sns.relplot(data=data, x='time', y='psth', errorbar='ci', col='day',
            kind='line', hue='reward_group',
            hue_order=['R-','R+'], palette=sns.color_palette(['#D81B60', '#238443']),
            height=3, aspect=0.8)
for ax in fig.axes.flatten():
    ax.axvline(0, color='#FF9600', linestyle='--')
    ax.set_title('')    

fig = sns.relplot(data=data, x='time', y='psth', errorbar='se', col='day', row='cell_type',
            kind='line', hue='reward_group',
            hue_order=['R-','R+'], palette=sns.color_palette(['#D81B60', '#238443']), row_order=['wS2', 'wM1',],
            height=3, aspect=0.8)
for ax in fig.axes.flatten():
    ax.axvline(0, color='#FF9600', linestyle='--')
    ax.set_title('')


# Individual mice PSTH's.

pdf_file = f'psth_individual_mice_auditory.pdf'
output_dir = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/sensory_plasticity/psth'
output_dir = io.adjust_path_to_host(output_dir)

with PdfPages(os.path.join(output_dir, pdf_file)) as pdf:
    for mouse_id in mice:
        # Plot.
        data = psth.loc[psth['day'].isin([-2, -1, 0, 1, 2])
                      & (psth['mouse_id'] == mouse_id)]

        sns.relplot(data=data, x='time', y='psth', errorbar='se', col='day', row='cell_type',
                    kind='line', hue='reward_group',
                    hue_order=['R-','R+'], palette=sns.color_palette(['#D81B60', '#238443']))
        plt.suptitle(mouse_id)
        pdf.savefig(dpi=300)
        plt.close()

