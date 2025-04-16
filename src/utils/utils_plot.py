import matplotlib.pyplot as plt
import seaborn as sns


# Color palettes.
reward_palette = sns.color_palette(['#c959affe', '#1b9e77'])
cell_types_palette = sns.color_palette(['#a3a3a3', '#1f77b4', '#ff7f0e'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# s2_m1_palette = sns.color_palette(['#6D9BC3', '#E67A59'])
s2_m1_palette = sns.color_palette(['steelblue', 'salmon'])
# s2_m1_palette = sns.color_palette(['#cc5500','#4682b4',]) 
stim_palette = sns.color_palette(['#1f77b4', '#FF9600', '#333333'])
behavior_palette = sns.color_palette(['#0ddddd', '#1f77b4', '#c959affe', '#1b9e77', '#333333', '#cccccc'])



# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
