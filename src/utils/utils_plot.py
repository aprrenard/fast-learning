import matplotlib.pyplot as plt
import seaborn as sns


# Color palettes.
reward_palette = sns.color_palette(['#c959affe', '#1b9e77'])
cell_types_palette = sns.color_palette(['#a3a3a3', '#c959affe', '#3351ffff'])
s2_m1_palette = sns.color_palette(['#c959affe', '#3351ffff'])
stim_palette = sns.color_palette(['#3333ffff', '#008000ff', '#be3133ff', '#FF9600'])

# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
