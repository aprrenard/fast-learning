import matplotlib.pyplot as plt
import seaborn as sns


# Set plot parameters.
sns.set_theme(
    context='paper',
    style='ticks',
    palette='deep',
    font='sans-serif',
    font_scale=1,
    rc={
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        # Font sizes
        'axes.labelsize': 10,      # X and Y axis label font size
        'xtick.labelsize': 10,      # X tick label font size
        'ytick.labelsize': 10,      # Y tick label font size
        'axes.titlesize': 10,      # Title font size
        'legend.fontsize': 10,      # Legend font size
        'font.size': 10,           # Base font size
    }
)

# # Color palettes.
# reward_palette = sns.color_palette(['#c959affe', '#1b9e77'])
# reward_palette_r = sns.color_palette([ '#1b9e77', '#c959affe'])
# cell_types_palette = sns.color_palette(['#a3a3a3', '#1f77b4', '#ff7f0e'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# # s2_m1_palette = sns.color_palette(['#6D9BC3', '#E67A59'])
# s2_m1_palette = sns.color_palette(['steelblue', 'salmon'])
# # s2_m1_palette = sns.color_palette(['#cc5500','#4682b4',]) 
# stim_palette = sns.color_palette(['#1f77b4', '#FF9600', '#333333'])
# behavior_palette = sns.color_palette(['#06fcfeff', '#1f77b4', '#c959affe', '#1b9e77', '#cccccc', '#333333'])

# Color palettes.
reward_palette = sns.color_palette(['#980099ff', '#009600ff'])
reward_palette_r = sns.color_palette([ '#009600ff', '#980099ff'])
cell_types_palette = sns.color_palette(['#807f7fff', 'salmon', 'steelblue'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# s2_m1_palette = sns.color_palette(['#6D9BC3', '#E67A59'])
s2_m1_palette = sns.color_palette(['steelblue', 'salmon'])
# s2_m1_palette = sns.color_palette(['#cc5500','#4682b4',]) 
stim_palette = sns.color_palette(['#0100fdff', '#FF9600', '#010101ff'])
behavior_palette = sns.color_palette(['#06fcfeff', '#0100fdff', '#980099ff', '#009600ff', '#807f7fff', '#010101ff'])

# Mice groups.
mice_groups = {
    'gradual_day0': ['GF278', 'GF301', 'GF305', 'GF306', 'GF313', 'GF317', 'GF318', 'GF323', 'GF328', ],
    'step_day0': ['GF339', 'AR176', ],
    'psth_mapping_increase': ['GF305', 'GF306', 'GF308', 'GF313', 'GF318', 'GF323', 'GF334',],
    'good_day0':['GF240', 'GF241','GF248','GF253','GF267','GF278','GF257','GF287',
                 'GF261','GF266','GF300','GF301','GF303','GF305','GF306',
                 'GF307','GF308','GF310','GF311','GF313','GF314',
                 'GF317','GF318','GF323','GF325','GF326','GF327','GF328','GF334','GF336','GF337','GF338','GF339',
                 'GF353','GF354','GF355','MI023','MI026','MI023','MI028','MI029','MI030','MI031',
                 'MI054','MI055','AR121','AR133','AR135','AR176',],
    'meh_day0': ['GF252', 'GF256','GF272','GF264','GF333','AR123','AR143','AR177','AR127',],
    'bad_day0': ['GF271','GF290','GF291','GF292','GF293','GF249','MI012','MI014','MI027','MI039',
                 'MI040','MI044','MI045','MI053','AR115','AR116','AR117','AR119','AR120','AR122','AR144',
                 'AR163',],
}