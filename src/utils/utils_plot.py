import matplotlib.pyplot as plt
import seaborn as sns


# Set plot parameters.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# Color palettes.
reward_palette = sns.color_palette(['#c959affe', '#1b9e77'])
cell_types_palette = sns.color_palette(['#a3a3a3', '#1f77b4', '#ff7f0e'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# s2_m1_palette = sns.color_palette(['#6D9BC3', '#E67A59'])
s2_m1_palette = sns.color_palette(['steelblue', 'salmon'])
# s2_m1_palette = sns.color_palette(['#cc5500','#4682b4',]) 
stim_palette = sns.color_palette(['#1f77b4', '#FF9600', '#333333'])
behavior_palette = sns.color_palette(['#0ddddd', '#1f77b4', '#c959affe', '#1b9e77', '#cccccc', '#333333'])

# Mice groups.
mice_groups = {
    'gradual_day0': ['GF278', 'GF301', 'GF305', 'GF306', 'GF313', 'GF317', 'GF318', 'GF323', 'GF328', ],
    'step_day0': ['GF339', 'AR176', ],
    'psth_mapping_increase': ['GF305', 'GF306', 'GF308', 'GF313', 'GF318', 'GF323', 'GF334',],
    'good_day0':['GF240', 'GF241','GF248','GF253','GF267','GF278''GF257','GF287',
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
# 