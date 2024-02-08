"""_summary_
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)
palette = sns.color_palette('deep')


# Calibration of 2P #1 and #3 on 31/03/2023.

PC = np.arange(0,105,5)
POWER = [2,4,8,15,22,31,42,55,67,82,100,115,132,150,166,182,200,214,229,242,255]

plt.figure()
sns.regplot(x=PC[4:], y=POWER[4:], ci=None, marker='', color=palette[1])
sns.lineplot(x=PC, y=POWER, linestyle='', marker='o', color=palette[0])
x_ticks = np.arange(0,110,10)
y_ticks = np.arange(0,275,25)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
sns.despine(trim=True)
plt.title('2P #1 power curve at 920 nm (2.7 W laser output)\n\
          EOM power in/out: 600/530 mW - Bias: 0 V')


PC = np.arange(0,105,5)
POWER = [3,5,8,11,17,22,30,36,44,53,63,74,85,97,107,120,132,144,157,168,180]

plt.figure()
sns.regplot(x=PC[4:], y=POWER[4:], ci=None, marker='', color=palette[1])
sns.lineplot(x=PC, y=POWER, linestyle='', marker='o', color=palette[0])
x_ticks = np.arange(0,110,10)
y_ticks = np.arange(0,275,25)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
sns.despine(trim=True)
plt.title('2P #3 power curve at 920 nm (2.68 W laser output)\n\
          EOM power in/out: 600/545 mW - Bias: 50 V')

# Calibration of 2P #3 on 08/05/23.

PC = np.arange(0,105,5)
POWER = [1.4,2.5,4.8,8,13,18,25,33,41,51,61,72,83,95,107,119,132,143,155,167,178]

plt.figure()
sns.regplot(x=PC[4:], y=POWER[4:], ci=None, marker='', color=palette[1])
sns.lineplot(x=PC, y=POWER, linestyle='', marker='o', color=palette[0])
x_ticks = np.arange(0,110,10)
y_ticks = np.arange(0,275,25)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
sns.despine(trim=True)
plt.title('2P #3 power curve at 940 nm (2.60 W laser output)\n\
          EOM power in: 800 mW - Bias: 50 V')