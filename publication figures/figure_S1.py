from figure_utils import get_score, get_ratio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

score_dict = get_score(
    score_files={
        'Roost': 'result_roost.pickle',
        'CrabNet': 'result_crabnet.pickle',
        'CGCNN': 'result_cgcnn.pickle',
        'MEGNet': 'result_megnet.pickle'
        }
    )
data_dict = get_ratio()

# Plot
mpl.style.use('default')
font = {
        'family': 'Proxima Nova',
        'size': 14,
        'weight': 'light'
    }

math_font = 'cm'

mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = math_font
mpl.rcParams['axes.labelweight'] = 'light'
mpl.rcParams['xtick.labelsize'] = font['size']-2
mpl.rcParams['ytick.labelsize'] = font['size']-2
mpl.rcParams['legend.fontsize'] = font['size']-2
mpl.rcParams['legend.frameon'] = False

data_segregations = ('non-poly', 'poly')

fig = plt.figure(figsize=(11, 5))
outer = gridspec.GridSpec(2, 1, figure=fig, left=0.05, right=0.95,
                          hspace=0.6)
a = gridspec.GridSpecFromSubplotSpec(1, 13, width_ratios=[20, 0.5, 15, 0.1, 15, 0.1, 15, 0.1, 15, 0.1, 15, 0.1, 15], 
                                     subplot_spec=outer[0], wspace=0.3)
b = gridspec.GridSpecFromSubplotSpec(1, 13, width_ratios=[20, 0.5, 15, 0.1, 15, 0.1, 15, 0.1, 15, 0.1, 15, 0.1, 15], 
                                     subplot_spec=outer[1], wspace=0.3)

x = np.array([0, 0.3, 0.5, 0.8, 1])
y = np.array([0, -0.9, -1, -0.8, 0])
delta = 1.5
offset = 0.15

ax = plt.subplot(a[0])
x_ = np.array([0, 0, 0.15, 0.3, 0.5, 0.5, 0.5, 0.5, 0.65, 0.65, 0.65, 0.8, 0.8, 1])
y_ = np.array([0, 1, 0.2, -0.9, -1, 0.8, -0.4, 1.2, -0.1, 0.4, 0.7, -0.8, 0.3, 0])
n_p = [2, 3, 13]
p = np.setdiff1d(np.arange(len(x_)), n_p)

fs = 18

ax.plot(x, y-0.5*delta, 'k-', lw=1.3)
ax.scatter(x_[n_p], y_[n_p]-0.5*delta, marker='x', c='darkorange')
ax.scatter(x_[p], y_[p]-0.5*delta, marker='x', c='k')
ax.text(0.025, 2.7, 'x', fontweight='semibold', color='darkorange', fontsize=fs)
ax.text(0.125, 2.7, ': non-poly', fontweight='semibold', fontsize=fs)
ax.set_ylim(-2.5, 4)
ax.set_xlim(-0.05, 1.05)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Composition',)
ax.set_ylabel('Formation Energy')

ax = plt.subplot(b[0])
ax.plot(x, y-0.5*delta, 'k-', lw=1.3)
ax.scatter(x_[n_p], y_[n_p]-0.5*delta, marker='x', c='k')
ax.scatter(x_[p], y_[p]-0.5*delta, marker='x', c='purple')

ax.text(0.23, 2.7, 'x', fontweight='semibold', color='purple', fontsize=fs)
ax.text(0.33, 2.7, ': poly', fontweight='semibold', fontsize=fs)
ax.set_ylim(-2.5, 4)
ax.set_xlim(-0.05, 1.05)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Composition',)
ax.set_ylabel('Formation Energy')

def plot_prop(ax, data_segregation, prop, data_segregations=('non-poly', 'poly')):
    method = ['Roost', 'CrabNet', 'MEGNet', 'CGCNN']
    bin_width = 0.5
    x_bar = np.array([-0.75, -0.25, 0.25, 0.75])
    color = [
        tuple(np.array([192, 192, 245])/255),
        tuple(np.array([96, 96, 250])/255),
        tuple(np.array([245, 192, 192])/255),
        tuple(np.array([250, 96, 96])/255)
    ]
    offset = 0.05
    offset_ = 0.01
    x_bar[:2] = x_bar[:2] - offset
    x_bar[0] = x_bar[0] - offset_
    x_bar[1] = x_bar[1] + offset_
    x_bar[2:] = x_bar[2:] + offset
    x_bar[2] = x_bar[2] - offset_
    x_bar[3] = x_bar[3] + offset_
    y_bar = [score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']==data_segregation]['MAE_mean'].drop_duplicates().values[0]
              for m in method]
    y_max = max(
        max([score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='stable']['MAE_max'].drop_duplicates().values[0] +
              score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='stable']['MAE_mean'].drop_duplicates().values[0]
              for m in method+['MEGNet']]),
        max([score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='unstable']['MAE_max'].drop_duplicates().values[0] +
              score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='unstable']['MAE_mean'].drop_duplicates().values[0]
              for m in method+['MEGNet']]),
    )
    y_err = np.array([score_dict[prop][1][m][data_segregation].values for m in method]).T
    ax.bar(x_bar, y_bar, yerr=y_err, width=bin_width, color=color, capsize=2, error_kw={'elinewidth': 1})
    ax.set_ylim(top=np.around(y_max, decimals=2)+0.01)
    ax.set_xlim(x_bar[0]-0.5*bin_width-0.1, x_bar[-1]+0.5*bin_width+0.5)
    ax.set_xticks(x_bar)
    ndata_train = np.round(score_dict[prop][2].loc[data_segregation]['ndata_train']/1000).astype('int')
    ax.text(1.08, -0.25, f"$\sim${ndata_train}K", ha='right', fontsize=font['size']-4,
            transform=ax.transAxes)
    if data_segregation == data_segregations[0] and prop == 'formation_energy_per_atom':
        ax.set_xticklabels(method, rotation=45, rotation_mode='anchor', ha='right')
        ax.text(-2.07, 0.159, 'MAE', fontsize=font['size']-2)
        ax.text(-1.15, 0.135, 'Comp', fontweight='semibold', color=color[1], fontsize=font['size']-2)
        ax.text(0.1, 0.11, 'Comp', fontweight='semibold', color=color[3], fontsize=font['size']-2)
        ax.text(0.62, 0.121, '+', fontweight='semibold', color=color[3], fontsize=font['size']-2)
        ax.text(0.1, 0.135, 'Struct', fontweight='semibold', color=color[3], fontsize=font['size']-2)
    else:
        ax.set_xticklabels([])
    ax.tick_params(direction='out', pad=1)
    ax.set_yticklabels(np.array(ax.get_yticks().tolist()).round(2), rotation=45, rotation_mode='anchor')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    comp = min(y_bar[:2])
    compstruct = min(y_bar[2:])
    improvement = (comp-compstruct)/compstruct
    ax.hlines(comp, x_bar[np.argmin(y_bar[:2])], x_bar[-1]+0.5*bin_width+0.5, 
              colors='k', linestyles='dashed', linewidth=1.1)
    ax.hlines(compstruct, x_bar[np.argmin(y_bar[2:])+2], x_bar[-1]+0.5*bin_width+0.5, 
              colors='k', linestyles='dashed', linewidth=1.1)
    bottom, top = ax.get_ylim()
    
    if abs(comp-compstruct)/(top-bottom) > 0.2:
        length = 0.06*(top-bottom)
        offset = 0.01*(top-bottom)
        head_length = 0.3*length
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, max(comp, compstruct)-length-offset, 0, length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, min(comp, compstruct)+length+offset, 0, -length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.text(x_bar[-1]+1.12, min(comp, compstruct)+0.38*(comp-compstruct), 
                str(np.round(improvement*100, 1))+'%', fontweight='semibold', ha='right')
    else:
        length = 0.06*(top-bottom)
        offset = 0.01*(top-bottom)
        head_length = 0.3*length
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, max(comp, compstruct)+length+offset, 0, -length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, min(comp, compstruct)-length-offset, 0, length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.text(x_bar[-1]+0.99, max(comp, compstruct)+0.12*(top-bottom), 
                str(np.round(improvement*100, 1))+'%', fontweight='semibold', ha='right')
    return improvement, data_segregation, prop

improvements = {}
props = ('formation_energy_per_atom', 'band_gap', 'elasticity.K_VRH', 
          'elasticity.G_VRH', 'density', 'point_density')
for i, prop in zip((2, 4, 6, 8, 10, 12), props):

    ax = plt.subplot(a[i])
    improvement, data_segregation, prop = plot_prop(ax, data_segregations[0], prop, data_segregations)
    improvements[(prop, data_segregation)] = improvement
    ax = plt.subplot(b[i])
    improvement, data_segregation, prop = plot_prop(ax, data_segregations[1], prop, data_segregations)
    improvements[(prop, data_segregation)] = improvement

plt.text(0.026, 0.9, '(A)', transform=fig.transFigure)
plt.text(0.191, 0.9, '(B)    eV/atom', transform=fig.transFigure)
plt.text(0.249, 0.95, '$E_\mathrm{f}$', transform=fig.transFigure)
plt.text(0.32, 0.9, '(C)         eV', transform=fig.transFigure)
plt.text(0.383, 0.95, '$E_\mathrm{g}$', transform=fig.transFigure)
plt.text(0.448, 0.9, '(D)     log(GPa)', transform=fig.transFigure)
plt.text(0.497, 0.95, '$K_\mathrm{VRH}$', transform=fig.transFigure)
plt.text(0.578, 0.9, '(E)     log(GPa)', transform=fig.transFigure)
plt.text(0.63, 0.95, '$G_\mathrm{VRH}$', transform=fig.transFigure)
plt.text(0.708, 0.9, '(F)     g/cm$^3$', transform=fig.transFigure)
plt.text(0.763, 0.95, '$\\rho$', transform=fig.transFigure)
plt.text(0.831, 0.9, '(G)     1e-3/$\AA^3$', transform=fig.transFigure)
plt.text(0.86, 0.95, 'point density', transform=fig.transFigure)
plt.text(0.026, 0.425, '(H)', transform=fig.transFigure)
plt.text(0.198, 0.425, '(I)', transform=fig.transFigure)
plt.text(0.323, 0.425, '(J)', transform=fig.transFigure)
plt.text(0.449, 0.425, '(K)', transform=fig.transFigure)
plt.text(0.579, 0.425, '(L)', transform=fig.transFigure)
plt.text(0.702, 0.425, '(M)', transform=fig.transFigure)
plt.text(0.831, 0.425, '(N)', transform=fig.transFigure)

# Plot stable/unstable ratios
from matplotlib.patches import Patch
ax = fig.add_axes([1/7+0.005, 1/2+0.03, 0.05, 0.05])
ax.pie([0.5, 0.5], colors=['palegreen', 'palegoldenrod'],
        wedgeprops={"edgecolor":"0",})
legend_elements = [Patch(facecolor='palegreen', edgecolor='0',
                      label='stable'),
                    Patch(facecolor='palegoldenrod', edgecolor='0',
                      label='unstable'),
                    ]
    
ax = fig.add_axes([1/7-0.045, 1/2-0.05, 0.05, 0.05])
ax.legend(handles=legend_elements, loc='center', handlelength=1, 
          handleheight=1, handletextpad=0.3, fontsize=font['size']-2)
ax.axis('off')

xs = (2/7-0.01, 3/7-0.024, 4/7-0.038, 5/7-0.053, 6/7-0.067, 1-0.081)

for prop, x in zip(props, xs):
    for i, data_segregation in enumerate(data_segregations):
        df = data_dict[(data_segregation, prop)]
        if i == 0:
            ax = fig.add_axes([x, 1/2+0.03, 0.05, 0.05])
        else:
            ax = fig.add_axes([x, 0.055, 0.05, 0.05])
        ax.pie(np.unique(df['stab'].sort_values().values, return_counts=True)[1], 
                colors=['palegreen', 'palegoldenrod'],
                wedgeprops={"edgecolor":"0",}
                )

# Plot grey shading for point_density
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.fill_betweenx([0.02, 0.988], [0.832, 0.832], [0.963, 0.963], alpha=0.1, color='k')
ax.axis('off')

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/Figure S1.png', dpi=600, bbox_inches='tight')