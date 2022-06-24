from figure_utils import get_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

score_dict = get_score(
    score_files={
        'CGCNN': 'result_cgcnn.pickle',
        'MEGNet': 'result_megnet.pickle',
        'CGCNN-S': 'result_cgcnn-S.pickle',
        'MEGNet-S': 'result_megnet-S.pickle',
        }
    )

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
ax.text(0.025, 2.7, 'x', fontweight='bold', color='darkorange', fontsize=fs)
ax.text(0.125, 2.7, ': non-poly', fontweight='bold', fontsize=fs)
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

ax.text(0.23, 2.7, 'x', fontweight='bold', color='purple', fontsize=fs)
ax.text(0.33, 2.7, ': poly', fontweight='bold', fontsize=fs)
ax.set_ylim(-2.5, 4)
ax.set_xlim(-0.05, 1.05)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Composition',)
ax.set_ylabel('Formation Energy')

def plot_prop(ax, data_segregation, prop):
    method = ['MEGNet-S', 'CGCNN-S', 'MEGNet', 'CGCNN']
    bin_width = 0.5
    x_bar = np.array([-0.75, -0.25, 0.25, 0.75])
    color = [
        tuple(np.array([202, 178, 214])/255),
        tuple(np.array([106, 61, 154])/255),
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
              for m in method]),
        max([score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='unstable']['MAE_max'].drop_duplicates().values[0] +
              score_dict[prop][0][m][score_dict[prop][0][m]['data segregation']=='unstable']['MAE_mean'].drop_duplicates().values[0]
              for m in method]),
    )
    y_err = np.array([score_dict[prop][1][m][data_segregation].values for m in method]).T
    ax.bar(x_bar, y_bar, yerr=y_err, width=bin_width, color=color, capsize=2, error_kw={'elinewidth': 1})
    ax.set_ylim(top=np.around(y_max, decimals=2)+0.05)
    ax.set_xlim(x_bar[0]-0.5*bin_width-0.1, x_bar[-1]+0.5*bin_width+0.5)
    ax.set_xticks(x_bar)
    ndata_train = np.round(score_dict[prop][2].loc[data_segregation]['ndata_train']/1000).astype('int')
    if data_segregation == 'non-poly' and prop == 'formation_energy_per_atom':
        ax.set_xticklabels(method, rotation=45, rotation_mode='anchor', ha='right')
        ax.text(-0.03, 0.94, 'MAE', fontsize=font['size']-2, transform=ax.transAxes, horizontalalignment='right')
        ax.text(-1.15, 0.415, 'Struct', fontweight='bold', color=color[1], fontsize=font['size']-2)
        ax.text(0.1, 0.342, 'Comp', fontweight='bold', color=color[3], fontsize=font['size']-2)
        ax.text(0.6, 0.372, '+', fontweight='bold', color=color[3], fontsize=font['size']-2)
        ax.text(0.1, 0.415, 'Struct', fontweight='bold', color=color[3], fontsize=font['size']-2)
        ax.text(1.08, -0.23, f"$\sim${ndata_train}K", ha='right', fontsize=font['size']-4,
            transform=ax.transAxes)
    else:
        ax.set_xticklabels([])
        ax.text(1.08, -0.13, f"$\sim${ndata_train}K", ha='right', fontsize=font['size']-4,
            transform=ax.transAxes)
    ax.tick_params(direction='out', pad=1)
    ax.set_yticklabels(np.array(ax.get_yticks().tolist()).round(2), rotation=45, rotation_mode='anchor')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    struct = min(y_bar[:2])
    compstruct = min(y_bar[2:])
    improvement = np.abs(struct-compstruct)/compstruct
    ax.hlines(struct, x_bar[np.argmin(y_bar[:2])], x_bar[-1]+0.5*bin_width+0.5, 
              colors='k', linestyles='dashed', linewidth=1.1)
    ax.hlines(compstruct, x_bar[np.argmin(y_bar[2:])+2], x_bar[-1]+0.5*bin_width+0.5, 
              colors='k', linestyles='dashed', linewidth=1.1)
    bottom, top = ax.get_ylim()
    
    if abs(struct-compstruct)/(top-bottom) > 0.3:
        length = 0.06*(top-bottom)
        offset = 0.01*(top-bottom)
        head_length = 0.3*length
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, max(struct, compstruct)-length-offset, 0, length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, min(struct, compstruct)+length+offset, 0, -length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.text(x_bar[-1]+1.04, min(struct, compstruct)+0.38*(struct-compstruct), 
                str(np.round(improvement*100, 1))+'%', fontweight='bold', ha='right')
    else:
        length = 0.06*(top-bottom)
        offset = 0.01*(top-bottom)
        head_length = 0.3*length
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, max(struct, compstruct)+length+offset, 0, -length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.arrow(x_bar[-1]+0.5*bin_width+0.3, min(struct, compstruct)-length-offset, 0, length,
                  length_includes_head=True, lw=0.7, head_length=head_length, 
                  head_width=0.15, color='k', ls='--'
        )
        ax.text(x_bar[-1]+0.99, max(struct, compstruct)+0.12*(top-bottom), 
                str(np.round(improvement*100, 1))+'%', fontweight='bold', ha='right')
    return improvement, data_segregation, prop

improvements = {}
props = ('formation_energy_per_atom', 'band_gap', 'elasticity.K_VRH', 
          'elasticity.G_VRH', 'density', 'point_density')
for i, prop in zip((2, 4, 6, 8, 10, 12), props):

    ax = plt.subplot(a[i])
    improvement, data_segregation, prop = plot_prop(ax, 'non-poly', prop)
    improvements[(prop, data_segregation)] = improvement
    ax = plt.subplot(b[i])
    improvement, data_segregation, prop = plot_prop(ax, 'poly', prop)
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

# Plot grey shading for point_density
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.fill_betweenx([0.05, 0.988], [0.832, 0.832], [0.963, 0.963], alpha=0.1, color='k')
ax.axis('off')

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/Figure S1.png', dpi=600, bbox_inches='tight')