import numpy as np
from copy import deepcopy

from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_iterator(iterator1, iterator2, iterator3):
    '''
    get_iterator generates combined iterators (fed to zip) spanning every combination of
    elements from component iterators in a hierachical manner.
    
    Example: When iterator1 = [A, B], iterator2 = [1, 2], iterator3 = [a, b], get_iterator yields 
        my_iterator1 = [A, A, A, A, B, B, B, B]
        my_iterator2 = [1, 1, 2, 2, 1, 1, 2, 2]
        my_iterator3 = [a, b, a, b, a, b, a, b], which can be inputted into zip to iterate over.

    Parameters
    ----------
    iterator1 : list or int
        Fist hierarchy component iterator. If list, taken as the iterator. If int,
        take list(range(iterator1)) as the iterator.
    iterator2 : list or int
        Second hierarchy component iterator. If list, taken as the iterator. If int,
        take list(range(iterator2)) as the iterator.
    iterator3 : list or int
        Third hierarchy component iterator. If list, taken as the iterator. If int,
        take list(range(iterator3)) as the iterator.

    Returns
    -------
    tuple of three combined iterators readily to be passed into zip.
    '''
    def _(iterator):
        if isinstance(iterator, int):
            iterator_list = list(range(iterator))
            iterator_len = deepcopy(iterator)
        elif isinstance(iterator, list):
            iterator_list = deepcopy(iterator)
            iterator_len = len(iterator)
        else:
            raise ValueError('iterator must be either int or list')
        return iterator_list, iterator_len
    iterator1_list, iterator1_len = _(iterator1)
    iterator2_list, iterator2_len = _(iterator2)
    iterator3_list, iterator3_len = _(iterator3)
    my_iterator1 = list(np.repeat(iterator1_list, iterator2_len*iterator3_len))
    my_iterator2 = list(np.tile(np.repeat(iterator2_list, iterator3_len), iterator1_len))
    my_iterator3 = list(np.tile(iterator3_list, iterator1_len*iterator2_len))
    
    return my_iterator1, my_iterator2, my_iterator3

def plot(y, y_hat, prop, plotdir):
    
    mpl.style.use('default')
    mpl.use('Agg')
    font = {
            'family': 'Calibri',
            'size': 27,
            'weight': 'light'
        }

    math_font = 'cm'

    mpl.rc('font', **font)
    mpl.rcParams['mathtext.fontset'] = math_font
    mpl.rcParams['axes.labelweight'] = 'light'
    mpl.rcParams['xtick.labelsize'] = font['size']-2
    mpl.rcParams['ytick.labelsize'] = font['size']-2
    mpl.rcParams['figure.figsize'] = [8, 7]
    mpl.rcParams['legend.fontsize'] = 25
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['figure.dpi'] = 50

    unit = {
        'formation_energy_per_atom': 'meV/atom',
        'band_gap': 'eV',
        'density': '$g/cm^3$',
        'elasticity.K_VRH': 'log(GPa)',
        'elasticity.G_VRH': 'log(GPa)',
        'point_density': '$1e-3/\AA^3$',
        }
    
    prop_d_proper = {
        'formation_energy_per_atom': '$E_\mathrm{f}$',
        'band_gap': '$E_\mathrm{g}$',
        'density': '$ρ$',
        'elasticity.K_VRH': '$K_\mathrm{VRH}$',
        'elasticity.G_VRH': '$G_\mathrm{VRH}$',
        'point_density': 'Point Density'
        }
    
    # Plot
    fig1, ax1 = plt.subplots(figsize=(8, 8.3)) 
    ax1.scatter(y, y_hat, 4)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax1.set_xlabel('True Value')
    ax1.set_ylabel('Predicted Value')
    
    if 'meV' in unit[prop]:
        ax1.set_title(prop_d_proper[prop]+' MAE: %.1f ' % (metrics.mean_absolute_error(y, y_hat)*1000)+unit[prop])
    elif '1e-3' in unit[prop]:
        ax1.set_title(prop_d_proper[prop]+' MAE: %.2f ' % (metrics.mean_absolute_error(y, y_hat)*1000)+unit[prop])
    else:
        ax1.set_title(prop_d_proper[prop]+' MAE: %.3f ' % (metrics.mean_absolute_error(y, y_hat))+unit[prop])
    plt.tight_layout()
    plt.savefig(f"{plotdir}.png", dpi=600,)
    plt.close()

def get_prop_d():
    '''
    Return a dictionary for corresponding property names to simple property name

    Returns
    -------
    prop_d : dictionary
    '''
    prop_d = {
        'formation_energy_per_atom': 'E_f',
        'band_gap': 'E_g',
        'density': 'rho',
        'elasticity.K_VRH': 'K_VRH',
        'elasticity.G_VRH': 'G_VRH',
        'point_density': 'point_density'
    }
    return prop_d