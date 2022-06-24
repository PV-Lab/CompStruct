import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy

# Combine results for every model
def score_combine(directory, struct=False):
    for root, dirs, files in os.walk(directory):
        break
    for file in files:
        if struct:
            if ('.pickle' in file) & ('-S' in file):
                keys = pickle.load(open(os.path.join(root, file), 'rb')).keys()
                break
        else:
            if ('.pickle' in file) & (not '-S' in file):
                keys = pickle.load(open(os.path.join(root, file), 'rb')).keys()
                break
    temp = {key: [] for key in keys}
    for file in files:
        if struct:
            if ('.pickle' in file) & ('-S' in file):
                d = pickle.load(open(os.path.join(root, file), 'rb'))
                for key in keys:
                    temp[key] = temp[key]+[d[key]]
        else:
            if ('.pickle' in file) & (not '-S' in file):
                d = pickle.load(open(os.path.join(root, file), 'rb'))
                for key in keys:
                    temp[key] = temp[key]+[d[key]]
    
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return {key: locs for key,locs in tally.items() 
                                if len(locs)>1}
    
    d = list_duplicates(zip(temp['data segregation'], temp['property']))
    
    indices = []
    for value in d.values():
        temp['score'][value[0]] = np.stack((temp['score'][v] for v in value))
        indices = indices + value[1:]
    
    for key in keys:
        for index in sorted(indices, reverse=True):
            del temp[key][index]
    
    struct_name = {
        True: '-S',
        False: '',
        }
    
    return pickle.dump(temp, open(f"result_{directory.split('/')[-1]}{struct_name[struct]}.pickle", 'wb'))

for model in ('megnet', 'cgcnn',):
    for struct in (False, True):
        score_combine(f'../results/{model}', struct)
for model in ('crabnet', 'roost'):
    score_combine(f'../results/{model}')

# Get combined scores for everything
def get_score(score_files={
        'Roost': 'result_roost.pickle',
        'CrabNet': 'result_crabnet.pickle',
        'CGCNN': 'result_cgcnn.pickle',
        'MEGNet': 'result_megnet.pickle'
        }):
    
    # whether getting score for -S models
    struct = any(['-S' in value for value in score_files.values()])
    
    file = {k: pd.DataFrame(pickle.load(open(v, 'rb'))) for k, v in score_files.items()}
    
    result = []
    for i, (k,v) in enumerate(file.items()):
        score = np.vstack((v['score'][i] for i in range(len(v['score']))))
        v['score_num'] = v['score'].apply(lambda x: x.shape[0])
        v = v.drop(['score',], axis=1)
        df = v.loc[v.index.repeat(v['score_num'])]
        df['MAE'] = score[:,0]
        df['reference MAE'] = score[:,1]
        df['model'] = k
        result.append(df)
    result = pd.concat(result)
    
    keys = list(score_files.keys())
    
    score_dict = {}
    for prop in ('formation_energy_per_atom', 'band_gap', 'density', 'elasticity.K_VRH',
                 'elasticity.G_VRH', 'point_density'):
        df = result[result['property']==prop]
        
        # Process point_density
        if prop == 'point_density':
            df['MAE'] = df['MAE'].apply(lambda x: x*1000)
            
        # Get rid of outliers
        if (prop == 'point_density') & struct:
            df = df.reset_index().drop(index=129).set_index('index')
        
        sorterIndex = dict(zip(keys, range(len(keys))))
        df['model index'] = df['model'].map(sorterIndex)
        df = df.sort_values(by=['data segregation','model index'], ascending=[False, False])
        
        df_min = df[['data segregation', 'model', 'MAE']].groupby(['data segregation', 'model']).min()
        df_max = df[['data segregation', 'model', 'MAE']].groupby(['data segregation', 'model']).max()
        df_mean = df[['data segregation', 'model', 'MAE']].groupby(['data segregation', 'model']).mean()
        
        df['MAE_min'] = df_min.loc[pd.MultiIndex.from_arrays([
            df['data segregation'].values,
            df.model.values,
            ])]['MAE'].values
        df['MAE_max'] = df_max.loc[pd.MultiIndex.from_arrays([
            df['data segregation'].values,
            df.model.values,
            ])]['MAE'].values
        df['MAE_mean'] = df_mean.loc[pd.MultiIndex.from_arrays([
            df['data segregation'].values,
            df.model.values,
            ])]['MAE'].values
        
        df['MAE_min'] = df['MAE_mean'] - df['MAE_min']
        df['MAE_max'] = df['MAE_max'] - df['MAE_mean']
        
        df_dict = {key: df[df.model==key] for key in keys}
        
        yerr = {}
        for key in keys:
            df_temp = df_dict[key][['data segregation', 'MAE_min', 'MAE_max']].drop_duplicates().T
            df_temp.columns = df_temp.iloc[0]
            df_temp = df_temp[1:]
            yerr[key] = df_temp
        
        data_len = df_dict['CGCNN'].drop_duplicates(subset=['data segregation','property'])[['data segregation', 'ndata_train']]
        data_len['ndata_train'] = data_len['ndata_train'].astype('int')
        data_len = data_len.set_index('data segregation',)
        
        score_dict[prop] = (df_dict, yerr, data_len)
        
    return score_dict

# Used for plotting the stable/unstable pie charts in Figure S1
def get_ratio():
    file_names = ('E_f.h5', 'E_g.h5', 'K_VRH.h5', 'G_VRH.h5', 
                  'rho.h5', 'point_density.h5')
    props = ('formation_energy_per_atom', 'band_gap', 'elasticity.K_VRH',
             'elasticity.G_VRH', 'density', 'point_density')
    
    result = {}
    for file_name, prop in zip(file_names, props):
        
        df = pd.read_hdf(f'../data/megnet/{file_name}', key='data')
        df = df.rename({prop: 'prop'}, axis='columns')
        
        if prop == 'point_density':
            df['prop'] = df['prop'].apply(lambda x: x*1000)
        
        df_non_poly = deepcopy(df[df['npoly']==0][['pretty_formula', 'prop', 'e_above_hull']])
        df_non_poly['stab'] = df_non_poly['e_above_hull'].apply(lambda x: 'stable' if x <= 0.1 else 'unstable')
        df_non_poly = df_non_poly.drop(columns='e_above_hull')
        
        df_poly = deepcopy(df[df['npoly']!=0][['pretty_formula', 'prop', 'e_above_hull']])
        df_poly['stab'] = df_poly['e_above_hull'].apply(lambda x: 'stable' if x <= 0.1 else 'unstable')
        df_poly = df_poly.drop(columns='e_above_hull')
        
        result[('non-poly', prop)] = df_non_poly
        result[('poly', prop)] = df_poly
    return result