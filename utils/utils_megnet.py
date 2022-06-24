import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import random
import os

class Data:
    '''
    for loading MEGNet data
    '''
    def __init__(self, file_directory):
        self.queried_dataframe = pd.read_hdf(file_directory, key='data')
        
    def dataset(self, data_segregation, property_):
        '''
        Select to load a dataset

        Parameters
        ----------
        data_segregation : string
            One of 'stable', 'unstable', 'poly', and 'non-poly'
        property_ : string
            One of 'formation_energy_per_atom', 'band_gap', 'density', 
            'elasticity.K_VRH', 'elasticity.G_VRH', 'point_density'
        '''
        
        def stable():
            self.dataframe = self.queried_dataframe[self.queried_dataframe['e_above_hull']<=0.1]
        def unstable():
            self.dataframe = self.queried_dataframe[self.queried_dataframe['e_above_hull']>0.1]
        def poly():
            self.dataframe = self.queried_dataframe[self.queried_dataframe['npoly']!=0]
        def non_poly():
            self.dataframe = self.queried_dataframe[self.queried_dataframe['npoly']==0]
        
        switcher = {
            'stable': stable,
            'unstable': unstable,
            'poly': poly,
            'non-poly': non_poly,
            }
        
        f = switcher.get(data_segregation, lambda: (_ for _ in ()).throw(KeyError("Only supports stable, unstable, poly, and non-poly.")))
        f()
        
        self.X = self.dataframe['pymatgen_structure'].values
        self.y = self.dataframe[property_].values[:,np.newaxis]

def set_seed(seed):
    '''
    Set seed for tensorflow keras models
    '''
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
        
def filter_valid(X, y, graph_converter):
    '''
    filter valid structures according to the MEGNet github
    
    https://github.com/materialsvirtuallab/megnet/blob/master/README.md#training-a-new-megnetmodel-from-structures
    '''
    
    graphs_valid = []
    index_valid = []
    index_invalid = []
    
    op = tqdm(X)
    for i, s in enumerate(op):
        op.set_description('Filtering valid structures to convert to graphs')
        
        try:
            graph = graph_converter.convert(s)
            graphs_valid.append(graph)
            index_valid.append(i)
        except:
            index_invalid.append(i)
    
    X_valid = graphs_valid
    y_valid = y[index_valid]
    
    return X_valid, y_valid, index_valid, index_invalid

