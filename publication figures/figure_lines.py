import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy

# Combine results for every model
def score_combine_ndata(directory):
    for root, dirs, files in os.walk(directory):
        break
    # Get keys in the .pickle files
    for file in files:
        if ('.pickle' in file):
            keys = pickle.load(open(os.path.join(root, file), 'rb')).keys()
            break
    temp = {key: [] for key in keys}
    temp['ndata'] = []
    for file in files:
        if ('.pickle' in file):
            d = pickle.load(open(os.path.join(root, file), 'rb'))
            for key in keys:
                temp[key] = temp[key]+[d[key]]
            ndata = file.rstrip('.pickle').split('_')[-1]
            temp['ndata'].append(ndata)
    
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return {key: locs for key,locs in tally.items() 
                                if len(locs)>1}
    
    d = list_duplicates(zip(temp['data segregation'], temp['property'], temp['ndata']))
    
    indices = []
    for value in d.values():
        temp['score'][value[0]] = np.stack((temp['score'][v] for v in value))
        indices = indices + value[1:]
    
    keys = temp.keys()
    
    for key in keys:
        for index in sorted(indices, reverse=True):
            del temp[key][index]
    
    model, case = directory.split('/')[-2:]
    
    return pickle.dump(temp, open(f"result_{directory.split('/')[-1]}{struct_name[struct]}{directory.split('/')[-2]}.pickle", 'wb'))

for case in ('ndata', 'stability'):
    for model in ('cgcnn', 'crabnet', 'roost'):
        score_combine(f'../results/{case}/{model}')