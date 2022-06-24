import numpy as np
import random
import os
import torch

def set_seed(seed):
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # for torch seed
    torch.manual_seed(seed)