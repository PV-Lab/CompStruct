import argparse, os, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
from utils.utils_megnet import Data, set_seed, filter_valid
from utils.utils import plot, get_prop_d

parser = argparse.ArgumentParser(description='MatErials Graph Network (MEGNet)')

parser.add_argument('-S', '--struct', action='store_true',
                    help='run structure-only input (dummy Comp inputs, \
                    i.e., all atoms regarded as hydrogen)')
parser.add_argument('-ds', '--data-segregation', nargs="+", type=str, metavar='STR',
                    default=['stable', 'unstable', 'poly', 'non-poly'],
                    help="data segregation to be run (default: ['stable', 'unstable', \
                    'poly', 'non-poly'])")
parser.add_argument('-p', '--property', nargs="+", type=str, metavar='STR',
                    default=['formation_energy_per_atom', 'band_gap', 'density', 
                             'elasticity.K_VRH', 'elasticity.G_VRH', 'point_density'],
                    help="property to be run (default: ['formation_energy_per_atom', \
                    'band_gap', 'density', 'elasticity.K_VRH', 'elasticity.G_VRH', \
                    'point_density'])")
parser.add_argument('--data-seed', nargs="+", type=int, metavar='INT',
                    default=[10, 20, 30],
                    help="data seeds (for train/validation/test splits) to be run (default: [10, 20, 30])")
parser.add_argument('--model-seed', nargs="+", type=int, metavar='INT',
                    default=[1, 2, 3],
                    help="model seeds (for model initialization) to be run (default: [1, 2, 3])")
parser.add_argument('--epochs', default=1000, type=int, metavar='INT',
                    help='number of total epochs to run (default: 1000)')
args = parser.parse_args()

prop_d = get_prop_d()

for ds in args.data_segregation:
    for prop in args.property:
        for data_seed in args.data_seed:
            for model_seed in args.model_seed:
                
                resultdir = f"{ds}_{prop_d[prop]}_{data_seed}_{model_seed}"
                if args.struct:
                    resultdir += "-S"
                callbackdir = "callback/megnet/" + resultdir
                modeldir = "models/megnet/" + resultdir
                plotdir = "plots/megnet/" + resultdir
                
                # Check whether certain results has already been run
                if os.path.isfile(f'score/megnet/{resultdir}.pickle'):
                    print('{ds}_{prop_d[prop]}_{data_seed}_{model_seed} has already been run.')
                    pass
                else:
                    # Load data for specfic data segregation and property
                    print('Loading dataset...')
                    if not args.struct:
                        data = Data(f'data/megnet/{prop_d[prop]}.h5')
                    else:
                        data = Data(f'data/megnet/{prop_d[prop]}-S.h5')
                    print('Dataset loaded')
                    data.dataset(ds, prop)
                    
                    def train(train_index, val_index, test_index, callback_dir, epochs=args.epochs, 
                              graph_converter=CrystalGraph(cutoff=4, 
                                bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5))
                              ):
                
                        X_train = data.X[train_index]
                        X_val = data.X[val_index]
                        X_test = data.X[test_index]
                        y_train = data.y[train_index]
                        y_val = data.y[val_index]
                        y_test = data.y[test_index]
                
                        X_train, y_train, index_train_valid, index_train_invalid = filter_valid(
                            X_train, y_train, graph_converter)
                        X_val, y_val, index_val_valid, index_val_invalid = filter_valid(
                            X_val, y_val, graph_converter)
                        X_test, y_test, index_test_valid, index_test_invalid = filter_valid(
                            X_test, y_test, graph_converter)
                        
                        set_seed(model_seed)
                        model = MEGNetModel(
                            nfeat_edge = 100,
                            nfeat_global = 2,
                            nblocks = 3,
                            lr = 1e-3,
                            n1 = 64,
                            n2 = 32,
                            n3 = 16,
                            nvocal = 95,
                            embedding_dim = 16,
                            graph_converter=graph_converter,
                        )
                        model.train_from_graphs(X_train, y_train, X_val, y_val, 
                                                epochs=epochs, batch_size=128, 
                                                save_checkpoint=True, automatic_correction=True, 
                                                dirname=callback_dir)
                        
                        y_train_hat = model.predict_graphs(X_train)
                        y_val_hat = model.predict_graphs(X_val)
                        y_test_hat = model.predict_graphs(X_test)
                        
                        # Plot parity plot (predicted vs actual) for the test set
                        plot(y_test, y_test_hat, prop, plotdir)
                        
                        MAE = np.mean(np.abs(y_test - y_test_hat), axis=0)
                        # dummy MAE for reference as in https://www.nature.com/articles/s41524-020-00406-3
                        MAE_ref = np.mean(np.abs(y_test - np.mean(y_test)), axis=0)
                        
                        pred = {
                            'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 
                            'y_train_hat': y_train_hat, 'y_val_hat': y_val_hat, 'y_test_hat': y_test_hat
                        }
                        
                        return (MAE, MAE_ref), model, pred
                    
                    # Split train, validation, test data with 60%, 20%, 20%
                    train_all_index, test_index = train_test_split(np.arange(len(data.y)), 
                                                                   test_size=0.2, 
                                                                   random_state=data_seed)
                    train_index, val_index = train_test_split(train_all_index, 
                                                              test_size=0.25, 
                                                              random_state=data_seed)
                    # Initialize things
                    result = {}
                    os.makedirs(callbackdir, exist_ok=True)
                    os.makedirs('plots/megnet', exist_ok=True)
                    os.makedirs('results/megnet/prediction', exist_ok=True)
                    # Train and save
                    s, model, pred = train(train_index, val_index, test_index, callbackdir)
                    model.save_model(modeldir)
                    
                    result['data segregation'] = ds
                    result['property'] = prop
                    result['score'] = s
                    
                    pickle.dump(result, open(f'results/megnet/{resultdir}.pickle', 'wb'))
                    pickle.dump(pred, open(f'results/megnet/prediction/{resultdir}.pickle', 'wb'))
