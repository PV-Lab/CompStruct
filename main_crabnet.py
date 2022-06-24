import os
import numpy as np
import pandas as pd
import torch
import argparse

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from crabnet.utils.get_compute_device import get_compute_device
from utils.utils import plot, get_prop_d

import pickle

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


def get_model(data_dir, file_name, model_name, classification=False, batch_size=None,
              transfer=None, verbose=True, epochs=40):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{model_name}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{model_name}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'{data_dir}/{file_name}'
    try:
        val_data = f'{data_dir}/{file_name.rstrip(".csv")}_val.csv'
    except:
        print('Please ensure you have inputted validation data')

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=epochs, losscurve=True)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network(model_name)
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'data/crabnet'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(data_dir, file_name, model_name, classification, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{model_name}', verbose=verbose)
    model.load_network(f'{model_name}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'{data_dir}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(data_dir, file_name, model_name, classification, verbose=True, save_csv=True):
    model = load_model(data_dir, file_name, model_name, classification, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'{model_name} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'{model_name} mae: {mae:0.3g}')
    
    if save_csv:
        # save predictions to a csv
        fname = f'{file_name.replace(".csv", "")}_hat.csv'
        to_csv(output, fname)
    
    return output[0], output[1]


# if __name__ == '__main__':

parser = argparse.ArgumentParser(description='Compositionally-Restricted Attention-Based Network (CrabNet)')

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
parser.add_argument('--epochs', default=1000, type=int, metavar='INT',
                    help='number of total epochs to run (default: 1000)')

args = parser.parse_args()

args_d = vars(args)
prop_d = get_prop_d()

for ds in args.data_segregation:
    for prop in args.property:
        for data_seed in args.data_seed:
            
            data_dir="data/crabnet"
            resultdir = f"{ds}_{prop_d[prop]}_{data_seed}"
            datadir = f"data/crabnet/{ds}_{prop_d[prop]}_{data_seed}"
            plotdir = "plots/crabnet/" + resultdir
            
            args_d['data_train'] = f"{resultdir}.csv"
            args_d['data_val'] = f"{resultdir}_val.csv"
            args_d['data_test'] = f"{resultdir}_test.csv"
            args_d['model_name'] = resultdir
            
            # Check whether certain results has already been run
            if os.path.isfile(f'score/crabnet/{resultdir}.pickle'):
                print('{ds}_{prop_d[prop]}_{data_seed} has already been run.')
                pass
            else:
                # Initialize things
                result = {}
                os.makedirs('models/crabnet', exist_ok=True)
                os.makedirs('plots/crabnet', exist_ok=True)
                os.makedirs('results/crabnet/prediction', exist_ok=True)
                
                model = get_model(data_dir, args_d['data_train'], args_d['model_name'], classification=False, verbose=True, epochs=args.epochs)
                y_test, y_test_hat = save_results(data_dir, args_d['data_test'], args_d['model_name'], classification=False, verbose=False, save_csv=True)
                y_train, y_train_hat = save_results(data_dir, args_d['data_train'], args_d['model_name'], classification=False, verbose=False, save_csv=False)
                y_val, y_val_hat = save_results(data_dir, args_d['data_val'], args_d['model_name'], classification=False, verbose=False, save_csv=False)
                plot(y_test, y_test_hat, prop, plotdir)
                
                MAE = np.mean(np.abs(y_test - y_test_hat), axis=0)
                # dummy MAE for reference as in https://www.nature.com/articles/s41524-020-00406-3
                MAE_ref = np.mean(np.abs(y_test - np.mean(y_test)), axis=0)
                s = (MAE, MAE_ref)
                
                pred = {
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 
                    'y_train_hat': y_train_hat, 'y_val_hat': y_val_hat, 'y_test_hat': y_test_hat
                }
                
                
                result['data segregation'] = ds
                result['property'] = prop
                result['score'] = s
                pickle.dump(result, open(f'results/crabnet/{resultdir}.pickle', 'wb'))
                pickle.dump(pred, open(f'results/crabnet/prediction/{resultdir}.pickle', 'wb'))