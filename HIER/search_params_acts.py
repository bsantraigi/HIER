"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import argparse
from functools import partial
from main_acts import *

import optuna


parser = argparse.ArgumentParser() 

# parser.add_argument("-embed", "--embedding_size", default=100, type=int, help = "Give embedding size") #
# parser.add_argument("-heads", "--nhead", default=4, type=int,  help = "Give number of heads") #
# parser.add_argument("-hid", "--nhid", default=100, type=int,  help = "Give hidden size") #

# parser.add_argument("-l_e1", "--nlayers_e1", default=3, type=int,  help = "Give number of layers for Encoder 1") #
# parser.add_argument("-l_e2", "--nlayers_e2", default=3, type=int,  help = "Give number of layers for Encoder 2") #
# parser.add_argument("-l_d", "--nlayers_d", default=3, type=int,  help = "Give number of layers for Decoder")

# parser.add_argument("-d", "--dropout",default=0.2, type=float, help = "Give dropout") # 
parser.add_argument("-bs", "--batch_size", default=8, type=int, help = "Give batch size") #
parser.add_argument("-e", "--epochs", default=3, type=int, help = "Give number of epochs") #
parser.add_argument("-model", "--model_type", default="SET++", help="Give model name one of [SET++, HIER++]")

args = parser.parse_args()

class ARGS:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __str__(self):
        return str(self.__dict__)
        
def define_args(main_args, trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    args2 = {
#         'embedding_size': trial.suggest_int("embedding_size", 50, 400),        
        'nhead': trial.suggest_int("nhead", 2, 8),
        'embedding_perhead': trial.suggest_int("embedding_perhead", 25, 40),
        'nhid_perhead': trial.suggest_int("nhid_perhead", 10, 40),
#         'nhid': trial.suggest_int("nhid", 50, 400),
        'nlayers_e1': trial.suggest_int("nlayers_e1", 2, 6),
        'nlayers_e2': trial.suggest_int("nlayers_e2", 2, 6),
        'nlayers_d': trial.suggest_int("nlayers_d", 2, 6),
        'dropout': trial.suggest_float("dropout", 0.05, 0.8),
        'batch_size': main_args.batch_size,
        'epochs': main_args.epochs,
        'model_type': main_args.model_type
    }
    
    # following need to be divisible by nhead
    args2['embedding_size'] = args2['embedding_perhead']*args2['nhead']
    args2['nhid'] = args2['nhid_perhead']*args2['nhead']    
    
    args = ARGS(**args2)
    print(args)
    return args



def objective(main_args, trial):
    print("\n\n===>", trial)
    # Generate the model.
    args = define_args(main_args, trial)

    # Generate the optimizers.
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the dataset.
    #train_loader, valid_loader = get_mnist()

    # Training of the model
    
    def callback(epoch, val_accuracy): # NewMethod
        trial.report(val_accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # NewMethod
    val_criteria = run(args, optuna_callback=callback) # callback should be called internally after each epoch.
    
    return val_criteria


if __name__ == "__main__":
    """GLOBALS
    """
    study = optuna.create_study(study_name='hier-study', direction="maximize", storage=f'sqlite:///{args.model_type.lower()}.db', load_if_exists=True)
    study.optimize(partial(objective, args), n_trials=30)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html('search_history.html')
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html('search_parallel.html')
