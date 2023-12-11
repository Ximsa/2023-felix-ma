import sys
import time
import torch
#torch.cuda.is_available = lambda : False # for multiprocessing to work
import itertools
import pandas
import edn_format
import numpy as np
from networkx import pagerank
from torch_geometric.utils import to_networkx
from functools import partial
from itertools import takewhile
from toolz.itertoolz import iterate, first, concat, cons
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge, valmap, keyfilter, get_in, merge_with
import models
import datasets
import main
from util import select_keys

def run_config(model_names, dataset_names, samplers, budget, seed, repeats, hyperparameters):
    """
    Runs experiments as described in given run config
    :returns: dataframe with run statistics
    """
    def run_one_experiment(model_constructor, dataset, dataset_name, sampler_name, hyperparams):
        model = partial(model_constructor,
                        dataset,
                        **select_keys(hyperparams,
                                      (['hidden_dim_size',
                                        'dropout',
                                        'distance_loss_weight'])))
        run_results = main.run(model=model,
                               dataset=dataset,
                               sampler=sampler_name,
                               runs=repeats,
                               label_propagation_uncertainty_treshold=hyperparams['label_propagation_uncertainty_treshold'],
                               budget=budget,
                               seed=seed,
                               learning_rate=hyperparams['learning_rate'],
                               entropy_pagerank_weighting=hyperparams['entropy_pagerank_weighting'])
        # append additional run information to results
        return list(map(partial(merge,
                                {'dataset_name': dataset_name,
                                 'sampler_name': sampler_name,},
                                hyperparams),
                        run_results))
    results = {}
    # perform jobs
    for model_name, dataset_name, sampler_name in itertools.product(model_names, dataset_names, samplers):
        dataset = datasets.get_dataset(dataset_name)
        model_constructor = models.models[model_name]
        keys, values = zip(*hyperparameters.items())
        for bundle in itertools.product(*values):
            config = dict(zip(keys, bundle))
            result = run_one_experiment(model_constructor,
                                        dataset,
                                        dataset_name,
                                        sampler_name,
                                        config)
            result = pandas.DataFrame.from_records(result)
            name = "-".join([dataset_name, model_name, sampler_name] + list(map(str, config.values())))
            results[name] = (dataset_name, result)
    return results

def load_and_run_config(filename):
    with open(filename, "r") as f:
        contents = f.read()
        config = edn_format.loads(contents)
        print(contents)
        return run_config(**config)

if __name__ == "__main__":
    results = load_and_run_config(sys.argv[1])
    for name, (dataset_name, result) in results.items():
        filename = "results/"+ dataset_name + "/" + name + ".csv"
        csv = result.to_csv(sep=";", index=False)
        # postprocessing: seperate runs by empty rows
        csv = [line if line.find(";;") == -1 else "\n" for line in csv.splitlines()]
        f = open(filename, "w")
        f.write("\n".join(csv))
        f.close()

"""
    #dataset = datasets.get_dataset('Cora')

    example_run_config = {
        'dataset_names': ['Cora'],
        'samplers': ['model','kmedoids'],
        'budget': 14,
        'seed': 3133742069,
        'repeats': 1,
        'average_repeats': True,
        'hyperparameters':{
            'hidden_dim_size': [128],
            'dropout': [0.6],
            'distance_loss_weight': [0.5],
            'train_epochs': [6],
            'learning_rate': [0.005]}}
config = {
    'model_names': ['GPN-GCN'],
    'dataset_names': ['Cora'],
    'samplers': ['own'],
    'budget': 42,
    'seed': 3133742069,
    'repeats': 1,
    'hyperparameters':{
        'entropy_pagerank_weighting': [0.5],
        'label_propagation_uncertainty_treshold': [0.2],
        'hidden_dim_size': [64],
        'dropout': [0.5],
        'distance_loss_weight': [1],
        'learning_rate': [0.005]}}
    
results = run_config(**config)
for name, result in results.items():
    result.to_csv(name + ".csv", sep=";")

# hyperparams:
# lr 0.01, 0.005, 0.001
# distance loss weight: 0.5 1 2
# 
"""
