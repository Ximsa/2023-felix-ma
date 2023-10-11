from toolz.dicttoolz import valmap, keyfilter

import torch
import itertools
import pandas
import multiprocess
import numpy as np
from networkx import pagerank
from torch_geometric.utils import to_networkx
from functools import partial
from itertools import takewhile
from toolz.itertoolz import iterate, first, concat, cons
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge, valmap, keyfilter, get_in, merge_with


from models import GPN_Encoder
import datasets
import main


def run_funs_parallel(funs):
    with multiprocess.Pool(processes=multiprocess.cpu_count()) as pool:
        actions = [pool.apply_async(fun) for fun in funs]
        return [action.get() for action in actions]

def run_config(dataset_names, samplers, budget, seed, repeats, average_repeats, hyperparameters):
    """
    Runs experiments as described in given run config
    :returns: dataframe with run statistics
    """
    def run_one_experiment(dataset, dataset_name, sampler_name, hyperparams):
        rank = torch.tensor(list(pagerank(to_networkx(dataset)).values()))
        gpn_model = partial(GPN_Encoder,
                            num_node_features=dataset.num_node_features,
                            num_classes=dataset.num_classes,
                            pagerank_scores=rank,
                            **keyfilter(lambda x: x in ['hidden_dim_size',
                                                        'embedding_dim',
                                                        'dropout',
                                                        'distance_loss_weight'],
                                        hyperparams))
        run_results = main.run(model=gpn_model,
                               dataset=dataset,
                               sampler=sampler_name,
                               runs=repeats,
                               budget=budget,
                               seed=seed,
                               **keyfilter(lambda x: x in ['train_epochs',
                                                           'learning_rate'],
                                           hyperparams))
        # store results
        intermediate_results = []
        for result in run_results:
            for i in range(len(result['test']['accuracy'])):
                row = merge({'dataset_name': dataset_name,
                             'sampler_name': sampler_name,
                             'budget': result['budget_used'][i]},
                            hyperparams,
                            {'test_accuracy': result['test']['accuracy'][i],
                             'test_macro_f1': result['test']['macro-f1'][i],
                             #'test_confusion': result['test']['confusion'][i],
                             'train_accuracy': result['train']['accuracy'][i],
                             'train_macro_f1': result['train']['macro-f1'][i],
                             #'train_confusion': result['train']['confusion'][i]
                             })
                intermediate_results.append(row)
        return intermediate_results
    torch.set_num_threads(1) # limit torch to 1 thread for multiprocessing efficiency increase
    results = []
    funs = []
    # queue up jobs
    for dataset_name, sampler_name in itertools.product(dataset_names, samplers):
        dataset = datasets.get_dataset(dataset_name)
        keys, values = zip(*hyperparameters.items())
        for bundle in itertools.product(*values):
            config = dict(zip(keys, bundle))
            funs.append(partial(run_one_experiment, dataset, dataset_name, sampler_name, config))
    print(str(len(funs)) +" jobs to be started")
    results.append(run_funs_parallel(funs))
    results = pandas.DataFrame.from_records(np.array(results).flatten())
    # check if averaging is needed
    if average_repeats:
        grouping = list(takewhile(lambda x: x != "test_accuracy", results.columns))
        results = results.groupby(grouping).agg(["mean", "std"])
    return results

def load_and_run_config(filename):
    with open(filename, "r") as f:
        config = yaml.load(f)
        print(config)
        return run_config(**config)

dataset = datasets.get_dataset('Cora')

example_run_config = {
    'dataset_names': ['Cora'],
    'samplers': ['model','kmedoids'],#,'kmedoids','pagerank','random'],
    'budget': 14,
    'seed': 3133742069,
    'repeats': 2,
    'average_repeats': True,
    'hyperparameters':{
        'embedding_dim': [2,16],
        'hidden_dim_size': [128],
        'dropout': [0.6],
        'distance_loss_weight': [0.5],
        'train_epochs': [6],
        'learning_rate': [0.005]}}

config = {
    'dataset_names': ['Cora'],
    'samplers': ['model'],
    'budget': 98,
    'seed': 3133742069,
    'repeats': 10,
    'average_repeats': True,
    'hyperparameters':{
        'embedding_dim': [16,20,24],
        'hidden_dim_size': [128,196,256],
        'dropout': [0.4,0.6,0.8],
        'distance_loss_weight': [0.25, 0.5, 1, 2],
        'train_epochs': [4,6,8],
        'learning_rate': [1/100,1/200,1/500,1/1000]}}

result = run_config(**config)
result.to_csv("results.csv", sep=";")


