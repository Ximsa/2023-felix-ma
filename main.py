#imported libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from networkx import pagerank
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import random
import copy
import matplotlib
import itertools
import pandas
import operator
import numpy as np
from functools import partial
from toolz.itertoolz import iterate, first, concat, cons
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge, valmap, keyfilter, get_in, merge_with
import matplotlib.pyplot as plt

#own libraries
import datasets
import sampling
from models import GPN_Encoder, GCN
from util import cond, plot_embeddings, plot_clusterer

def accuracy(predictions, true_labels, mask):
    """Calculates accuracy, macro-f1 and the confusion matrix

    :param predictions: Predicted labels
    :param true_labels: Ground truth
    :param mask: Mask for instance selection
    :returns: Dict of accuracy scores (acc, macro-f1, and confusion matrix)
    """
    return {"accuracy": accuracy_score(predictions[mask], true_labels[mask]),
            "macro-f1": f1_score(true_labels[mask], predictions[mask], average='macro'),
            "confusion": confusion_matrix(true_labels[mask], predictions[mask])}

def train(n, optimizer, model, dataset, early_stopping=False): #TODO: implement early stopping
    """Trains and alters given model for n epochs

    :param n: Number of epochs
    :param optimizer: Optimizer
    :param model: Model to train on
    :param dataset: Dataset with train/validation/test split
    :param early_stoping: stop training when accuracy gets too high to prevent overfitting
    :returns: Dictionary of train and test statistics
    """
    num_labels = len(dataset.y.unique())
    model.train()
    train_stats = {"accuracy": [],
                   "macro-f1": [],
                   "confusion": []}
    test_stats = {"accuracy": [],
                   "macro-f1": [],
                   "confusion": []}
    for i in range(n):
        optimizer.zero_grad()
        probabilities = model(dataset.x, dataset.edge_index)
        loss = model.loss(dataset, probabilities)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(probabilities, dim=1)
        def combine_training_stats(x):
            return x[0] + [x[1]]
        train_stats = merge_with(combine_training_stats, train_stats, accuracy(predictions, dataset.y, dataset.train_mask))
        test_stats = merge_with(combine_training_stats, test_stats, accuracy(predictions, dataset.y, dataset.test_mask))
    return train_stats, test_stats

def run(model, dataset, sampler='model', runs=10, budget=100, seed=133742069, train_epochs=16, learning_rate=0.001):
    """Runs experiments on given model "runs" times with the same settings

    :param model: Model constructor with 0 args for model construction
    :param dataset: Data for training, testing, and validation
    :param sampler: Sampler used for the active learner
    :param runs: Number of experiment repeats
    :param budget: Number of labels moving from validation to training
    :param seed: Initial seed for each strategy. Seed will change each run
    :param train_epochs: Number of epochs to train between samling
    :param learning_rate: Learning rate of the optimizer
    :returns: run statistics
    """
    def run_once(model, dataset, budget, learning_rate):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        classifier = cond(sampler, # initialize classifier if needed
                          ("kmedoids", "model"), lambda: KMedoids(
                              n_clusters=dataset.num_classes,
                              init='k-medoids++').fit(
                                  dataset.x[torch.logical_or(dataset.train_mask,dataset.val_mask)]),
                          lambda: None)()
        sampler_fun = sampling.sampler[sampler]
        full_train_stats = {"accuracy": [],
                            "macro-f1": [],
                            "confusion": []}
        full_test_stats = {"accuracy": [],
                           "macro-f1": [],
                           "confusion": []}
        while(budget > 0):
            # ask active learner for vertices
            sampled_indices = sampler_fun(min(budget, dataset.num_classes), model, dataset, classifier)
            budget -= len(sampled_indices)
            # move sampled vertices from the validation to the training set
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            train_stats, test_stats = train(train_epochs, optimizer, model, dataset)
            def combine_training_stats(x):
                return x[0] + x[1]
            full_train_stats = merge_with(combine_training_stats, full_train_stats, train_stats)
            full_test_stats = merge_with(combine_training_stats, full_test_stats, test_stats)
        return {"train": full_train_stats,
                "test": full_test_stats}
    # perform experiments
    results = []
    for i in range(runs):
        print(seed)
        torch.manual_seed(seed) # update seeds
        np.random.seed(seed)
        random.seed(seed)
        dataset.train_mask, dataset.val_mask, dataset.test_mask = datasets.create_split(dataset, seed=seed) # update splits with given seed
        results.append(run_once(model(), copy.deepcopy(dataset), budget, learning_rate))
        seed = random.randrange(2**31) # generate seed for next run
    return results

def run_config(dataset_names, samplers, budget, seed, repeats, hyperparameters):
    """
    Runs experiments as described in given run config
    :returns: dataframe with run statistics
    """
    results = pandas.DataFrame(columns=['dataset_name',
                                        'sampler_name',
                                        'embedding_dim',
                                        'hidden_dim_multiplier',
                                        'dropout',
                                        'distance_loss_weight',
                                        'train_epochs',
                                        'learning_rate',
                                        'train_accuracy',
                                        'train_macro_f1',
                                        'test_accuracy',
                                        'test_macro_f1',
                                        'test_confusion',])
    for dataset_name, sampler_name in zip(dataset_names, samplers):
        dataset = datasets.get_dataset(dataset_name)
        keys, values = zip(*hyperparameters.items())
        for bundle in itertools.product(*values):
            config = dict(zip(keys, bundle))
            rank = torch.tensor(list(pagerank(to_networkx(dataset)).values()))
            gpn_model = partial(GPN_Encoder,
                                num_node_features=dataset.num_node_features,
                                num_classes=dataset.num_classes,
                                pagerank_scores=rank,
                                **keyfilter(lambda x: x in ['hidden_dim_multiplier',
                                                            'embedding_dim',
                                                            'dropout',
                                                            'distance_loss_weight'],
                                            config))
            result = run(model=gpn_model,
                         dataset=dataset,
                         sampler=sampler_name,
                         runs=repeats,
                         budget=budget,
                         seed=seed,
                         **keyfilter(lambda x: x in ['train_epochs',
                                                     'learning_rate'],
                                     config))
            # todo: average runs
            result = result[0]
            # insert result into dataframe
            results.loc[len(results)] = [dataset_name,
                                         sampler_name,
                                         hyperparameters['embedding_dim'],
                                         hyperparameters['hidden_dim_multiplier'],
                                         hyperparameters['dropout'],
                                         hyperparameters['distance_loss_weight'],
                                         hyperparameters['train_epochs'],
                                         hyperparameters['learning_rate'],
                                         result['train']['accuracy'],
                                         result['train']['macro-f1'],
                                         result['test']['accuracy'],
                                         result['test']['macro-f1'],
                                         result['test']['confusion']]
    return results

dataset = datasets.get_dataset('Cora')

example_run_config = {
    'dataset_names': ['Cora'],
    'samplers': ['random'],
    'budget': 80,
    'seed': 3133742069,
    'repeats': 1,
    'hyperparameters':{
        'embedding_dim': [16],
        'hidden_dim_multiplier': [8],
        'dropout': [0.5, 0.8],
        'distance_loss_weight': [0.5, 1, 2],
        'train_epochs': [20],
        'learning_rate': [1e-3, 1e-4]}}
run_config(**example_run_config)

# prototypical
rank = torch.tensor(list(pagerank(to_networkx(dataset)).values()))
gpn_model = partial(GPN_Encoder,
                    num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    pagerank_scores=rank,
                    embedding_dim=16,
                    dropout=0.5)
gcn_model = partial(GCN,
                    in_channels=dataset.num_node_features,
                    hidden_channels=128,
                    num_layers=2,
                    out_channels=dataset.num_classes,
                    dropout=0.5)

results_gpn = {}
results_gcn = {}
for sampler_name in sampling.sampler.keys():
    results_gpn[sampler_name] = run(model=gpn_model, dataset=dataset, sampler=sampler_name,runs=3, budget=100)
    continue
    results_gcn[sampler_name] = run(model=gcn_model, dataset=dataset, sampler=sampler_name,runs=5, budget=100)
    print(sampler_name + "\tgpn f1: " + str(sum(map(lambda d: d['test']['macro-f1'], results_gpn[sampler_name])) / len(results_gpn[sampler_name])))
    print(sampler_name + "\tgcn f1: " + str(sum(map(lambda d: d['test']['macro-f1'], results_gcn[sampler_name])) / len(results_gcn[sampler_name])))

result = results_gpn[sampler_name]
result
"""

#embeddings = result_gcn[0]['model'](dataset.x, dataset.edge_index)
model = results_gpn['random'][0]['model']
modified_dataset = results_gpn['random'][0]['dataset']
embeddings = model.embeddings
prototypes = model.prototypes
probabilities = model(dataset.x, dataset.edge_index).detach()
num_classes = dataset.y.unique().size(0)
plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                labels=torch.cat([dataset.y, torch.full([num_classes], num_classes)]))
plot_embeddings(dataset.x, dataset.y)
"""
