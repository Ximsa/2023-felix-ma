#imported libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from networkx import pagerank
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import random
import copy
import matplotlib
import itertools
import pandas
import operator
import yaml
import numpy as np
import math
from functools import partial
from itertools import takewhile
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

# TODO: peforms worse than normal training
def few_shot_training(n, optimizer, model, dataset):
    """
    Trains and alters given model using few shot learning

    :param n: Number of few-shot repeats
    :param optimizer: Optimizer
    :param model: Model to train on
    :param dataset: Dataset with train/validation/test split
    :returns: Dictionary of train and test statistics
    """
    def get_support_query_validation_indices(dataset,
                                             target_support_size=4,
                                             validation_ratio=0.2,
                                             query_min_count=4):
        """
        Samples a sensible support/query/validation split for given dataset

        :param dataset: Dataset containing labels and masks
        :param target_support_size: Support set size
        :param validation_ratio: validation ratio per class
        :param query_min_count: Query set size
        :returns: Indices for support/query/validation splits
        """
        vertices = torch.stack([dataset.y,torch.arange(len(dataset.y))]).T[dataset.train_mask]
        buckets = {}
        support = []
        query = []
        validation = []
        for label, index in vertices:
            label = label.item()
            index = index.item()
            if label in buckets:
                buckets[label].append(index)
            else:
                buckets[label] = [index]
        for label, indices in buckets.items():
            bucket_size = len(indices)
            # support set ranges between 0 and 3 items
            support_size = min(target_support_size, bucket_size)
            # only add validation if enough samples are availible
            validation_size = math.floor(bucket_size * validation_ratio)
            random.shuffle(indices)
            #print(indices, bucket_size, support_size, validation_size)
            temp_validation = indices[:validation_size]
            indices = indices[validation_size:]
            # add some support indices to the query set, if query set is too small
            temp_support = indices[:support_size]
            temp_query = indices[support_size:] # remainder is query set
            if len(temp_query) < query_min_count:
                temp_query += random.sample(temp_support,
                                            min(len(temp_support), query_min_count-len(temp_query)))
            # add one support index to validation, if validation is empty
            validation += temp_validation if len(temp_validation) > 0 else random.sample(temp_support,1)
            support += temp_support
            query += temp_query
        return support, query, validation
    
    def resample_query_support(query_indices,support_indices):
        """
        resamples query and support
        """
        query_len = len(query_indices)
        support_len = len(support_indices)
        indices = list(set(query_indices + support_indices))
        random.shuffle(indices)
        return indices[query_len:], indices[-support_len:]
    
    labels = None
    best_acc = 0
    no_increment_count = 0
    foo = []
    best_model_state = model.state_dict()
    for _ in range(10):
        accs = []
        support, query, validation = get_support_query_validation_indices(dataset)
        acc = 0
        model.train()
        for _ in range(n):
            support, query = resample_query_support(support, query)
            optimizer.zero_grad()
            logits = model(dataset.x, dataset.edge_index) # calculating logits updates the embeddings
            loss = model.loss(dataset, logits, support, query)
            loss.backward()
            optimizer.step()
        model.eval()
        labels = model(dataset.x, dataset.edge_index).argmax(dim=1)
        acc = accuracy_score(labels[validation], dataset.y[validation])
        foo.append(acc)
        if acc > best_acc:
            best_model_state = model.state_dict()
            best_acc = acc
            no_increment_count = 0
        else:
            if no_increment_count < 5:
                no_increment_count += 1
            else:
                break
    model.load_state_dict(best_model_state)
    return (accuracy(labels, dataset.y, dataset.train_mask), # train acc
            accuracy(labels, dataset.y, dataset.test_mask)) # test acc

def train(n, optimizer, model, dataset):
    """
    Trains and alters given model for n epochs

    :param n: Number of epochs
    :param optimizer: Optimizer
    :param model: Model to train on
    :param dataset: Dataset with train/validation/test split
    :returns: Dictionary of train and test statistics
    """
    num_labels = len(dataset.y.unique())
    model.train()
    for i in range(n):
        optimizer.zero_grad()
        logits = model(dataset.x, dataset.edge_index) # getting logits also updates the embeddings
        loss = model.loss(dataset, logits, dataset.train_mask, dataset.train_mask) # no few-shot learning
        loss.backward()
        optimizer.step()
    model.eval()
    logits = model(dataset.x, dataset.edge_index)
    predictions = torch.argmax(logits, dim=1)
    return (accuracy(predictions, dataset.y, dataset.train_mask), # train acc
            accuracy(predictions, dataset.y, dataset.test_mask)) # test acc

def run(model,
        dataset,
        sampler='model',
        runs=10,
        budget=100,
        seed=133742069,
        train_epochs=16,
        learning_rate=0.001):
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
        initial_budget = budget
        budget_history = []
        while(budget > 0):
            # ask active learner for vertices
            sampled_indices = sampler_fun(min(budget, dataset.num_classes), model, dataset, classifier)
            budget -= len(sampled_indices)
            budget_history.append(initial_budget - budget)
            # move sampled vertices from the validation to the training set
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            train_stats, test_stats = few_shot_training(train_epochs, optimizer, model, dataset)
            #train_stats, test_stats = train(1, optimizer, model, dataset) # "smoothing"
            def combine_training_stats(x): # x[0] is the full training stat, x[1] the new train stat
                return x[0] + [x[1]]
            full_train_stats = merge_with(combine_training_stats, full_train_stats, train_stats)
            full_test_stats = merge_with(combine_training_stats, full_test_stats, test_stats)
        print(full_test_stats['accuracy'][-1])
        return {"budget_used": budget_history,
                "train": full_train_stats,
                "test": full_test_stats}
    # perform experiments
    results = []
    for i in range(runs):
        torch.manual_seed(seed) # update seeds
        np.random.seed(seed)
        random.seed(seed)
        dataset.train_mask, dataset.val_mask, dataset.test_mask = datasets.create_split(dataset, seed=seed) # update splits with given seed
        m = model()
        m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        results.append(run_once(model(), copy.deepcopy(dataset), budget, learning_rate))
        seed = random.randrange(2**31) # generate seed for next run
    return results


"""
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
