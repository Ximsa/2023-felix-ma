#imported libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import LabelPropagation
from networkx import pagerank
from scipy.stats import entropy
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

def few_shot_training(optimizer, model, dataset):
    """
    Trains and alters given model using few shot learning

    :param optimizer: Optimizer
    :param model: Model to train on
    :param dataset: Dataset with train/validation/test split
    :returns: Dictionary of train and test statistics
    """
    def get_train_validation_indices(dataset, validation_ratio=0.2):
        """
        Samples a sensible train/validation split for given dataset

        :param dataset: Dataset containing labels and masks
        :param validation_ratio: validation ratio per class
        :returns: indices for support/query/validation splits, support and query are further split by classes
        """
        train_prop_mask = torch.logical_or(dataset.train_mask, dataset.propagated_mask)
        vertices = torch.stack([dataset.y,torch.arange(len(dataset.y))]).T[train_prop_mask]
        buckets = {}
        training = []
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
            # only add validation if enough samples are availible
            validation_size = math.ceil((bucket_size - 1) * validation_ratio)
            random.shuffle(indices)
            validation += indices[:validation_size]
            training += indices[validation_size:]
        return training, validation    
    labels = None
    best_acc = 0
    no_increment_count = 0
    best_model_state = model.state_dict()
    training, validation = get_train_validation_indices(dataset)
    i = 0
    for _ in range(20 if len(validation) != 0 else 4): # train 5 epochs when validation set is empty
        i+=1
        accs = []
        acc = 0
        model.train()
        optimizer.zero_grad()
        logits = model(dataset.x, dataset.edge_index) # calculating logits updates the embeddings
        loss = model.loss(dataset, logits, training)
        loss.backward()
        optimizer.step()
        model.eval()
        labels = model(dataset.x, dataset.edge_index).argmax(dim=1)
        acc = accuracy_score(labels[validation], dataset.y[validation])
        if acc > best_acc:
            best_model_state = model.state_dict()
            best_acc = acc
            no_increment_count = 0
        else:
            if no_increment_count < 4:
                no_increment_count += 1
            else:
                break
    model.load_state_dict(best_model_state)
    return (accuracy(labels, dataset.y, dataset.train_mask), # train acc
            accuracy(labels, dataset.y, dataset.test_mask)) # test acc

def label_propagation(model, dataset, steps=2, uncertainty_threshold=0.2):
    """
    Propagates trainig labels and adds their labels to y, modifies dataset
    """
    dataset.propagated_mask = torch.zeros_like(dataset.propagated_mask, dtype=torch.bool)
    dataset.y = dataset.ground_truth.clone() # for sanity
    propagator = LabelPropagation(num_layers=steps, alpha=1)
    logits = propagator(dataset.ground_truth, dataset.edge_index, mask=dataset.train_mask)
    labels = logits.argmax(dim=-1)
    propagated_logits = logits.nonzero(as_tuple=True)[0]
    # remove uncertain logits
    normalized_uncertainty_scores = torch.from_numpy(entropy(logits[propagated_logits].T) / math.log(dataset.num_classes))
    propagated_logits = propagated_logits[torch.nonzero(normalized_uncertainty_scores <= uncertainty_threshold)]
    dataset.propagated_mask[propagated_logits] = True
    dataset.propagated_mask[dataset.test_mask] = False # prevent test leak
    dataset.y[dataset.propagated_mask] = labels[dataset.propagated_mask]
    #print("Propagated", dataset.propagated_mask.sum(), "labels\twrong samples:", (dataset.y != dataset.ground_truth).sum(), "\t", uncertainty_threshold)

def run(model,
        dataset,
        sampler='model',
        runs=10,
        label_propagation_uncertainty_treshold=0.2,
        budget=100,
        seed=133742069,
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
        classifier = None
        sampler_fun = sampling.sampler[sampler]
        # unintitialized stats
        full_train_stats = {"accuracy": [],
                            "macro-f1": [],
                            "confusion": []}
        full_test_stats = {"accuracy": [],
                           "macro-f1": [],
                           "confusion": []}
        initial_budget = budget
        def combine_training_stats(x): # x[0] is the full training stat, x[1] the new train stat
                return x[0] + [x[1]]
        #train once on random labels for initialisation
        train_mask_backup = dataset.train_mask.clone()
        dataset.train_mask = dataset.val_mask
        dataset.y = torch.randint_like(dataset.y, low=0, high=dataset.num_classes)
        train_stats, test_stats = few_shot_training(optimizer, model, dataset)
        full_train_stats = merge_with(combine_training_stats, full_train_stats, train_stats)
        full_test_stats = merge_with(combine_training_stats, full_test_stats, test_stats)
        dataset.train_mask = train_mask_backup
        budget_history = [0]
        train_class_distribution = [[]]
        while(budget > 0):
            # ask active learner for vertices
            sampled_indices = sampler_fun(min(budget, dataset.num_classes), model, dataset, classifier)
            budget -= len(sampled_indices)
            budget_history.append(initial_budget - budget)
            # move sampled vertices from the validation to the training set, also restore propagated indices if applicable
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            # apply label propagation
            label_propagation(model, dataset, uncertainty_threshold=label_propagation_uncertainty_treshold)
            dataset.y[dataset.train_mask] = dataset.ground_truth[dataset.train_mask]
            # perform training
            train_stats, test_stats = few_shot_training(optimizer, model, dataset)
            #train_stats, test_stats = train(16, optimizer, model, dataset)
            
            full_train_stats = merge_with(combine_training_stats, full_train_stats, train_stats)
            full_test_stats = merge_with(combine_training_stats, full_test_stats, test_stats)
            train_class_distribution.append(torch.bincount(dataset.ground_truth[dataset.train_mask]).numpy())
        print(full_test_stats['accuracy'][-1])
        return {"budget_used": budget_history,
                "train": full_train_stats,
                "test": full_test_stats,
                "train_distribution": train_class_distribution}
    # perform experiments
    results = []
    random.seed(seed)
    seeds = [random.randrange(2**31) for i in range(runs)]
    for i in range(runs):
        torch.manual_seed(seeds[i]) # update seeds
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        dataset.train_mask, dataset.val_mask, dataset.test_mask = datasets.create_split(dataset, seed=seed) # update splits with given seed
        results.append(run_once(model(), copy.deepcopy(dataset), budget, learning_rate))
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
