#imported libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import copy
import matplotlib
import numpy as np
from functools import partial
from toolz.itertoolz import iterate, first
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge
import matplotlib.pyplot as plt

#own libraries
import datasets
import sampling
from models import GPN_Encoder, GCN
from util import cond, plot_embeddings

def accuracy(predictions, true_labels, mask):
    """Calculates accuracy, macro-f1 and the confusion matrix

    :param predictions: Predicted labels
    :param true_labels: Ground truth
    :param mask: Mask for instance selection
    :returns: Dictionary of accuracy scores (acc, macro-f1, and confusion matrix)
    """
    return {"accuracy": accuracy_score(predictions[mask], true_labels[mask]),
            "macro-f1": f1_score(true_labels[mask], predictions[mask], average='macro'),
            "confusion": confusion_matrix(true_labels[mask], predictions[mask]),}

def train(n, optimizer, model, dataset):
    """Trains and alters given model for n epochs

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
        probabilities = model(dataset.x, dataset.edge_index)
        loss = model.loss(dataset, probabilities)
        loss.backward()
        optimizer.step()
    probabilities = model(dataset.x, dataset.edge_index)
    predictions = torch.argmax(probabilities, dim=1)
    return {"train": accuracy(predictions, dataset.y, dataset.train_mask),
            "test": accuracy(predictions, dataset.y, dataset.val_mask)}

def select_vertices(n, model, dataset, classifier):
    """Selects vertices of a dataset to be included into the test set

    :param n: Number of samples to draw
    :param model: Future use
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    # get indices which we can sample from
    #return random_sampling(n, model, dataset)
    return model_sampling(n, model, dataset)

def run(model, dataset, sampler='model', runs=10, budget=100, train_epochs=30, learning_rate=0.001):
    """Runs experiments on given model "runs" times with the same settings

    :param model: Model constructor with 0 args for model construction
    :param dataset: Data for training, testing, and validation
    :param sampler: Sampler used for the active learner
    :param runs: Number of experiment repeats
    :param budget: Number of labels moving from validation to training
    :param train_epochs: Number of epochs to train between samling
    :param learning_rate: Learning rate of the optimizer
    :returns: run statistics with models
    """
    def run_once(model, dataset, budget, learning_rate):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        classifier = cond(sampler,
                          "kmeans", lambda: KMeans(
                              n_clusters=dataset.num_classes).fit(
                                  dataset.x[torch.logical_or(dataset.train_mask,dataset.val_mask)]),
                          lambda: None)()
        sampler_fun = sampling.sampler[sampler]
        while(budget > 0):
            # ask active learner for vertices
            sampled_indices = sampler_fun(min(budget, 10), model, dataset, classifier)
            budget -= len(sampled_indices)
            # move sampled vertices from the validation to the training set
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            train_stats = train(train_epochs, optimizer, model, dataset)
        return merge(train_stats,
                     {"model": model, "dataset": dataset})
    return [run_once(model(), copy.deepcopy(dataset), budget, learning_rate) for i in range(runs)]

dataset = datasets.get_dataset('Cora')

# run prototypical
model = partial(GPN_Encoder,
                num_node_features=dataset.num_node_features,
                embedding_dim=16,
                num_classes=dataset.num_classes,
                dropout=0.5)
results = {}
for desc, sampler in sampling.sampler.items():
    results[desc] = run(model=model, dataset=dataset, sampler=desc,runs=10, budget=100)
    print(desc + "\tf1: " + str(sum(map(lambda d: d['test']['macro-f1'], results[desc])) / len(results[desc])))
"""
#run conventional gcn
model = partial(GCN,
                in_channels=dataset.num_node_features,
                hidden_channels=128,
                num_layers=2,
                out_channels=dataset.num_classes,
                dropout=0.5)
result_gcn = run(model=model, dataset=dataset, runs=5, budget=100)


#embeddings = result_gcn[0]['model'](dataset.x, dataset.edge_index)"""
model = result_gpn[0]['model']
modified_dataset = result_gpn[0]['dataset']
embeddings = model.embeddings
prototypes = model.prototypes
probabilities = model(dataset.x, dataset.edge_index).detach()
num_classes = dataset.y.unique().size(0)
plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.full([num_classes], num_classes)]))

#plot_embeddings(dataset.x, dataset.y)
