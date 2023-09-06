import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import datasets
import util
import matplotlib
import numpy as np
from functools import partial
from toolz.itertoolz import iterate
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge
import matplotlib.pyplot as plt
from models import GPN_Encoder, GCN

def accuracy(predictions, true_labels, mask):
    """Calculates accuracy, macro-f1 and the confusion matrix

    :param predictions: Predicted labels
    :param true_labels: Ground truth
    :param mask: Mask for instance selection
    :returns: dictionary of accuracy scores (acc, macro-f1, and confusion matrix)
    """
    return {"accuracy": accuracy_score(predictions[mask], true_labels[mask]),
            "macro-f1": f1_score(true_labels[mask], predictions[mask], average='macro'),
            "confusion": confusion_matrix(true_labels[mask], predictions[mask]),}

def train(n, optimizer, model, dataset):
    """Trains and alters given model for n epochs

    :param n: number of epochs
    :param optimizer: optimizer
    :param model: model to train on
    :param dataset: dataset with train/validation/test split
    :returns: dictionary of train and test statistics
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

def select_vertices(n, model, dataset):
    """selects vertices of a dataset to be included into the test set

    :param n: number of samples to draw
    :param model: future use
    :param dataset: data to sample from
    :returns: selected vertex indices
    """
    # get indices which we can sample from
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices

def run(model, dataset, runs=10, budget=100, learning_rate=0.001):
    """constructs given model and performs training

    :param model: model constructor with 0 args for model construction
    :param dataset: data for training, testing, and validation
    :param budget: number of labels moving from validation to training
    :param learning_rate: learning rate of the optimizer
    :returns: run statistics with model
    """
    def run_once(model, dataset, budget, learning_rate):
        """rund given model "runs" times with the same settings

        :param model: model constructor with 0 args for model construction
        :param dataset: data for training, testing, and validation
        :param runs: number of repeats for each experiment
        :param budget: number of labels moving from validation to training
        :param learning_rate: learning rate of the optimizer
        :returns: list of run statistics with models
        """
        model = model()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        acc = {}
        while(budget > 0):
            # ask active learner for vertices
            sampled_indices = select_vertices(min(budget, 10), model, dataset)
            budget -= len(sampled_indices)
            # move sampled vertices from the validation to the training set
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            # train ~20 episodes
            stats = train(20, optimizer, model, dataset)
        return merge(stats, {"model": model, "dataset": dataset})
    return [run_once(model, dataset, budget, learning_rate) for i in range(runs)]

dataset = datasets.get_dataset('Cora')

# run prototypical
model = partial(GPN_Encoder,
                num_node_features=dataset.num_node_features,
                embedding_dim=16,
                num_classes=dataset.num_classes,
                dropout=0.5)
result_gpn = run(model=model, dataset=dataset, runs=1,budget=100)

#run conventional gcn
model = partial(GCN,
                in_channels=dataset.num_node_features,
                hidden_channels=128,
                num_layers=2,
                out_channels=dataset.num_classes,
                dropout=0.5)
result_gcn = run(model=model, dataset=dataset, runs=5, budget=100)


#embeddings = result_gcn[0]['model'](dataset.x, dataset.edge_index)
model = result_gpn[0]['model']
modified_dataset = result_gpn[0]['dataset']
embeddings = model.embeddings
prototypes = model.prototypes
util.plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.Tensor((modified_dataset.y.max()+1)*[1+modified_dataset.y.max()])]))
