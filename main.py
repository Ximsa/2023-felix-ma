import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
import datasets
import util
import matplotlib
import numpy as np
from toolz.itertoolz import iterate
from toolz.functoolz import thread_last
import matplotlib.pyplot as plt
from models import GPN_Encoder


def center(points):
    return points.sum(0)/points.size(0)

def get_distances(embeddings, prototypes):
    return -torch.cdist(embeddings, prototypes)

def get_prototypes(embeddings, labels, num_labels):
    # position unknown prototypes to the center to "force" labels away from center
    prototypes = torch.stack([
        (center(embeddings[labels==label]) if len(embeddings[labels==label]) > 0 else torch.zeros(embeddings.size(1))) for label in range(num_labels)])
    return prototypes

def calc_loss(embeddings, labels, num_labels):
    prototypes = get_prototypes(embeddings, labels, num_labels)
    prototype_distances = get_distances(embeddings, prototypes)
    return F.nll_loss(F.log_softmax(prototype_distances, dim=1), labels)

def my_loss(prototypes):
    x = get_distances(prototypes, prototypes)
    x = torch.exp(x)
    x = torch.max(x,dim=1).values
    x = torch.mean(x)
    return x


def accuracy(embeddings, true_labels, mask):
    prototypes = get_prototypes(embeddings[mask], true_labels[mask], len(true_labels.unique()))
    distances = get_distances(embeddings[mask], prototypes)
    predictions = distances.argmax(dim=1)
    accuracy = (predictions == true_labels[mask]).sum() / len(true_labels[mask])
    return {"accuracy": (predictions == true_labels[mask]).sum() / len(true_labels[mask]),
            "macro-f1": f1_score(true_labels[mask], predictions, average='macro'),
            "confusion": confusion_matrix(true_labels[mask], predictions),}

def train(n, optimizer, model, dataset):
    num_labels = len(dataset.y.unique())
    model.train()
    for i in range(n):
        optimizer.zero_grad()
        embeddings = model(dataset)
        loss = calc_loss(embeddings[dataset.train_mask], dataset.y[dataset.train_mask], num_labels)
        loss.backward(retain_graph=True)
        optimizer.step()
    prototypes = get_prototypes(embeddings[dataset.train_mask], dataset.y[dataset.train_mask], num_labels)
    distances = get_distances(embeddings[dataset.train_mask], prototypes)
    predictions = distances.argmax(dim=1)
    return [accuracy(embeddings, dataset.y, dataset.train_mask),
            accuracy(embeddings, dataset.y, dataset.val_mask)]

def select_vertices(n, model, dataset):
    # get indices which we can sample from
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices

def run(runs=1, budget=100, dataset='Cora', learning_rate=0.01):
    dataset = datasets.get_dataset(dataset)
    embedding_dims = 16
    model = GPN_Encoder(dataset.num_node_features, embedding_dims, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    while(budget > 0):
        # ask active learner for vertices
        sampled_indices = select_vertices(10, model, dataset)
        budget -= len(sampled_indices)
        # move sampled vertices from the validation to the training set
        dataset.val_mask[sampled_indices] = False
        dataset.train_mask[sampled_indices] = True
        # train ~20 episodes
        acc = train(40, optimizer, model, dataset)
        print(acc)
run()

embeddings = model(dataset)
prototypes = get_prototypes(embeddings, dataset.y)
util.plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.Tensor((dataset.y.max()+1)*[1+dataset.y.max()])]))
predictions = thread_last(dataset.y,
                          (get_prototypes, embeddings),
                          (get_distances, embeddings)).argmax(dim=1)
#util.plot_confusion_matrix(predictions, dataset.y)
datasets.get_class_sizes(dataset)

#util.plot_embeddings(dataset.x, labels=dataset.y)
