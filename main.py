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

def get_distances(embeddings, prototypes):
    return -torch.cdist(embeddings, prototypes)

def get_prototypes(embeddings, labels, num_labels):
    # position unknown prototypes to the center to "force" labels away from center
    prototypes = torch.stack([
        (torch.mean(embeddings[labels==label], dim=0) if len(embeddings[labels==label]) > 0 else torch.zeros(embeddings.size(1))) for label in range(num_labels)])
    return prototypes

def accuracy(embeddings, true_labels, mask):
    prototypes = get_prototypes(embeddings[mask], true_labels[mask], len(true_labels.unique()))
    distances = get_distances(embeddings[mask], prototypes)
    predictions = distances.argmax(dim=1)
    return {"accuracy": accuracy_score(predictions, true_labels[mask]),
            "macro-f1": f1_score(true_labels[mask], predictions, average='macro'),
            "confusion": confusion_matrix(true_labels[mask], predictions),}

def train(n, optimizer, model, dataset):
    num_labels = len(dataset.y.unique())
    model.train()
    for i in range(n):
        optimizer.zero_grad()
        embeddings = model(dataset.x, dataset.edge_index)
        loss = model.loss(embeddings[dataset.train_mask],
                          dataset.y[dataset.train_mask],
                          num_labels)
        loss.backward()#retain_graph=True)
        optimizer.step()
    prototypes = get_prototypes(embeddings[dataset.train_mask], dataset.y[dataset.train_mask], num_labels)
    distances = get_distances(embeddings[dataset.train_mask], prototypes)
    predictions = distances.argmax(dim=1)
    return {"train": accuracy(embeddings, dataset.y, dataset.train_mask),
            "test": accuracy(embeddings, dataset.y, dataset.val_mask)}

def select_vertices(n, model, dataset):
    # get indices which we can sample from
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices

def run(model, dataset, runs=10, budget=100, learning_rate=0.001):
    def run_once(model, dataset, budget, learning_rate):
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
num_classes = len(dataset.y.unique()) # todo: put into dataset
# run prototypical
model = partial(GPN_Encoder,dataset.num_node_features, embedding_dim=16, dropout=0.5)
result_gpn = run(model=model, dataset=dataset, runs=5,budget=100)

#run conventional gcn
model = partial(GCN, in_channels=dataset.num_node_features, hidden_channels=128, num_layers=2, out_channels=num_classes, dropout=0.5)
result_gcn = run(model=model, dataset=dataset, runs=5,budget=100)


embeddings = result_gpn[0]['model'](dataset.x, dataset.edge_index)
modified_dataset = result_gpn[0]['dataset']
prototypes = get_prototypes(embeddings[modified_dataset.train_mask], modified_dataset.y[modified_dataset.train_mask], len(modified_dataset.y.unique()))
util.plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.Tensor((modified_dataset.y.max()+1)*[1+modified_dataset.y.max()])]))
