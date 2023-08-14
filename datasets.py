import torch
import math
from functools import partial
from enum import Enum
from collections import Counter
from torch_geometric.utils import homophily
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn.functional import gini
from matplotlib import pyplot as plt

datasets = {"Cora": Planetoid,
            "CiteSeer": Planetoid,
            "PubMed": Planetoid,
            "Reddit": lambda name, **kwargs: Reddit(**kwargs),
            "ogbn-arxiv": PygNodePropPredDataset}

def get_dataset(dataset_name):
    load_function = datasets[dataset_name]
    dataset_location = ''.join(["/tmp/", dataset_name])
    dataset = load_function(root=dataset_location, name=dataset_name)[0]
    dataset.y = torch.flatten(dataset.y)
    return dataset

def get_homophily(dataset):
    return homophily(dataset.edge_index, dataset.y, method='edge_insensitive')

def get_class_sizes(dataset):
    return torch.bincount(dataset.y)

#for name in datasets.keys():
#    dataset = get_dataset(name)
#    homo = get_homophily(dataset)
#    sizes = get_class_sizes(dataset)
#    print(name, homo)

