import torch
import math
import numpy as np
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

def create_split(data, train_portion=0.0, val_portion=0.8, seed=None):  
    y = data.y.cpu().detach().numpy()
    unique, counts = np.unique(y, return_counts=True)
    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []
    for cl in unique:
        tmp = np.argwhere(y==cl)
        c1 = int(len(tmp)*train_portion)
        c2 = int(len(tmp)*(train_portion+val_portion))
        rng.shuffle(tmp)
        train.append(tmp[:c1])
        val.append(tmp[c1:c2])
        test.append(tmp[c2:])
    train_ix = np.concatenate(train)
    val_ix = np.concatenate(val)
    test_ix = np.concatenate(test)
    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    train_mask[train_ix] = True
    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[val_ix] = True
    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[test_ix] = True
    return train_mask, val_mask, test_mask

def get_dataset(dataset_name):
    load_function = datasets[dataset_name]
    dataset_location = ''.join(["/tmp/", dataset_name])
    dataset = load_function(root=dataset_location, name=dataset_name)[0]
    dataset.y = torch.flatten(dataset.y)
    dataset.train_mask, dataset.val_mask, dataset.test_mask = create_split(dataset)
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
