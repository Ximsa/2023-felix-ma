import torch
import math
import json
import random
import numpy as np
from enum import Enum
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import f1_score
from torch.func import vmap
from torch.nn.functional import normalize
from toolz.functoolz import pipe, thread_first, identity, do
from toolz.itertoolz import groupby, first
from toolz.dicttoolz import valmap, dissoc
from functools import partial
from itertools import permutations
from torch_geometric.utils import degree
import torch.nn.functional as F

def random_sampling(n, model, dataset, perfect=None, entropy_pagerank_weighting = None):
    """
    Selects vertices randomly.
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param perfect: unused
    :param entopy_pagerank_weighting: unused
    :returns: Selected vertex indices
    """
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices
    
    
def sub_sampler(num_samples, indices, logits, ranks, entropy_pagerank_weighting):
    """
    Selects vertices based on entropy or pagerank from selected indices.
    :param num_samples: Number of samples to draw
    :param indices: indices to sample from
    :param logits: for entropy calculation
    :param ranks: for pagerank weighting
    :param entopy_pagerank_weighting: subsampling strategy, -1 is random, 0 pagerank, 1 entropy 
    :returns: Selected vertex indices
    """
    weights = torch.ones(len(indices))
    if(entropy_pagerank_weighting >= 0):
        normalized_entropies = pipe(logits[indices].T,
                                    entropy,
                                    torch.from_numpy,
                                    partial(normalize, dim=0, p=1))
        normalized_pageranks = pipe(ranks[indices],
                                    partial(normalize, dim=0, p=1))
        weights = (normalized_pageranks * (1-entropy_pagerank_weighting)
                   + normalized_entropies * entropy_pagerank_weighting)
    
    normalized_weights = normalize(weights, dim=0, p=1).numpy()
    selected_indices = np.random.choice(indices,
                                        size=num_samples,
                                        p=normalized_weights,
                                        replace=False)
    return selected_indices

def own_sampling(n, model, dataset, perfect=False, entropy_pagerank_weighting = 0.5, compensate_undersampled=False):
    """
    Selects vertices of a dataset using the model as classifier to be included into the test set.
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param perfect: use oracle to get class assignments
    :param entopy_pagerank_weighting: subsampling strategy, -1 is random, 0 pagerank, 1 entropy 
    :param compensate_undersampled: tries to sample more from undersampled classes
    :returns: Selected vertex indices
    """
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    logits = model(dataset.x, dataset.edge_index).detach() # use model as classifier
    labels = logits.argmax(dim=1)
    if perfect: labels = dataset.ground_truth.clone()
    # there might be an empty (validation) class, especially during early trainning. Thus override k labels per class.
    logits[exclude_mask] = -1 # exclude labeled and unlabeled from sampling
    labels[exclude_mask] = -1
    # set k to a sane number, i.e.: 4
    num_vertices = dataset.val_mask.sum()
    num_classes = dataset.num_classes
    k = 0 if perfect else 4 # prevent knn when having perfect labels
    split_logits = logits.split(split_size=1, dim=1)
    taken_indices = []
    for class_index in range(len(split_logits)): # sample k for each class
        scores, indices = split_logits[class_index].sort(dim=0, descending=True)
        num_sampled = 0
        i = 0
        while(num_sampled < k):
            sampled_index = indices[i]
            if(scores[i] == -1):
                print("sampled unlabeled or test index")
                exit(1)
            if(sampled_index not in taken_indices): # index isnt sampled yet
                num_sampled += 1
                taken_indices.append(sampled_index)
                labels[sampled_index] = class_index
            i+=1
    #print(torch.bincount(dataset.ground_truth[dataset.train_mask]))
    # determine samples per bucket
    samples_per_class = torch.zeros(dataset.num_classes)
    if compensate_undersampled:
        # compensate by moving samples from oversampled to undersampled buckets
        distribution = torch.bincount(dataset.ground_truth[dataset.train_mask])
        samples_per_class[:len(distribution)] = distribution
        samples_per_class = samples_per_class - dataset.train_mask.sum() // dataset.num_classes
        samples_per_class -= 1
        samples_per_class[samples_per_class > 0] = 0
        samples_per_class = samples_per_class.abs()
    else:
        # one sample per class
        samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
        remainder = n % dataset.num_classes
        if remainder > 0:
            samples_per_class[torch.multinomial(
                torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # perform subsampling on each bucket
    sampled_indices = torch.tensor([], dtype=int)
    labels = labels.numpy()
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y))) # group by label
    grouped_indices = dissoc(grouped_indices, -1) # remove train and test
    for label, indices in grouped_indices.items():
        selected_indices = sub_sampler(num_samples=int(min(samples_per_class[label], len(indices))),
                                       indices=indices,
                                       logits=logits,
                                       ranks=dataset.pagerank,
                                       entropy_pagerank_weighting=entropy_pagerank_weighting)
        sampled_indices = torch.cat([sampled_indices,
                                     torch.from_numpy(selected_indices)])
    return sampled_indices

def k_medoids_sampling(n, model, dataset, perfect=False, entopy_pagerank_weighting = 0.5):
    pass

sampler = {
    'random': random_sampling,
    'own': own_sampling,
    'k-medoids': k_medoids_sampling,}

"""
def classifier_sampling(n, model, dataset, labels):
    Selects vertices of a dataset using the labels from a classifier,
    which are used for increasing diversity.
    From those vertices the ones with the highest entropy and degree are sampled
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param labels: Labels to base the decision on
    :returns: Selected vertex indices
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    labels[exclude_mask] = dataset.num_classes # create an "excluded" class
    labels = labels.numpy() # convert to np for groupby
    # group indices by label and remove excluded indices
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y)))
    grouped_indices = dissoc(grouped_indices, dataset.num_classes)
    #print(valmap(len, grouped_indices))
    # determine samples to be drawn per class
    logits = model(dataset.x, dataset.edge_index).detach()
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    remainder = n % dataset.num_classes
    if remainder > 0:
        samples_per_class[torch.multinomial(
            torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # draw samples based on entropy and pagerank score
    sampled_indices = torch.tensor([], dtype=int)
    entopy_pagerank_weighting = 0
    for label, indices in grouped_indices.items():
        num_samples = int(min(samples_per_class[label], len(indices)))
        if(num_samples > 0):
            normalized_entropies = pipe(logits[indices].T,
                                        entropy,
                                        torch.from_numpy,
                                        partial(normalize, dim=0, p=1), 
                                        lambda e: torch.exp(-4 * torch.square(e - 1)),
                                        partial(normalize, dim=0, p=1))
            normalized_pageranks = pipe(dataset.pagerank[indices],
                                        partial(normalize, dim=0, p=1))
            weights = (normalized_pageranks * (1-entopy_pagerank_weighting)
                       + normalized_entropies * entopy_pagerank_weighting).numpy()
            #weights = np.ones_like(weights, dtype='float32')
            normalized_weights = weights / np.sum(weights)
            normalized_weights[-1] = 1 - np.sum(normalized_weights[0:-1])
            selected_indices = np.random.choice(indices,
                                                size=num_samples,
                                                p=normalized_weights,
                                                replace=False)
            sampled_indices = torch.cat([sampled_indices,
                                         torch.from_numpy(selected_indices)])
    return sampled_indices


sub_samplers = ['random', 'entropy', 'pagerank', 'own']

"""
