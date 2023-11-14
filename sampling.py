import torch
import math
import json
import random
import numpy as np
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

# TODO sampling strategy involving most uncertain neighbours of labeled data?

def random_sampling(n, model, dataset, classifier):
    """Randomly samples vertices of a dataset to be included into the test set
    
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param classifier: unused
    :param generator: torch random number generator
    :returns: Selected vertex indices
    """
    # get indices which we can sample from
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices

def entropy_sampling(n, model, dataset, classifier):
    """Selects vertices of a dataset using entopy to be included into the test set
    
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param classifier: unused
    :param generator: unused
    :returns: Selected vertex indices
    """
    logits = model(dataset.x, dataset.edge_index)
    scores = pipe(logits.T.detach(),
                  entropy,
                  torch.from_numpy)
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    scores[exclude_mask] = -1
    scores, indices = torch.sort(scores, descending=True)
    return indices[:n]

def degree_sampling(n, model, dataset, classifier):
    """Selects vertices of a dataset using their degree to be included into the test set
    
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param classifier: unused
    :param generator: unused
    :returns: Selected vertex indices
    """
    degrees = degree(dataset.edge_index[0], dataset.num_nodes)
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    degrees[exclude_mask] = -1
    degrees, indices = torch.sort(degrees, descending=True)
    return indices[:n]

def pagerank_sampling(n, model, dataset, classifier):
    """Selects vertices of a dataset using their pagerank to be included into the test set
    
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param classifier: unused
    :param generator: unused
    :returns: Selected vertex indices
    """
    scores = dataset.pagerank.clone()
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    scores[exclude_mask] = -1
    scores, indices = torch.sort(scores, descending=True)
    return indices[:n]

def model_sampling(n, model, dataset, classifier, perfect=False):
    """
    Selects vertices of a dataset using the model as classifier to be included into the test set.
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param classifier: unused
    :param perfect: use oracle to get class assignments
    :returns: Selected vertex indices
    """
    logits = model(dataset.x, dataset.edge_index).detach() # choose model sampling
    labels = logits.argmax(dim=1)
    # there might be an empty (validation) class, especially during early trainning. Thus override k labels per class.
    logits[dataset.train_mask] = -1 # exclude train
    logits[dataset.test_mask] = -1 # and test from sampling
    # set k to a sane number
    num_vertices = dataset.val_mask.sum()
    num_classes = dataset.num_classes
    k = math.floor(min(num_vertices / num_classes,
                       (n / num_classes)*4))
    split_logits = logits.split(split_size=1, dim=1)
    taken_indices = []
    for class_index in range(len(split_logits)):
        scores, indices = split_logits[class_index].sort(dim=0, descending=True)
        #now sample k indices
        num_sampled = 0
        i = 0
        while(num_sampled < k):
            sampled_index = indices[i]
            if(sampled_index not in taken_indices): # index isnt sampled yet
                num_sampled += 1
                taken_indices.append(sampled_index)
                labels[sampled_index] = class_index
            i+=1
    return classifier_sampling(n,
                               model,
                               dataset,
                               labels)

def kmedoids_sampling(n, model, dataset, classifier):
    """
    Selects vertices of a dataset using the classifier to be included into the test set.
    From those vertices the ones with the highest entrophy and degree are sampled
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param classifier: Classifier that creates pseudo-labels
    :param generator: unused
    :returns: Selected vertex indices
    """ 
    return model_sampling(n,
                          model,
                          dataset,
                          classifier,
                          -1)


def classifier_sampling(n, model, dataset, labels):
    """
    Selects vertices of a dataset using the labels from a classifier,
    which are used for increasing diversity.
    From those vertices the ones with the highest entropy and degree are sampled
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param labels: Labels to base the decision on
    :returns: Selected vertex indices
    """
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
    entopy_pagerank_weighting = 0.75
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


sampler = {"random": random_sampling,
           "entropy": entropy_sampling,
           "degree": degree_sampling,
           "pagerank": pagerank_sampling,
           "model": model_sampling,
           "kmedoids": kmedoids_sampling,}
