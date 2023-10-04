import torch
import math
import json
import random
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
    probabilities = model(dataset.x, dataset.edge_index)
    scores = pipe(probabilities.T.detach(),
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

def model_sampling(n, model, dataset, classifier):
    """
    Selects vertices of a dataset using the model as classifier to be included into the test set.
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param classifier: fallback for the first few iterations
    :param generator: unused
    :returns: Selected vertex indices
    """
    # first two samples are drawn by a classifier
    logits = model(dataset.x, dataset.edge_index).detach()
    labels = (torch.from_numpy(classifier.predict(dataset.x))
              if dataset.train_mask.sum() <= n
              else logits.argmax(dim=1))
    # there might be an empty (validation) class, especially during early trainning. Thus the nearest logit for each class gets assigned to that class
    logits[dataset.train_mask] = 0 # exclude train
    logits[dataset.test_mask] = 0 # and test
    selected_logits = logits.argmax(dim=0)
    for selected_class in range(len(selected_logits)):
        label_index = selected_logits[selected_class]
        labels[label_index] = selected_class
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
    return classifier_sampling(n,
                               model,
                               dataset,
                               torch.from_numpy(classifier.predict(dataset.x)))


def classifier_sampling(n, model, dataset, labels, perfect_sampling=False, oversampling_compensation=False):
    """
    Selects vertices of a dataset using the labels from a classifier,
    which are used for increasing diversity.
    From those vertices the ones with the highest entropy and degree are sampled
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param labels: Labels to base the decision on
    :param perfect_sampling: Only samples correct labels (oracle)
    :param oversampling_compensation: tries to compensate lesser sampled classes, doesn't work with pseudolabels
    :returns: Selected vertex indices
    """
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    labels[exclude_mask] = dataset.num_classes # create an "excluded" class
    labels = labels.numpy() # convert to np for groupby
    # group indices by label and remove excluded indices
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y)))
    grouped_indices = dissoc(grouped_indices, dataset.num_classes)
    print(valmap(len, grouped_indices))
    # determine samples to be drawn per class
    # determine classes that have been oversampled
    logits = model(dataset.x, dataset.edge_index).detach()
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    remainder = n % dataset.num_classes
    if remainder > 0:
        samples_per_class[torch.multinomial(
            torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # draw samples based on entropy and pagerank score
    sampled_indices = torch.tensor([], dtype=int)
    # hyperparameter weighting pagerank vs entropy
    # increase entropy weight
    entropy_pagerank_balance = max(0.3, 0.7-0.003*dataset.train_mask.sum())
    for label, indices in grouped_indices.items():
        num_samples = samples_per_class[label]
        if(num_samples > 0):
            sorted_scores, sorted_indices_indices = pipe(
                logits[indices].T, # secondary sampling by entropy and pagerank
                entropy,
                torch.from_numpy,
                lambda entropies: ((1-entropy_pagerank_balance) * normalize(entropies, dim=0, p=1)
                                   + entropy_pagerank_balance * normalize(dataset.pagerank[indices], dim=0, p=1)), # combine entropy with pagerank
                torch.sort)
            sorted_indices = torch.tensor(indices)[sorted_indices_indices]
            sampled_indices = torch.cat([sampled_indices,
                                         sorted_indices[-samples_per_class[label]:]])
    return sampled_indices


sampler = {"random": random_sampling,
           "entropy": entropy_sampling,
           "degree": degree_sampling,
           "pagerank": pagerank_sampling,
           "model": model_sampling,
           "kmedoids": kmedoids_sampling,} # TODO: experiment with KNeighbours
