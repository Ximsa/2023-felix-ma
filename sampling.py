import torch

import json
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import f1_score
from torch.func import vmap
from toolz.functoolz import pipe, thread_first, identity, do
from toolz.itertoolz import groupby, first
from toolz.dicttoolz import valmap, dissoc
from itertools import permutations
from torch_geometric.utils import degree

def random_sampling(n, model, dataset, classifier):
    """Randomly samples vertices of a dataset to be included into the test set
    
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
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
    :returns: Selected vertex indices
    """
    degrees = degree(dataset.edge_index[0], dataset.num_nodes)
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    degrees[exclude_mask] = -1
    degrees, indices = torch.sort(degrees, descending=True)
    return indices[:n]

def model_sampling(n, model, dataset, classifier):
    """
    Selects vertices of a dataset using the model as classifier to be included into the test set.
    From those vertices the ones with the highest entrophy and degree are sampled
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param classifier: fallback if model f1 accuracy is low
    :returns: Selected vertex indices
    """
    model_labels = model(dataset.x, dataset.edge_index).argmax(dim=1).detach()
    clustered_labels = torch.from_numpy(classifier.predict(dataset.x))
    chosen_labels = clustered_labels
    if dataset.test_mask.sum() > 0:
        clustered_train_labels = clustered_labels[dataset.test_mask]
        mapping = find_agreeing_label_mapping(dataset.y[dataset.test_mask],
                                              clustered_train_labels,
                                              x_labels = dataset.y.unique())
        clustered_train_labels = clustered_train_labels.apply_(lambda x: mapping[x])
        clustered_f1 = f1_score(dataset.y[dataset.test_mask], clustered_train_labels, average='macro')
        model_train_labels = model_labels[dataset.test_mask]
        model_f1 = f1_score(dataset.y[dataset.test_mask], model_train_labels, average='macro')
        chosen_labels = clustered_labels if model_f1 < clustered_f1 else model_labels
        #print(clustered_f1, model_f1)
    return classifier_sampling(n,
                               model,
                               dataset,
                               chosen_labels)

def kmedoids_sampling(n, model, dataset, classifier):
    """
    Selects vertices of a dataset using the classifier to be included into the test set.
    From those vertices the ones with the highest entrophy and degree are sampled
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    return classifier_sampling(n,
                               model,
                               dataset,
                               torch.from_numpy(classifier.predict(dataset.x)))


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
    # determine samples to be drawn per class
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    remainder = n % dataset.num_classes
    if remainder > 0:
        samples_per_class[torch.multinomial(
            torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # draw samples based in entropy and degree score
    probabilities = model(dataset.x, dataset.edge_index).detach()
    sampled_indices = torch.tensor([], dtype=int)
    degrees = degree(dataset.edge_index[0], dataset.num_nodes)
    for label, indices in grouped_indices.items():
        num_samples = samples_per_class[label]
        if(num_samples > 0):
            sorted_scores, sorted_indices_indices = pipe(
                probabilities[indices].T,
                entropy,
                torch.from_numpy,
                lambda entropies: entropies * degrees[indices],
                torch.sort)
            sorted_indices = torch.tensor(indices)[sorted_indices_indices]
            sampled_indices = torch.cat([sampled_indices,
                                         sorted_indices[-samples_per_class[label]:]])
    return sampled_indices

def disagreement_sampling(n, model, dataset, classifier):
    """Samples vertices based upon disagreement between the classifier and the model
    
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    #todo:finish
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    # both classifier made predictions, but their label assignment might mismatch
    model_prediction = model(dataset.x, dataset.edge_index).argmax(dim=1)
    classifier_prediction = torch.from_numpy(classifier.predict(dataset.x))
    #fix mismatch
    mapper_mask = torch.logical_or(dataset.val_mask, dataset.train_mask)
    mapping = find_agreeing_label_mapping(model_prediction[mapper_mask],
                                          classifier_prediction[mapper_mask])
    classifier_prediction = classifier_prediction.apply_(lambda x: mapping[x])
    disagreeing_labels = (classifier_prediction == model_prediction)
    
def find_agreeing_label_mapping(xs, ys, x_labels=None, unique_assignment=True):
    """
    Finds a (good, non-perfect) permutation, that maximizes the aggreement between label mappings.
    Assumes labels to range from 0 to n without holes

    :param xs: labels to match
    :param model: labels provided from another classifier
    :returns: mapping that tries to maximize the agreement
    """
    if x_labels == None:
        x_labels = xs.unique(sorted=True)
    found_y_labels = []
    for x_label in x_labels:
        y_labels = ys.unique()
        best_score = 0
        best_y_label = 0
        for y_label in y_labels:
            if(not unique_assignment or y_label not in found_y_labels):
                # score is aggreeing labels divided by size of classes
                score = torch.logical_and(xs == x_label, ys == y_label).sum()# / torch.logical_or(xs == x_label, ys == y_label).sum()
                if(score >= best_score):
                    best_score = score
                    best_y_label = y_label
        found_y_labels.append(best_y_label)
    return torch.tensor(found_y_labels)


sampler = {"random": random_sampling,
           "entropy": entropy_sampling,
           "degree": degree_sampling,
           "model": model_sampling,
           "kmedoids": kmedoids_sampling,}
