import torch

from scipy.stats import entropy
from sklearn.cluster import KMeans
from torch.func import vmap
from toolz.functoolz import pipe, thread_first, identity
from toolz.itertoolz import groupby, first
from toolz.dicttoolz import valmap, dissoc
from itertools import permutations
from torch_geometric.utils import to_dense_adj

def random_sampling(n, model, dataset):
    """Randomly samples vertices of a dataset to be included into the test set
    
    :param n: Number of samples to draw
    :param model: Future use
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    # get indices which we can sample from
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices

def entropy_sampling(n, model, dataset):
    """Selects vertices of a dataset using entopy to be included into the test set
    
    :param n: Number of samples to draw
    :param model: Future use
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    probabilities = model(dataset.x, dataset.edge_index)
    scores = pipe(probabilities.T.detach(),
                  entropy,
                  torch.from_numpy)
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    scores[exclude_mask] = 0
    scores, indices = torch.sort(scores, descending=True)
    return indices[:n] if not inverse else indices[-(exclude_mask.sum()+n):-(exclude_mask.sum())]

def model_sampling(n, model, dataset):
    """Selects vertices of a dataset using the model as classifier to be included into the test set
    
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    return classifier_sampling(n,
                               model,
                               dataset,
                               model(dataset.x, dataset.edge_index).argmax(dim=1).detach().numpy())

def kmeans_sampling(n, model, dataset, classifier):
    return classifier_sampling(n,
                               model,
                               dataset,
                               classifier.predict(dataset.x))

def disagreement_sampling(n, model, dataset, classifier):
    # todo: maybe train gcn and gpn and use their disagreement for instance selection
    # also use prediction probabilities and not just hard predictions for selecytion
    pass


def classifier_sampling(n, model, dataset, labels):
    """
    Selects vertices of a dataset using the labels from a classifier,
    which are used for increasing diversity
    
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :returns: Selected vertex indices
    """
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    labels[exclude_mask] = dataset.num_classes # create an "excluded" class
    # group indices by label and remove excluded indices
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y)))
    grouped_indices = dissoc(grouped_indices, dataset.num_classes)
    # determine samples to be drawn per class
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    samples_per_class[torch.multinomial(
        torch.ones(dataset.num_classes, dtype=float),
        n % dataset.num_classes)] += 1
    # draw samples based in entropy score
    probabilities = model(dataset.x, dataset.edge_index)
    sampled_indices = torch.tensor([], dtype=int)
    for label, indices in grouped_indices.items():
        num_samples = samples_per_class[label]
        if(num_samples > 0):
            sorted_scores, sorted_indices_indices = pipe(probabilities.detach()[indices].T,
                                                         entropy,
                                                         torch.from_numpy,
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
    
def find_agreeing_label_mapping(xs, ys, unique_assignment=True):
    """
    Finds a (good, non-perfect) permutation, that maximizes the aggreement between label mappings.
    Assumes labels to range from 0 to n without holes

    :param xs: labels to match
    :param model: labels provided from another classifier
    :returns: mapping that tries to maximize the agreement
    """
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
