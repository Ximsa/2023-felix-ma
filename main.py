import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import datasets
import util
import matplotlib
from toolz.itertoolz import iterate
from toolz.functoolz import thread_last
from models import GPN_Encoder

def center(points):
    return points.sum(0)/points.size(0)

def get_distances(embeddings, prototypes):
    return -torch.cdist(embeddings, prototypes)

def get_prototypes(embeddings, labels):
    prototypes = torch.stack([
        center(embeddings[labels==label]) for label in sorted(labels.unique())])
    return prototypes

def calc_loss(embeddings, labels):
    prototypes = get_prototypes(embeddings, labels)
    prototype_distances = get_distances(embeddings, prototypes)
    return F.nll_loss(F.log_softmax(prototype_distances, dim=1), labels)


def train(n, optimizer, model, dataset):
    model.train()
    unlabeled_mask = dataset.unlabeled_mask
    labeled_mask = dataset.labeled_mask
    for i in range(n):
        optimizer.zero_grad()
        embeddings = model(dataset)
        loss = calc_loss(embeddings[labeled_mask], dataset.y[labeled_mask])
        loss.backward()
        optimizer.step()
    predictions = thread_last(
        dataset.y,
        (get_prototypes, embeddings),
        (get_distances, embeddings)).argmax(dim=1)
    accuracy = (predictions == dataset.y).sum() / len(dataset.y)
    return accuracy

def select_labels(n, model, dataset):
    embeddings = model(dataset.x)
    prototypes = get_prototypes(embeddings[dataset.labeled_mask])
    prototype_distances = get_distances(embeddings, prototypes)
    


dataset = datasets.get_dataset('CiteSeer')
# we split into labeled / unlabeled / test
dataset.labeled_mask = dataset.train_mask#torch.Tensor()
dataset.unlabeled_mask = dataset.val_mask#torch.logical_or(dataset.train_mask, dataset.val_mask)
embedding_dims = 16
model = GPN_Encoder(dataset.num_node_features, embedding_dims, 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

print(train(1, optimizer, model, dataset))

embeddings = model(dataset)
prototypes = get_prototypes(embeddings, dataset.y)
util.plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.Tensor((dataset.y.max()+1)*[1+dataset.y.max()])]))
predictions = thread_last(
    dataset.y,
    (get_prototypes, embeddings),
    (get_distances, embeddings)).argmax(dim=1)
#util.plot_confusion_matrix(predictions, dataset.y)
datasets.get_class_sizes(dataset)
