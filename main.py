import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import datasets
import util
from models import GPN_Encoder


def center(points):
    return points.sum(0)/points.size(0)

def get_distances(embeddings, prototypes):
    return -torch.cdist(embeddings, prototypes)

def get_prototypes(embeddings, labels):
    prototypes = torch.stack(
        [center(embeddings[labels==label]) for label in sorted(labels.unique())])
    return prototypes

def calc_loss(embeddings, labels):
    prototypes = get_prototypes(embeddings, labels)
    prototype_distances = get_distances(embeddings, prototypes)
    return F.nll_loss(F.log_softmax(prototype_distances, dim=1), labels)

dataset = datasets.get_dataset('CiteSeer')

embedding_dims = 16
model = GPN_Encoder(dataset.num_node_features, embedding_dims, 0.5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for i in range(100):
    optimizer.zero_grad()
    embeddings = model(dataset)
    predictions = (get_distances(embeddings, get_prototypes(embeddings, dataset.y))).argmax(dim=1)
    correct = (predictions == dataset.y).sum()
    if i % 10 == 0: print(correct / len(predictions))
    my_loss = calc_loss(embeddings, dataset.y)
    my_loss.backward()
    optimizer.step()


embeddings = model(dataset)
prototypes = get_prototypes(embeddings, dataset.y)
util.plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                     labels=torch.cat([dataset.y, torch.Tensor((dataset.y.max()+1)*[1+dataset.y.max()])]))
