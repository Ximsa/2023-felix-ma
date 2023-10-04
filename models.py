import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn.models

class GPN_Encoder(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 num_classes,
                 pagerank_scores,
                 hidden_dim_multiplier = 4,
                 embedding_dim = 16,
                 dropout = 0.5,
                 distance_loss_weight = 1.0):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, embedding_dim*hidden_dim_multiplier)
        self.conv2 = GCNConv(embedding_dim*hidden_dim_multiplier, embedding_dim)
        self.dropout = dropout
        self.distance_loss_weight = distance_loss_weight
        self.pagerank_scores = pagerank_scores
        self.prototypes = torch.rand([num_classes, embedding_dim])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self.embeddings = x
        # transform to probabilities
        distances = torch.cdist(x, self.prototypes)
        scores = torch.exp(-distances)
        total_scores = torch.sum(scores, dim=1)
        logits = scores / total_scores.unsqueeze(-1)
        return logits

    def get_prototypes(self, labels, train_mask, num_classes):
        # TODO: use pagerank (networkX)weights, see https://dl.acm.org/doi/pdf/10.1145/3607144
        embeddings = self.embeddings[train_mask]
        pagerank_scores = self.pagerank_scores[train_mask]
        # position unknown prototypes to the center to "force" labels away from center
        prototypes = []
        for label in range(num_classes):
            if len(embeddings[labels==label]) > 0:
                prototypes.append(
                    (embeddings[labels==label] * pagerank_scores[labels==label].unsqueeze(1)).sum(dim=0) / pagerank_scores[labels==label].sum())
            else:
                prototypes.append(torch.zeros(embeddings.size(1)))
                           
        prototypes = torch.stack(prototypes)
        return prototypes

    def cosine_loss(self):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 6
        geometric_center = torch.mean(self.prototypes, dim=0)
        normalized_prototypes = F.normalize(self.prototypes - geometric_center)
        cosine_distances = torch.mm(normalized_prototypes,normalized_prototypes.T) - torch.eye(normalized_prototypes.size(0))
        biggest_distances = torch.max(cosine_distances, dim=1).values
        loss = torch.mean(biggest_distances)
        return loss

    def euclidean_loss(self):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 5
        prototype_distances = torch.cdist(self.prototypes, self.prototypes)
        distance_scores = torch.exp(-prototype_distances) * (1 - torch.eye(prototype_distances.size(0)))
        biggest_distances = torch.max(distance_scores, dim=1).values
        loss = torch.mean(biggest_distances)
        return loss

    def prototype_loss(self, ground_truth, train_mask, num_labels):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 3 and 4
        prototype_distances = torch.cdist(self.embeddings[train_mask], self.prototypes)
        prototype_scores = torch.exp(-prototype_distances)
        prototype_total_scores = torch.sum(prototype_scores, dim=1)
        prototype_probabilities = prototype_scores / prototype_total_scores.unsqueeze(-1)
        loss = F.nll_loss(F.log_softmax(prototype_probabilities, dim=1), ground_truth)
        return loss

    def loss(self, dataset, probabilities):
        ground_truth = dataset.y[dataset.train_mask]
        train_mask = dataset.train_mask
        num_classes = dataset.num_classes
        self.prototypes = self.get_prototypes(ground_truth, train_mask, num_classes) # prototypes are a learnable parameter now
        prototype_loss = self.prototype_loss(ground_truth, train_mask, num_classes)
        euclidean_loss = self.euclidean_loss()
        cosine_loss = self.cosine_loss()
        return prototype_loss + self.distance_loss_weight * (euclidean_loss + cosine_loss)

class GCN(torch_geometric.nn.models.GCN):
    def loss(self, dataset, probabilities):
        return F.cross_entropy(F.softmax(probabilities[dataset.train_mask], dim=1), dataset.y[dataset.train_mask])
