import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn.models

class GPN_Encoder(torch.nn.Module):
    def __init__(self, num_node_features, embedding_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, embedding_dim*8)
        self.conv2 = GCNConv(embedding_dim*8, embedding_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_prototypes(self, embeddings, labels, num_labels):
    # position unknown prototypes to the center to "force" labels away from center
        prototypes = torch.stack([
            (torch.mean(embeddings[labels==label], dim=0) if len(embeddings[labels==label]) > 0 else torch.zeros(embeddings.size(1))) for label in range(num_labels)])
        return prototypes

    def cosine_loss(self, prototypes):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 6
        geometric_center = torch.mean(prototypes, dim=0)
        normalized_prototypes = F.normalize(prototypes - geometric_center)
        cosine_distances = torch.mm(normalized_prototypes,normalized_prototypes.T) - torch.eye(normalized_prototypes.size(0))
        biggest_distances = torch.max(cosine_distances, dim=1).values
        loss = torch.mean(biggest_distances)
        return loss

    def euclidean_loss(self, prototypes):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 5
        prototype_distances = torch.cdist(prototypes, prototypes)
        distance_scores = torch.exp(-prototype_distances) * (1 - torch.eye(prototype_distances.size(0)))
        biggest_distances = torch.max(distance_scores, dim=1).values
        loss = torch.mean(biggest_distances)
        return loss

    def prototype_loss(self, embeddings, prototypes, ground_truth, num_labels):
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 3 and 4
        prototype_distances = torch.cdist(embeddings, prototypes)
        prototype_scores = torch.exp(-prototype_distances)
        prototype_total_scores = torch.sum(prototype_scores, dim=1)
        prototype_probabilities = prototype_scores / prototype_total_scores.unsqueeze(-1)
        loss = F.nll_loss(F.log_softmax(prototype_probabilities, dim=1), ground_truth)
        return loss

    def loss(self, embeddings, ground_truth, num_labels):
        prototypes = self.get_prototypes(embeddings, ground_truth, num_labels)
        prototype_loss = self.prototype_loss(embeddings, prototypes, ground_truth, num_labels)
        euclidean_loss = self.euclidean_loss(prototypes)
        cosine_loss = self.cosine_loss(prototypes)
        return prototype_loss + euclidean_loss + cosine_loss

class GCN(torch_geometric.nn.models.GCN):
    def loss(self, embeddings, ground_truth, num_labels):
        return F.nll_loss(F.log_softmax(embeddings, dim=1), ground_truth)
