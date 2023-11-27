import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SimpleConv, SGConv
import torch_geometric.nn.models

class GPN(torch.nn.Module):
    def __init__(self,
                 dataset,
                 hidden_dim_size = 128,
                 dropout = 0.5,
                 distance_loss_weight = 1.0):
        super().__init__()
        #self.eval()
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        self.pagerank_scores = dataset.pagerank
        self.conv1 = GCNConv(num_node_features, hidden_dim_size)
        self.conv2 = GCNConv(hidden_dim_size, hidden_dim_size)
        self.dropout = dropout
        self.distance_loss_weight = distance_loss_weight
        # initialize embeddings by performing one forward call, used to initialize prototypes
        self.prototypes = torch.rand([dataset.num_classes, hidden_dim_size])
        """self.forward(dataset.x, dataset.edge_index)
        self.prototypes = torch.normal(mean=torch.zeros([dataset.num_classes, hidden_dim_size]), std=self.embeddings.detach().abs().mean())
        self.zero_grad()
        self.train()"""

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

    def get_prototypes(self, labels, mask, num_classes):
        labels = labels[mask]
        embeddings = self.embeddings[mask]
        pagerank_scores = self.pagerank_scores[mask]
        # position unknown prototypes to the center to "force" labels away from center
        prototypes = []
        for label in range(num_classes):
            if len(embeddings[labels==label]) > 0:
                normalized_pagerank = F.normalize(pagerank_scores[labels==label], p=1, dim=0).unsqueeze(1)
                prototypes.append(
                    (embeddings[labels==label] * normalized_pagerank).sum(dim=0))
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

    def prototype_loss(self, ground_truth, mask, num_labels): # intra-class loss
        # modeled after https://dl.acm.org/doi/pdf/10.1145/3607144, equation 3 and 4
        prototype_distances = torch.cdist(self.embeddings[mask], self.prototypes)
        prototype_scores = torch.exp(-prototype_distances)
        prototype_total_scores = torch.sum(prototype_scores, dim=1)
        prototype_logits = prototype_scores / prototype_total_scores.unsqueeze(-1)
        loss = F.nll_loss(torch.log(prototype_logits), ground_truth[mask])
        return loss
    
    def loss(self, dataset, logits, support_indices, query_indices=None):
        if query_indices is None:
            query_indices = support_indices
        ground_truth = dataset.y
        num_classes = dataset.num_classes
        self.prototypes = self.get_prototypes(ground_truth, support_indices, num_classes)
        prototype_loss = self.prototype_loss(ground_truth, query_indices, num_classes)
        euclidean_loss = self.euclidean_loss()
        cosine_loss = self.cosine_loss()
        return prototype_loss + self.distance_loss_weight * (euclidean_loss + cosine_loss)

class GPN_GAT(GPN):
    def __init__(self,
                 dataset,
                 hidden_dim_size = 128,
                 dropout = 0.5,
                 distance_loss_weight = 1.0):
        super().__init__(dataset,
                         hidden_dim_size,
                         dropout,
                         distance_loss_weight)
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        self.conv1 = GATConv(num_node_features, hidden_dim_size)
        self.conv2 = GATConv(hidden_dim_size, embedding_dim)

class GCN(torch_geometric.nn.models.GCN):
    def __init__(self,
                 dataset,
                 hidden_dim_size = 128,
                 dropout = 0.5,
                 distance_loss_weight = 1.0): # unused
        super().__init__(dataset.num_node_features,
                         hidden_dim_size,
                         num_layers=2,
                         out_channels=dataset.num_classes,
                         dropout=dropout)

    def loss(self, dataset, logits, support_indices, query_indices=None):
        
        return F.cross_entropy(F.softmax(logits[support_indices], dim=1), dataset.y[support_indices])



models = {"GCN": GCN,
          "GPN-GCN": GPN,
          "GPN-GAT": GPN_GAT,}
