import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GPN_Encoder(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 2*num_hidden)
        self.conv2 = GCNConv(2*num_hidden, num_hidden)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, edge_index)
