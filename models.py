import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GPN_Encoder(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden * 4)
        self.conv2 = GCNConv(num_hidden*4, num_hidden)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x
