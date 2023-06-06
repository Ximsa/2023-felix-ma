from matplotlib import pyplot as plt



fig = plt.figure(figsize=(10,10))
fig.set_facecolor('w')
plt.scatter([1,2,3], [4,5,3])

import torch

x = torch.rand(5, 3)
print(x)

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
