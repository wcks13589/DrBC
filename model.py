import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

class DrBC(MessagePassing):
    def __init__(self, in_channels, out_channels, n_layer=4):
        super(DrBC, self).__init__(aggr='add')
        self.n_layer = n_layer
        
        self.layer1 = torch.nn.Linear(in_channels, out_channels)
        self.GRU = torch.nn.GRUCell(out_channels, out_channels)
        self.layer2 = torch.nn.Linear(out_channels, out_channels//2)
        self.layer3 = torch.nn.Linear(out_channels//2, 1)
        
    def block(self, x, edge_index):
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)+1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)
        
        
    def encode(self, x, edge_index):
        h_l = []
        h = self.layer1(x).relu()
        
        h = F.normalize(h, p=2, dim=-1)
        
        h_l.append(h.unsqueeze(1))
        
        for i in range(self.n_layer):
            h_n = self.block(h, edge_index)
            
            #h = self.a[str(i)](h_n, h_l[-1])
            h = self.GRU(h_n, h)
            
            h_l.append(h.unsqueeze(1))

        z = torch.cat(h_l, 1)
        z = F.normalize(z, p=2, dim=-1)       
        z = torch.max(z,1)[0]

        return z
    
    def decode(self, z):
        
        z = self.layer2(z).relu()

        return self.layer3(z)
    
    def loss_f(self, y_pred, y_true, sample_edge_pairs):

        i, j = sample_edge_pairs

        y_ij = y_pred[i] - y_pred[j]
        b_ij = y_true[i] - y_true[j]

        y = torch.sigmoid(y_ij)
        b = torch.sigmoid(b_ij)

        loss = (-b)*torch.log(y)-(1-b)*torch.log(1-y)
        
        return loss.sum()
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j