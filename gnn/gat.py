import torch
import torch.nn as nn 
import torch.nn.functional as F 
# from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GATConv as GATv2Conv



class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels, 48, heads=8, dropout=0.6)
        self.conv3 = GATv2Conv(int(48 * 8), 48, heads=8, dropout=0.6)
        
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATv2Conv(int(48 * 8), 2, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)