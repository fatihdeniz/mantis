import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels=2, dim=256, heads=8):
        super().__init__()

        # First Graph Attention Layer
        self.conv1 = GATv2Conv(in_channels, dim, heads=heads, dropout=0.6)
        
        # Second Graph Attention Layer
        self.conv3 = GATv2Conv(int(dim * heads), dim, heads=heads, dropout=0.6)
        
        # Final Graph Attention Layer
        self.conv2 = GATv2Conv(int(dim * heads), out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # Apply dropout to input features
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Apply first GAT layer and activation function
        x = F.elu(self.conv1(x, edge_index))
        
        # Apply dropout
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Apply second GAT layer and activation function
        x = F.elu(self.conv3(x, edge_index))
        
        # Apply dropout
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Apply final GAT layer
        x = self.conv2(x, edge_index)
        
        # Apply log softmax activation function
        return F.log_softmax(x, dim=-1)