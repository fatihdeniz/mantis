import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import sys
sys.path.append('../')


class Gcn(torch.nn.Module):
    def __init__(self, num_features, dim=16, num_classes=1, num_layers=2, model_type='gcn', lstm=False):
        super(Gcn, self).__init__()

        self.conv1 = SAGEConv(num_features, dim) if model_type == 'sage' else GCNConv(num_features, dim)
        self.gcs = nn.ModuleList()
        self.num_layers = num_layers
        self.lstm = lstm
        for i in range(1, num_layers):
            conv = SAGEConv(dim, dim) if model_type == 'sage' else GCNConv(dim, dim)
            self.gcs.append(conv)
        if lstm:
            self.conv2 = nn.Linear(num_layers * dim + num_features, num_classes)
        else:
            self.conv2 = nn.Linear(dim, num_classes)
            
    
    def forward(self, x, edge_index, data=None, save_embedding=False, edge_weight=None):
        conv1_x = F.relu(self.conv1(x, edge_index, edge_weight))
        x_layers = [x, F.dropout(conv1_x, training=self.training)]

        for i in range(1, self.num_layers):
            x_layer = F.relu(self.gcs[i - 1](x_layers[i], edge_index, edge_weight))
            x_layers.append(F.dropout(x_layer, training=self.training))

        if self.lstm:
            x = torch.cat(x_layers, dim=-1)  # Concatenate along the feature dimension
        else:
            x = x_layers[-1]  # Use only the last layer's embedding

        if save_embedding:
            return x

        x = self.conv2(x)
        return F.log_softmax(x, dim=-1)