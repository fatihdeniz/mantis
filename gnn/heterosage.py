import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, HANConv, Linear

class HAN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels: int, out_channels: int, num_layers: int, heads=8):
        super().__init__()
        
        assert num_layers > 0
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            han_conv = HANConv(7, hidden_channels, heads=heads, dropout=0.6, metadata=metadata)
            self.convs.append(han_conv)
        
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, training=self.training) for key, x in x_dict.items()}
        
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        out = self.lin(x_dict['domain'])
        return out

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, training=self.training) for key, x in x_dict.items()}
        
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        return self.lin(x_dict['domain'])