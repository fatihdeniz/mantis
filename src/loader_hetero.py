import os
import torch
import numpy
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected

from src.dataset import DataProcess
from src.utils import gen_xavier
from src.loader import Loader

class HeteroLoader(Loader): 
    def __init__(self, nodes_file, edges_file, feature_names, args):
        args.hetero = True
        super(HeteroLoader, self).__init__(nodes_file, edges_file, feature_names['domain'], args)
        self.data = self.__prepare_hetero_data(feature_names, args)
        self.data['domain'].val_index_tensor = self.val_index_tensor
        self.data['domain'].test_index_tensor = self.test_index_tensor

    def __prepare_hetero_data(self,feature_names, args):
        domain_features = torch.tensor(self.nodes_df.loc[self.nodes_df['feat_domain'] == 1, feature_names['domain']].values.tolist(), dtype=torch.float)
        if feature_names.get('ip', False):
            ip_features = torch.tensor(self.nodes_df.loc[self.nodes_df['feat_ip'] == 1, feature_names['ip']].values.tolist(), dtype=torch.float)
        else:
            ip_features = torch.ones(self.nodes_df[self.nodes_df['feat_ip'] == 1].shape[0],1)

        subnet_features = torch.ones(self.nodes_df[self.nodes_df['feat_subnet'] == 1].shape[0],1)
        # asn_features = torch.tensor(self.nodes_df.loc[self.nodes_df['feat_asn'] == 1, feature_names].values.tolist(), dtype=torch.float)
        # city_features = torch.tensor(self.nodes_df.loc[self.nodes_df['feat_city'] == 1, feature_names].values.tolist(), dtype=torch.float)

        edge_index = []
        for i in range (5):
            edges = torch.tensor(self.edges_ind.loc[self.edges_ind['edge_type'] == i, ['source', 'target']].values.tolist(), dtype=torch.long).t().contiguous()
            if edges.size()[0] == 0:
                edges = torch.empty(2, 0, dtype=torch.long)
            edge_index.append(edges)
        
        self.y = torch.tensor(list(self.nodes_df.loc[self.nodes_df['feat_domain'] == 1, 'label']), dtype = torch.long)
        
        if args.use_syn:
            syn_file = args.syn_file
            syn_labels = args.syn_labels
            
            size0, size1, size2 = domain_features.size(0), ip_features.size(0), subnet_features.size(0)
            if not os.path.exists(syn_file):
                print(domain_features.size())
                node_count = size0 + size1 + size2
                gen_xavier(xavier_file = syn_file,
                         node_count=node_count, feature_count=len(syn_labels))
            syn_data = DataProcess.load_nodes_from_path(syn_file)[syn_labels].values.tolist()
            domain_features = torch.cat((domain_features, torch.tensor(syn_data[:size0], dtype=torch.float)), 1)
            ip_features = torch.cat((ip_features, torch.tensor(syn_data[size0:size0+size1], dtype=torch.float)), 1)
            subnet_features = torch.cat((subnet_features, torch.tensor(syn_data[size0+size1:], dtype=torch.float)), 1)
        
        data = HeteroData()
        
        data['domain'].x = domain_features  # domain feature matrix
        data['ipp'].x = ip_features          # edge feature matrix
        data['subnet'].x = subnet_features  # subnet feature matrix
        # data['asn'].x = asn_features  # asn feature matrix
        # data['city'].x = city_features  # city feature matrix

        # domains and ips are connected by an edge
        data['domain', 'to', 'ipp'].edge_index = edge_index[0]   # shape: [2, num_edges]
        data['ipp', 'to', 'subnet'].edge_index = edge_index[1]   # shape: [2, num_edges]
        
        data['domain', 'fqdnapex', 'domain'].edge_index = edge_index[2]   # shape: [2, num_edges]
        data['domain', 'similar_apex', 'domain'].edge_index = edge_index[3]   # shape: [2, num_edges]
        data['domain', 'similar_all', 'domain'].edge_index = edge_index[4]   # shape: [2, num_edges]
        
        # data['subnet', 'to', 'asn'].edge_index = edge_index[3]   # shape: [2, num_edges]
        # data['ipp', 'to', 'city'].edge_index = edge_index[2]   # shape: [2, num_edges]
        
        # adding reverse edges
        if data.is_directed():
            data = ToUndirected()(data)
            print("Data converted to undirected:", not data.is_directed())
        
        data['domain'].y = self.y
        data['domain'].train_mask = self.domain_mask
        data['domain'].test_mask = self.test_mask
        data['domain'].validation_mask = self.train_mask

        return data

    
    