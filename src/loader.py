from os import device_encoding
import os
import torch
import numpy
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected

from src.dataset import DataProcess
from src.utils import gen_xavier, seed_everything
from src.domain_utils import get_2ld

class Loader(): 
    def __init__(self, nodes_file, edges_file, feature_names, args):
        labelfeature_names=args.labelfeature_names
        train_percentage=args.train_percentage
        seed=args.seed
        
        seed_everything(seed)
        
        experiment_id=args.identifier
        gpu_id=args.gpu_id
        
        self.onehot_labelfeature = (len(labelfeature_names) > 2)
        self.edges_ind, self.nodes_df = DataProcess.load_hetero_data_from_path(edges_file, nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        
        if 'label' not in self.nodes_df.columns:
            self.nodes_df['label'] = 2
        
        self.edges_ind = self.edges_ind[self.edges_ind['edge_type'].isin(args.edge_type)]
        
        if args.hetero:
            num_nodes = self.nodes_df[self.nodes_df['feat_domain'] == 1].shape[0]
        else:
            keep_columns = {'source', 'target', 'edge_type'}
            remove_columns = set(self.edges_ind.columns) - keep_columns
            self.edges_ind.drop(remove_columns, axis=1, inplace=True)
            num_nodes = self.nodes_df.shape[0]
        
        self.num_nodes = num_nodes
        self.__set_masks(num_nodes, args)
        self.__init_labelfeature(labelfeature_names)
        self.data = self.__prepare_data(feature_names, args) if not args.hetero else None
        
        self.experiment_id = experiment_id
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu_id != -1 else 'cpu')
        if self.device != torch.device('cpu') and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(1.0, device=torch._C._cuda_getDevice())

    def __prepare_data(self,feature_names,args):
        features = torch.tensor(self.nodes_df[feature_names].values.tolist(), dtype=torch.float)
        self.edge_index = torch.tensor(self.edges_ind.values.tolist(), dtype=torch.long).t().contiguous()
        self.y = torch.tensor(list(self.nodes_df['label']), dtype = torch.long)
        data = Data(x=features, edge_index=self.edge_index, y=self.y)
        data.num_nodes = self.num_nodes
        data.n_id = torch.arange(self.num_nodes)
        data.train_mask = self.domain_mask
        data.test_mask = self.test_mask
        data.validation_mask = self.train_mask

        data.domain_mask = self.domain_mask
        if hasattr(self, 'popular_ip_mask'):
            data.popular_ip_mask = self.popular_ip_mask
            
        data.edge_weight = torch.ones(self.edge_index.size(), dtype=torch.float)[1]
        if 'edge_type' in self.edges_ind.columns: 
            data.edge_type = torch.tensor(self.edges_ind['edge_type'].values.tolist(), dtype=torch.float)
        
        if args.use_syn:
            syn_file = args.syn_file
            syn_labels = args.syn_labels
            
            if not os.path.exists(syn_file):
                gen_xavier(xavier_file = syn_file,
                         node_count=features.size(dim=0), feature_count=len(syn_labels))

            data.x = torch.cat((data.x, torch.tensor(DataProcess.load_nodes_from_path(syn_file)[syn_labels].values.tolist(), dtype=torch.float)), 1)

        if data.is_directed():
            data = ToUndirected()(data)
            print("Data converted to undirected:", not data.is_directed())
        
        print(data)
        return data

    def __set_masks(self, num_nodes, args) :
        self.domain_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.unknown_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.domain_mask[self.nodes_df[self.nodes_df['feat_domain'] == 1].index.values] = True
        self.unknown_mask[self.nodes_df[(self.nodes_df['feat_domain'] == 1) & (self.nodes_df['label']==2)].index.values] = True
        
        self.unknownapex_mask = torch.zeros(num_nodes, dtype=torch.bool)
        apex = self.nodes_df[(self.nodes_df['feat_domain'] == 1) & (self.nodes_df['label']==2)]['node'].apply(get_2ld)
        self.nodes_df.loc[(self.nodes_df['feat_domain'] == 1) & (self.nodes_df['label']==2), 'apex'] = apex.tolist()
        self.unknownapex_mask[self.nodes_df[self.nodes_df['node']==self.nodes_df['apex']].index.values] = True
        
        # print(max(numpy.unique(self.nodes_df['label'].values.tolist())))
        max_label = max(numpy.unique(self.nodes_df['label'].values.tolist()))
        max_label = 2 # this is because we only care about  labels 0 and 1, we should remove this line normally
        
        if args.label_source is not None:
            mal_nodes=self.nodes_df[(self.nodes_df['label'] == 1) & (self.nodes_df['label_source'].isin(args.label_source))].index.values 
            ben_nodes=self.nodes_df[(self.nodes_df['label'] == 0) & (self.nodes_df['label_source'].isin(args.label_source))].index.values 
            
        else:
            mal_nodes=self.nodes_df[(self.nodes_df['label'] == 1) ].index.values 
            ben_nodes=self.nodes_df[(self.nodes_df['label'] == 0) ].index.values 
            
        if args.label_source is not None and args.popularity_lists:
            # print('Popularity LISTS')
            # used for mimicip attack
            self.popular_ip_mask = torch.zeros(num_nodes, dtype=torch.bool)
            popular_domains = set(self.nodes_df[(self.nodes_df['label'] == 0) & (self.nodes_df['label_source'].isin(args.popularity_lists))]['node'])
            popular_ip_index = self.edges_df[(self.edges_df.edge_type == 0) & self.edges_df['domain'].isin(popular_domains)]['target'].values
            # print('Popular IP index', popular_ip_index)
            self.popular_ip_mask[popular_ip_index] = True

        numpy.random.seed(args.seed)
        
        if args.balance_label_source is not None:
            self.domain_mask = torch.zeros(num_nodes, dtype=torch.bool)

            balance_indexes=self.nodes_df[(self.nodes_df['label_source'] == args.balance_label_source)].index.values 
            
        if args.balance_labels:
            min_count = min(len(mal_nodes), len(ben_nodes))
            remove_count = max(len(mal_nodes), len(ben_nodes)) - min_count
            numpy.random.shuffle(balance_indexes)
            balance_indexes = balance_indexes[:remove_count]
            
            mal_nodes = list(set(mal_nodes) - set(balance_indexes))
            ben_nodes = list(set(ben_nodes) - set(balance_indexes))
        
        label_indices = [ben_nodes, mal_nodes]

        for i in range(max_label):
            
            class_index = label_indices[i]

            print(f'Label == {i}:', len(class_index))
            if len(class_index) == 0: continue
            numpy.random.shuffle(class_index) 

            train_mal_count = int(len(class_index) * args.train_percentage)

            # Compute train/test mask according to experiment_id
            test_mal_count = len(class_index) - train_mal_count
            train_start_index = (args.experiment_id *  test_mal_count) % len(class_index)
            if train_start_index + train_mal_count <= len(class_index):
                self.train_mask[class_index[train_start_index:train_start_index+train_mal_count]] = True
            else:
                self.train_mask[class_index[train_start_index:]] = True
                self.train_mask[class_index[:(train_start_index+train_mal_count)%len(class_index)]] = True

            test_mal_indices = numpy.setdiff1d(class_index, self.train_mask.nonzero().t().contiguous().tolist()[0])
            self.test_mask[test_mal_indices] = True

        self.val_index_tensor = self.train_mask.nonzero().t().contiguous()[0]
        self.test_index_tensor = self.test_mask.nonzero().t().contiguous()[0]
        self.domain_mask[self.test_index_tensor] = False
        
        print('Train', sum(self.train_mask), 'Test', sum(self.test_mask))
        
    def __init_labelfeature(self, labelfeature_names):
        if not self.onehot_labelfeature:
            for col in labelfeature_names:
                self.nodes_df.loc[self.test_index_tensor,col]=0.5
        else:
            self.nodes_df.loc[self.test_index_tensor,labelfeature_names[0]]=0
            self.nodes_df.loc[self.test_index_tensor,labelfeature_names[1]]=0
            self.nodes_df.loc[self.test_index_tensor,labelfeature_names[2]]=1

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def define_batch(self, initial_nodes):
        each_layer_sampling_size = -1
        return NeighborLoader(self.data,
            num_neighbors=[each_layer_sampling_size]*self.num_layers,
            batch_size=self.batch_size,
            input_nodes=initial_nodes, 
            directed=False)
