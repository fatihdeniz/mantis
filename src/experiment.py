from os import device_encoding
import sys
import torch
import numpy
import pandas as pd
import os.path as osp

import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from gnn.gcn_sage import Gcn
from gnn.gat import GAT
from src.learning_curve import LearningCurve
from src.score  import roc, prec_recall, score
from src.earlystopping import EarlyStopping

class Experiment(): 
    def __init__(self, data, args):
        model_type = args.model_type
        labelfeature_names=args.labelfeature_names
        epoch=args.epoch
        num_layers=args.num_layers
        dim=args.dim
        batch_size=args.outer_batch_size
        experiment_id=args.experiment_id
        gpu_id=args.gpu_id
        inner_batch_size=args.inner_batch_size
        extra=args.extra
        
        self.onehot_labelfeature = (len(labelfeature_names) > 2)
        self.args = args
        
        assert model_type in ['gcn', 'sage', 'gat']

        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu_id != -1 else 'cpu')
        if self.device != torch.device('cpu') and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(1.0, device=torch._C._cuda_getDevice())
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.dim = dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.experiment_id = experiment_id
        self.extra = extra
        self.data = data
        # self.val_index_tensor = data.validation_mask.nonzero().t().contiguous().tolist()[0]
        # print(self.val_index_tensor)
        
        self.model_file = args.model_file
        if self.model_file == None:
            self.earlystop = EarlyStopping(patience=100, verbose=False, delta=0.0005, filename=model_type + str(experiment_id))
        else:
            self.earlystop = EarlyStopping(patience=100, verbose=False, delta=0.0005, path=osp.dirname(self.model_file), filename=osp.basename(self.model_file))

    def __prepare_model(self, data) :
        if(self.model_type != 'gat'):
            model = Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes=2, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)
        else:
            model = GAT(data.num_features, 2).to(self.device) 
        if self.model_file != None  and osp.exists(self.model_file):
            model.load_state_dict(torch.load(self.model_file))
        return model
        
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def define_batch(self, initial_nodes):
        each_layer_sampling_size = -1
        return NeighborLoader(self.data,
            num_neighbors=[each_layer_sampling_size]*self.num_layers,
            batch_size=self.batch_size,
            input_nodes=self.data.validation_mask, # It should be training and testing nodes
            directed=False)

    def train_batches(self):
        val_index_tensor = self.data.validation_mask.nonzero().t().contiguous().tolist()[0]
        outer_batches = self.define_batch(val_index_tensor)
        model = self.__prepare_model(self.data)
        
        data = self.data.to(self.device)
        data.validation_mask[val_index_tensor] = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epoch):
            
            model.train()
            optimizer.zero_grad()
            total_loss,total_correct,total_test = 0,0,0
            
            for outer_batch in outer_batches:
                # print(outer_batch)

                val_index_batch = outer_batch.train_mask.nonzero().t().contiguous().tolist()[0][:outer_batch.batch_size]
                numpy.random.shuffle(val_index_batch) 
                
                batch_features = outer_batch.x
                
                prev_batch = None
                for index_batch in self.batch(val_index_batch, self.inner_batch_size):    

                    if prev_batch is not None:
                        ####### TO ADD LABEL FEATURE TO A NODE  
                        for prev in prev_batch:
                            batch_features[prev][outer_batch.y[prev]] = 1
                            batch_features[prev][1-outer_batch.y[prev]] = 0
                            
                            if self.onehot_labelfeature:
                                batch_features[prev][2] = 0
                            
                            outer_batch.validation_mask[prev] = False

                    ###### TO REMOVE LABEL FEATURE FROM A NODE
                    for index in index_batch:
                        
                        batch_features[index][0] = 0 if self.onehot_labelfeature else 0.5
                        batch_features[index][1] = 0 if self.onehot_labelfeature else 0.5
                        
                        if self.onehot_labelfeature:
                            batch_features[index][2] = 1
                        # print(batch_features[index][0], batch_features[index][1])
                        outer_batch.validation_mask[index] = True
                    
                    prev_batch = index_batch
                    outer_batch.x = batch_features

                    outer_batch = outer_batch.to(self.device)
                    out = model(outer_batch.x, outer_batch.edge_index)
#                     out = model(outer_batch.x, outer_batch.edge_index, outer_batch)
                    loss = F.nll_loss(out[outer_batch.validation_mask],outer_batch.y[outer_batch.validation_mask])
                    loss.backward()

                    total_loss += loss
                    total_test += len(index_batch)
                    total_correct += (out[outer_batch.validation_mask].argmax(dim=1) == outer_batch.y[outer_batch.validation_mask]).sum()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"Epoch: {epoch:03d} Loss {total_loss} \t Total Correct {total_correct}/{total_test} \t Val {(total_correct/total_test)}")

#             train_acc, val_acc, test_acc = self.__test(model, data)
#             print(f'Epoch: {epoch:03d}, Loss {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#                             f'Test: {test_acc:.4f}')
        return model

    def train(self):

        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epoch):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.validation_mask], data.y[data.validation_mask])

            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                val_acc, test_acc = self.__test(model, data, data.validation_mask), self.__test(model, data, data.test_mask)
                print(f'''Epoch: {epoch:03d}, Loss {loss:.4f}, Train: {val_acc['acc']:.4f}, Val: {test_acc['acc']:.4f}''')
        return model
    
    @torch.no_grad()
    def get_model(self, model_file, data_test, on_cpu=False):

        modelLoaded = self.__prepare_model(data_test)
        if on_cpu:
            modelLoaded.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        else:
            modelLoaded.load_state_dict(torch.load(model_file))
        modelLoaded.eval()
        
        return modelLoaded
    
    def save_model(self, model, model_file):
        torch.save(model.state_dict(), model_file)
    
    @torch.no_grad()
    def test(self, model_file, data_test, 
            show_curve=False):

        model = self.get_model(model_file, data_test)
        test_indices = data_test.test_mask.nonzero().t().contiguous().tolist()[0]
        mask = data_test.test_mask
        model.eval()
        with torch.no_grad():
            pred_raw = model(data_test.x, data_test.edge_index)
            pred = pred_raw.argmax(dim=1)

        if show_curve:
            pred_raw = F.softmax(pred_raw[mask], dim=1)
            roc(pred_raw.cpu(), data_test.y.cpu()[mask])
            prec_recall(pred_raw.cpu(), data_test.y.cpu()[mask])
    
        return score(pred[mask], data_test.y[mask])
        # try:
        #     with torch.no_grad():
        #         LearningCurve.save_prob_result(f'{self.experiment_id}_{self.extra}', 
        #                                        torch.exp(modelLoaded(data_test.x, data_test.edge_index)), 
        #                                    data_test.y, 
        #                                    test_indices, 
        #                                    [], # domain names are passed, I do not know why 
        #                                    raw_directory)
        # except:
        #     print('Could not generate roc curve!')

        # return self.__test(modelLoaded, data_test, data_test.test_mask)
#         return self.__test2(modelLoaded, data_test)
        
    @torch.no_grad()
    def __test(self, model, data_test, mask):
        model.eval()
        with torch.no_grad():
            pred_raw = model(data_test.x, data_test.edge_index)
            pred = pred_raw.argmax(dim=1)
        
        return score(pred[mask], data_test.y[mask])
    
    @torch.no_grad()
    def __test2(self, model, data_test):
        """
        This test function is for label feature approach, currently not being used
        It can be used if testing nodes have their true labels in their label features
        """
        model.eval()

        total_correct,total_test = 0,0
        outer_batches = self.define_batch(self.test_index_tensor)
           
        for outer_batch in outer_batches:
            print(outer_batch)
            test_index_batch = outer_batch.test_mask.nonzero().t().contiguous().tolist()[0]
            outer_batch.test_mask[test_index_batch] = False 
            test_index_batch = test_index_batch[:outer_batch.batch_size]
            
            features = outer_batch.x
            prev = None  

            for t_index in test_index_batch:    
#                 print(t_index)
                if prev is not None:
                    features[prev][outer_batch.y[prev]] = 0.5
                    features[prev][1-outer_batch.y[prev]] = 0.5
                    outer_batch.test_mask[prev] = False

            #         print(f'Prev node {prev}' )

                prev = t_index

                ###### TO REMOVE LABEL FEATURE FROM A NODE
                features[t_index][0] = 0.5
                features[t_index][1] = 0.5
                outer_batch.test_mask[t_index] = True

                if outer_batch.test_mask.sum()>1:
                    print('Active test size', )
                ##### TEST BATCH
            #     print(f'Number of active masks {numpy.unique(data.test_mask, return_counts = True)}');
            #     print(f'Current node {t_index}')
                #####
                outer_batch.x= features
                outer_batch = outer_batch.to(self.device)
                
                with torch.no_grad():
                    pred = model(outer_batch.x, outer_batch.edge_index, outer_batch).argmax(dim=1)

                correct = (pred[outer_batch.test_mask] == outer_batch.y[outer_batch.test_mask]).sum()
                total_correct += correct
                total_test += 1
        return [0, 0,  float(total_correct/total_test)]
    
    @torch.no_grad()
    def save_embedding(self, model_file, data, output='./data/embedding/embedding.csv'):

        model = self.get_model(model_file, data)

        with torch.no_grad():
            embedding = model(data.x, data.edge_index, save_embedding=True)

        emb_df = pd.DataFrame(embedding.cpu().numpy())
        emb_df.to_csv(output, index=False)
        print(embedding.size())