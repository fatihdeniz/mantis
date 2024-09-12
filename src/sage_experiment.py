import torch
import torch.nn.functional as F
# from pyg_source.torch_geometric.loader import NeighborLoader
from torch_geometric.loader import NeighborLoader
# from torch_geometric.loader import NeighborLoader

import sys
import numpy as np
from src.score import score
from gnn.gcn_sage import Gcn
# from src.labelfeature_experiment import LabelFeature_Experiment
from src.experiment import Experiment

class SAGE_Experiment(Experiment): 
    def __init__(self, data, args):
        super(SAGE_Experiment, self).__init__(data, args)
        self.val_index_tensor = data.validation_mask.nonzero().t().contiguous()[0]
        self.args=args
    
    def define_batch(self, initial_nodes):
        return NeighborLoader(self.data,
            num_neighbors=self.args.fanout,
            # num_neighbors=[-1,25,10],
            batch_size=self.batch_size,
            input_nodes=initial_nodes, # It should be training and testing nodes
            directed=False)
    
    def __prepare_model(self,data) :
        return Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes=2, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)
        
    
    
    def train(self):
        outer_batches = self.define_batch(self.val_index_tensor)
        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in range(self.args.epoch):
            model.train()
            optimizer.zero_grad()
            
            for batch in outer_batches:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out[batch.validation_mask], batch.y[batch.validation_mask])
                loss.backward()

            optimizer.step()

            val_loss, val_acc = self.__test(model, data, data.test_mask, getloss=True)
            if self.earlystop(val_loss.cpu(), val_acc, model): 
                print('Early stopping epoch:', epoch)
                break

            if epoch % 100 == 0:
                val_acc, test_acc = self.__test(model, data, data.validation_mask), self.__test(model, data, data.test_mask)
                print(f'''Epoch: {epoch:03d}, Loss {loss:.4f}, Val: {val_acc['acc']:.4f}, Test: {test_acc['acc']:.4f}''')
        self.earlystop.load_checkpoint(model)
        return model
    
    @torch.no_grad()
    def __test(self, model, data_test, mask, getloss=False):
        model.eval()
        with torch.no_grad():
            pred_raw = model(data_test.x, data_test.edge_index)
        pred = pred_raw.argmax(dim=1)
        if getloss: 
            return F.nll_loss(pred_raw[mask], data_test.y[mask]), score(pred[mask], data_test.y[mask])['acc']
        return score(pred[mask], data_test.y[mask])
    