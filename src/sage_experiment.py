import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from tqdm import tqdm, trange
from src.score import score, score_batch
import os.path as osp

from gnn.gcn_sage import Gcn
from src.experiment import Experiment
class SAGE_Experiment(Experiment): 
    def __init__(self, data, args):
        super(SAGE_Experiment, self).__init__(data, args)
        self.val_index_tensor = data.validation_mask.nonzero().t().contiguous()[0]
        self.test_index_tensor = data.test_mask.nonzero().t().contiguous()[0]
        self.args=args
    
    def define_batch(self, initial_nodes):
        if self.args.fanout is not None:
            return NeighborLoader(self.data,
                num_neighbors=self.args.fanout,
                weight_attr= 'edge_weight' if self.args.weighted else None,
                batch_size=self.batch_size,
                input_nodes=initial_nodes, 
                subgraph_type='induced')
    
    def __prepare_model(self,data) :
        model = Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes=2, num_layers=self.num_layers, 
                        model_type=self.model_type, lstm=self.args.lstm)
        if self.model_file != None  and osp.exists(self.model_file):
            try:
                model.load_state_dict(torch.load(self.model_file))
            except:
                print('MODEL LOAD FAILURE!!!')
        return model.to(self.device)
    
    
    def train(self):
        
        outer_batches = self.define_batch(self.val_index_tensor)
        
        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        trainer = trange(self.args.epoch)
        
        for epoch in trainer:
            model.train()
            optimizer.zero_grad()
            
            combined_pred = []
            combined_target = []
            
            for batch in outer_batches:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out[batch.validation_mask], batch.y[batch.validation_mask])
                loss.backward()
                
                combined_pred.append(out[batch.validation_mask])
                combined_target.append(batch.y[batch.validation_mask])
            
            optimizer.step()

            val_loss, val_acc = self.__test(model, data, data.test_mask, getloss=True)
            
            if self.earlystop(val_loss.cpu(), val_acc, model): 
                trainer.set_description(f'''Early stopping: {epoch:03d}, Loss {val_loss:.4f}, Val acc: {val_acc:.4f}''')
                break
                
            if epoch % 10 == 0:
                val_acc = score_batch(combined_pred, combined_target)
                test_acc = self.__test(model, data, data.test_mask)
                trainer.set_description(f'''Epoch: {epoch:03d}, Loss {loss:.4f}, Train: {val_acc['acc']:.4f}, Val: {test_acc['acc']:.4f}''')
        self.earlystop.load_checkpoint(model)
        return model
    
    # @torch.no_grad()
    def __test(self, model, data_test, mask, getloss=False):
        model.eval()
        test_batches = self.define_batch(self.test_index_tensor)
        
        with torch.no_grad():
            losses = []
            combined_pred = []
            combined_target = []

            for batch in test_batches:
                pred_raw = model(batch.x, batch.edge_index)
                c_loss = F.nll_loss(pred_raw[batch.test_mask], batch.y[batch.test_mask]) 
                losses.append(c_loss)

                combined_pred.append(pred_raw[batch.test_mask])
                combined_target.append(batch.y[batch.test_mask])

        combined_pred = torch.cat(combined_pred, dim=0)
        pred = combined_pred.argmax(dim=1)
        combined_target = torch.cat(combined_target, dim=0)

        if getloss: 
            total_loss = sum(losses) 
            return total_loss, score(pred, combined_target)['acc']
        return score(pred, combined_target)
    
    def test(self, model_file, data_test, 
            show_curve=False, return_pred=False):
        
        model = self.get_model(model_file, data_test)
        model = model.to(self.device)
        return self.__test(model, data_test, data_test.test_mask)
    
    
    def predict(self, model_file): 
        initial_nodes = self.data.train_mask.nonzero().t().contiguous()[0]
        self.batch_size = len(initial_nodes)
        test_batches = self.define_batch(initial_nodes)
        model = self.get_model(model_file, self.data)
        model.to(self.data.x.device)
        
        with torch.no_grad():
            combined_pred = []
            pred_ids = []
            for batch in test_batches:
                pred_raw = model(batch.x, batch.edge_index)
                combined_pred.append(pred_raw[batch.train_mask])
                pred_ids.append(batch.n_id[batch.train_mask])

        combined_ids = torch.cat(pred_ids, dim=0)
        combined_pred = torch.cat(combined_pred, dim=0)
        sorted_indices = torch.argsort(combined_ids)
        combined_pred = combined_pred[sorted_indices]
        pred_raw = F.softmax(combined_pred, dim=1)
        return pred_raw.t()[0]