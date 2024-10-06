import sys
from src.learning_curve import LearningCurve 
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import trange

from src.score import score
from gnn.heterosage import HeteroGNN, HAN
from src.experiment import Experiment


class Sage_Hetero(Experiment): 
    def __init__(self, data, args):
        super(Sage_Hetero, self).__init__(data, args)
        self.val_index_tensor = data['domain'].validation_mask.nonzero().t().contiguous()[0]
    
    def define_batch(self, mask):
        if self.args.fanout is not None:
            return NeighborLoader(self.data,
                num_neighbors=self.args.fanout,
                # weight_attr= 'edge_weight' if self.args.weighted else None,
                batch_size=self.batch_size,
                input_nodes=('domain', mask), # It should be training and testing nodes
                subgraph_type='induced')
                                 
    def get_model(self,data):
        return self.__prepare_model(data)
    def __prepare_model(self,data) :
        
        if self.args.model_type == 'han':
            model = HAN(data.metadata(), hidden_channels=self.dim, out_channels=2, 
                      num_layers=self.num_layers)
        else:
            model = HeteroGNN(data.metadata(), hidden_channels=self.dim, 
                    out_channels=2, num_layers=self.num_layers)
            
        return model.to(self.device)

    def train(self):
        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)  # Use neighbor loader if batching needed
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        epoch_counter = trange(self.epoch)
        for epoch in epoch_counter:
            model.train()
            optimizer.zero_grad()
            current_loss = 0

            data = data.to(self.device)
            out = model(data.x_dict, data.edge_index_dict)  # Forward pass

            # Compute the loss using the entire validation set (adjust mask as needed)
            loss = F.cross_entropy(out[data['domain'].validation_mask], data['domain'].y[data['domain'].validation_mask])
            current_loss += loss.item()

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Every 10 epochs, evaluate and print results
            if epoch % 10 == 0:
                val_acc = self.__test(model, data, istest=False)
                test_acc = self.__test(model, data, istest=True)
                epoch_counter.set_description(f'''Epoch: {epoch:03d}, Loss: {current_loss:.4f}, Train: {val_acc['acc']:.4f}, Test: {test_acc['acc']:.4f}''')

        return model
    
    @torch.no_grad()
    def __test(self, model, data_test, istest=True):
        model.eval()

        test_batches = self.define_batch(data_test['domain'].test_mask if istest else data_test['domain'].validation_mask)
        all_preds = []
        all_labels = []

        for batch in test_batches:
            pred = model(batch.x_dict, batch.edge_index_dict).argmax(dim=1)
            batch_mask = batch['domain'].test_mask if istest else batch['domain'].validation_mask

            all_preds.extend(pred[batch_mask].tolist())
            all_labels.extend(batch['domain'].y[batch_mask].tolist())

        return score(all_preds, all_labels)

    
    @torch.no_grad()
    def test(self, model_file, data_test, raw_directory='./data/raw_results/', save_embedding=False):
        modelLoaded = self.__prepare_model(data_test)
        modelLoaded.load_state_dict(torch.load(model_file))
        modelLoaded.eval()
        test_indices = data_test['domain'].test_index_tensor

        if save_embedding:
            with torch.no_grad():
                LearningCurve.save_prob_result(f'{self.experiment_id}_{self.extra}', 
                                            torch.exp(modelLoaded(data_test.x_dict, data_test.edge_index_dict).cpu()), 
                                            data_test['domain'].y.cpu(), 
                                            test_indices.cpu(), 
                                            [], # domain names are passed, I do not jnow why 
                                            raw_directory)

        return self.__test(modelLoaded, data_test, istest=True)

