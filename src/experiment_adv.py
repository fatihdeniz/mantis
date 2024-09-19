import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

import sys
import numpy as np
import os.path as osp
from tqdm import trange
from src.score import score
from src.earlystopping import EarlyStopping

from gnn.gcn_sage_attack import Gcn
from src.experiment import Experiment

class SAGE_Experiment(Experiment): 
    def __init__(self, data, args):
        super(SAGE_Experiment, self).__init__(data, args)
        self.val_index_tensor = data.validation_mask.nonzero().t().contiguous()[0]
        self.test_index_tensor = data.test_mask.nonzero().t().contiguous()[0]
        
        self.earlystop = EarlyStopping(patience=200, verbose=False, delta=0.0005, path=osp.dirname(self.model_file), filename=osp.basename(self.model_file))
    
    def define_batch(self, initial_nodes):
        return NeighborLoader(self.data,
            num_neighbors=self.args.fanout,
            # num_neighbors=[-1,25,10],
            batch_size=self.batch_size,
            input_nodes=initial_nodes, # It should be training and testing nodes
            subgraph_type='induced')
    
    def __prepare_model(self,data) :
        model = Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes=2, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)
        if self.model_file != None  and osp.exists(self.model_file):
            model.load_state_dict(torch.load(self.model_file))
            print('MODEL FOUND!!!', self.model_file)
        else:
            print('NO MODEL FOUND!!!', self.model_file)
            
        return model
        
    def prepare_model(self,data):
        return Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes=2, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)
    
    def getAdvX(self, model, batch, mask, noise_epsilon=0.001,pdg_step=10,epsilon=0.01,clipLmt=0.03):
        x_data = batch.x
        batch.y = batch.y.to(x_data.device)
        mask = mask.to(x_data.device)
        model = model.to(x_data.device) 
        
        X_adv = x_data.clone().detach()
        noise = torch.normal(0, noise_epsilon, size=X_adv.shape).to(x_data.device)
        X_adv += noise
        model.eval()
        for i in range(pdg_step):
            X_adv = X_adv.requires_grad_(True) 
            out = model(X_adv, batch.edge_index)
            loss = F.nll_loss(out[mask], batch.y[mask])
            grad_X = torch.autograd.grad(loss, X_adv)[0]
            X_adv = X_adv.requires_grad_(False) 
            X_adv += epsilon* grad_X.sign()
            eta = torch.clamp(X_adv-x_data, min=-1*clipLmt, max=clipLmt)
            X_adv = x_data.clone().detach() + eta
            
        X_adv.detach_()
        
        return X_adv
    
    
    def getRandomEdge(self, batch, ratio=0.05, drop=True, add=False):
        node_num, _ = batch.x.size()
        _, edge_num = batch.edge_index.size()
        perturb_num = int(edge_num * ratio)

        edge_index = batch.edge_index.clone().detach()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2, -1).long().to(edge_index.device)

        if drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

        if add:
            idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return new_edge_index.to(batch.x.device)
    
    def getPgdEdge(self, batch, ratio=0.05, drop=True, add=False):
        node_num, _ = batch.x.size()
        _, edge_num = batch.edge_index.size()
        perturb_num = int(edge_num * ratio)
        
        edge_weight = torch.ones(edge_num).to(edge_index.device)
        
        edge_index = batch.edge_index.clone().detach()
        idx_remain = edge_index
        idx_add = torch.tensor([]).reshape(2, -1).long().to(edge_index.device)

        if drop:
            idx_remain = edge_index[:, np.random.choice(edge_num, edge_num-perturb_num, replace=False)]

        if add:
            idx_add = torch.randint(node_num, (2, perturb_num))

        new_edge_index = torch.cat((idx_remain, idx_add), dim=1)
        new_edge_index = torch.unique(new_edge_index, dim=1)

        return new_edge_index.to(batch.x.device)
    
    def pgdMimicIP(self, model, batch, mask, noise_epsilon=0.001, pdg_step=10, epsilon=0.01, clipLmt=0.03, perturbation=0.10, attack_type=None):
        x_data = batch.x
        batch.y = batch.y.to(x_data.device)
        mask = mask.to(x_data.device)
        model = model.to(x_data.device) 
        
        _, edge_num = batch.edge_index.size()
        perturb_num = int(edge_num * perturbation)
        
        edge_weight_adv = batch.edge_weight.clone().detach()
        noise = torch.normal(0, noise_epsilon, size=edge_weight_adv.shape).to(x_data.device)
        edge_weight_adv += noise
        model.eval()
        for i in range(pdg_step):
            edge_weight_adv = edge_weight_adv.requires_grad_(True) 
            out = model(batch.x, batch.edge_index, edge_weight=edge_weight_adv)
            loss = F.nll_loss(out[mask], batch.y[mask])
            grad_X = torch.autograd.grad(loss, edge_weight_adv)[0]
            edge_weight_adv = edge_weight_adv.requires_grad_(False) 
            adj_changes = epsilon * grad_X.sign()
            
            if attack_type == "IP_Mimicry":
                # Mimicry attack: introduce perturbations in connections between domain and IP nodes
                mal_nodes = (batch.y == 1).nonzero().squeeze()
                ip_nodes = (batch.ip_mask == 1).nonzero().squeeze()
                perturb_domain = torch.randint(len(mal_nodes), (perturb_num,))
                perturb_ip = torch.randint(len(ip_nodes), (perturb_num,))
                perturb_idx = torch.cat((mal_nodes[perturb_domain], ip_nodes[perturb_ip]), dim=0)
                perturb_val = torch.zeros(perturb_num).to(x_data.device)  # perturbations for IP mimicry
                
                adj_changes[:, perturb_idx] = perturb_val.view(1, -1)

            eta = self.projection(perturb_num, adj_changes, clipLmt)
            edge_weight_adv = batch.edge_weight.clone().detach() + eta
            
        edge_weight_adv.detach_()
        
        return edge_weight_adv
    
    
    def minta_attack(self, model, batch, mask, max_group_size, perturbation):
        
        x_data = batch.x
        batch.y = batch.y.to(x_data.device)
        mask = mask.to(x_data.device)
        model = model.to(x_data.device) 
        model.eval()
        with torch.no_grad():
            # Step 1: Retrieve malicious domains based on the mask
            malicious_indices = (batch.y == 1).nonzero(as_tuple=True)[0]
            malicious_indices = malicious_indices[mask[malicious_indices] == 1]

            # Step 2: Group malicious domains
            groups = [malicious_indices[i:i + max_group_size] for i in range(0, len(malicious_indices), max_group_size)]

            # Initialize changes
            edge_index = batch.edge_index.clone().detach()
            edge_type = batch.edge_type.clone().detach()
            num_perturbations = 0

            perturbation_limit = int(malicious_indices.size()[0] * perturbation)

            # Step 3: For each group, find and flip edges to maximize loss
            model.eval()
            for group in groups:
                if num_perturbations >= perturbation_limit:
                    break

                # Initialize best loss and best flip
                max_loss = float('-inf')
                best_flip = None

                # Evaluate all possible single flips within the group
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        # Temporarily flip the edge
                        temp_edge_index, _ = self.flip_edge(edge_index, edge_type, group[i], group[j])

                        # Compute loss
                        out = model(batch.x, temp_edge_index)
                        loss = F.nll_loss(out[mask], batch.y[mask]).item()

                        # Check if this flip is better
                        if loss > max_loss:
                            max_loss = loss
                            best_flip = (group[i], group[j])

                # Apply the best flip
                if best_flip:
                    edge_index, edge_type = self.flip_edge(edge_index, edge_type, best_flip[0], best_flip[1])
                    num_perturbations += 1

            # Step 4: Return the modified edge_index
            return edge_index

    def flip_edge(self, edge_index, edge_type, node1, node2):
        all_edges = edge_index.t().clone().detach()

        all_types = edge_type.clone().detach()

        is_node1 = (all_edges[:, 0] == node1) | (all_edges[:, 1] == node1)
        is_node2 = (all_edges[:, 0] == node2) | (all_edges[:, 1] == node2)
        is_edge_type2 = (all_types == 2)
        is_edge_type0 = (all_types == 0)

        # Find subdomains (edge_type == 2)
        sub1 = all_edges[(all_edges[:, 0] == node1) & is_edge_type2][:, 1]
        sub2 = all_edges[(all_edges[:, 0] == node2) & is_edge_type2][:, 1]

        # Extract IP connections (edge_type == 0)
        node1_ips = all_edges[(all_edges[:, 0] == node1) & is_edge_type0]
        node2_ips = all_edges[(all_edges[:, 0] == node2) & is_edge_type0]

        # Extract reverse IP connections (for completeness)
        node1_ips_rev = all_edges[(all_edges[:, 1] == node1) & is_edge_type0]
        node2_ips_rev = all_edges[(all_edges[:, 1] == node2) & is_edge_type0]

        # Retain non-subdomain and non-node edges
        retain_mask = ~(is_node1 | is_node2 | ((all_edges[:, 0].unsqueeze(1) == sub1).any(dim=1) & is_edge_type0) | ((all_edges[:, 0].unsqueeze(1) == sub2).any(dim=1) & is_edge_type0))
        retain_edges = all_edges[retain_mask]
        retain_types = all_types[retain_mask]
    
        # retain_edges = all_edges[~(is_node1 | is_node2)]
        # retain_types = all_types[~(is_node1 | is_node2)]

        # Swap IP connections for nodes and subdomains
        node1_ips[:, 0], node2_ips[:, 0] = node2, node1
        node1_ips_rev[:, 1], node2_ips_rev[:, 1] = node2, node1

        # Create new subdomain-IP connections
        sub1_ips = torch.vstack([sub1.repeat(len(node2_ips)), node2_ips[:, 1].repeat_interleave(len(sub1))]).t()
        sub2_ips = torch.vstack([sub2.repeat(len(node1_ips)), node1_ips[:, 1].repeat_interleave(len(sub2))]).t()

        # Reverse connections for subdomains
        sub1_ips_rev = torch.vstack([node2_ips[:, 1].repeat(len(sub1)), sub1.repeat_interleave(len(node2_ips))]).t()
        sub2_ips_rev = torch.vstack([node1_ips[:, 1].repeat(len(sub2)), sub2.repeat_interleave(len(node1_ips))]).t()

        # Combine all edges and types
        new_edge_index = torch.cat([retain_edges, node1_ips, node2_ips, sub1_ips, sub2_ips, node1_ips_rev, node2_ips_rev, sub1_ips_rev, sub2_ips_rev], dim=0)
        new_edge_type = torch.cat([retain_types, torch.zeros_like(node1_ips[:, 0]), torch.zeros_like(node2_ips[:, 0]), 
                                   torch.zeros_like(sub1_ips[:, 0]), torch.zeros_like(sub2_ips[:, 0]),
                                   torch.zeros_like(node1_ips_rev[:, 1]), torch.zeros_like(node2_ips_rev[:, 1]),
                                   torch.zeros_like(sub1_ips_rev[:, 1]), torch.zeros_like(sub2_ips_rev[:, 1])], dim=0)

        return new_edge_index.t(), new_edge_type



    
    def mimicIP(self, model, batch, mask, perturbation=0.10):
        x_data = batch.x
        batch.y = batch.y.to(x_data.device)
        mask = mask.to(x_data.device)
        model = model.to(x_data.device) 
        model.eval()
        with torch.no_grad():

            try:
                mal_index = ((batch.y == 1) & (mask == 1)).nonzero().squeeze()
                popular_ip_index = (batch.popular_ip_mask == 1).nonzero().squeeze()

                perturb_num = min(int(mal_index.size()[0] * perturbation), len(popular_ip_index))
                # print(perturbation, perturb_num)
                if perturb_num == 0:
                    return batch.edge_index
            except Exception as e:
                print('Exception !!!', repr(e))
                return batch.edge_index
            
            selected_mal_indices = mal_index[torch.randint(0, mal_index.size(0), (perturb_num,))]
            prediction_score_initial = model(x_data, batch.edge_index)[selected_mal_indices]
            prediction_score_initial = F.softmax(prediction_score_initial, dim=1)[:, 1]

            # Tensor to store the change in prediction score 
            # its size is popular_ip_index * perturb_num
            score_change = torch.zeros((len(popular_ip_index),  perturb_num), dtype=torch.float, device=x_data.device)

            popular_ip_index = torch.cat([popular_ip_index, popular_ip_index])
            # For each popular_ip_index, add a connection from each of the selected perturb_num nodes to the selected popular IP
            for i, ip_index in enumerate(popular_ip_index[:len(popular_ip_index)//2]):
                mal_to_ip_edges = torch.zeros((2, perturb_num), dtype=torch.long, device=x_data.device)
                rev_mal_to_ip_edges = torch.zeros((2, perturb_num), dtype=torch.long, device=x_data.device)

                mal_to_ip_edges[0,:] = selected_mal_indices
                mal_to_ip_edges[1,:] = popular_ip_index[i:i+len(selected_mal_indices)]

                rev_mal_to_ip_edges[0, :] = popular_ip_index[i:i+len(selected_mal_indices)]
                rev_mal_to_ip_edges[1, :] = selected_mal_indices

                modified_edge_index = torch.cat([batch.edge_index, mal_to_ip_edges, rev_mal_to_ip_edges], dim=1)
                updated_prediction_score = model(x_data, modified_edge_index)

                score_diff = prediction_score_initial - F.softmax(updated_prediction_score[selected_mal_indices], dim=1)[:,1]
                for j in range(len(selected_mal_indices)):
                    score_change[(i+j)%len(score_change), j] = score_diff[j]
            num_edges = score_change.size(1)
            sorted_indices = torch.argsort(score_change, dim=0)

            selected_ip = []
            selected_ip_set = set()

            for col in range(num_edges):
                for index in reversed(sorted_indices[:, col]):
                    if index.item() not in selected_ip_set:
                        selected_ip_set.add(index.item())
                        selected_ip.append(index.item())
                        break

            selected_edges = torch.zeros((2, perturb_num), dtype=torch.long, device=x_data.device)
            # selected_ip_tensor = torch.tensor(selected_ip, dtype=torch.long, device=x_data.device)
            selected_ip_tensor = popular_ip_index[selected_ip]

            rev_selected_edges = torch.zeros((2, perturb_num), dtype=torch.long, device=x_data.device)
            selected_edges[0,:] = selected_mal_indices
            selected_edges[1,:] = selected_ip_tensor

            rev_selected_edges[0, :] = selected_ip_tensor
            rev_selected_edges[1, :] = selected_mal_indices
            mimicIP = torch.cat([batch.edge_index, selected_edges, rev_selected_edges], dim=1)
            
            return mimicIP
        
    
    def pgdEdge(self, model, batch, mask, noise_epsilon=0.001,pdg_step=10,epsilon=0.01,clipLmt=0.03, perturbation=0.10):
        x_data = batch.x
        batch.y = batch.y.to(x_data.device)
        mask = mask.to(x_data.device)
        model = model.to(x_data.device) 
        
        _, edge_num = batch.edge_index.size()
        perturb_num = int(edge_num * perturbation)
        
        edge_weight_adv = batch.edge_weight.clone().detach()
        noise = torch.normal(0, noise_epsilon, size=edge_weight_adv.shape).to(x_data.device)
        edge_weight_adv += noise
        model.eval()
        for i in range(pdg_step):
            edge_weight_adv = edge_weight_adv.requires_grad_(True) 
            out = model(batch.x, batch.edge_index, edge_weight=edge_weight_adv)
            loss = F.nll_loss(out[mask], batch.y[mask])
            grad_X = torch.autograd.grad(loss, edge_weight_adv)[0]
            edge_weight_adv = edge_weight_adv.requires_grad_(False) 
            adj_changes = epsilon* grad_X.sign()
            
            eta = self.projection(perturb_num, adj_changes, clipLmt)
            # print(eta)
            # eta = torch.clamp(edge_weight_adv-edge_weigth, min=-1*clipLmt, max=clipLmt)
            edge_weight_adv = batch.edge_weight.clone().detach() + eta
            
        edge_weight_adv.detach_()
        
        return edge_weight_adv
    
    def projection(self, n_perturbations: int, adj_changes, epsilon):
        if torch.clamp(adj_changes, 0, 1).sum() > n_perturbations:
            left = (adj_changes - 1).min()
            right = adj_changes.max()
            miu = SAGE_Experiment.bisection(left, right, adj_changes, n_perturbations, epsilon)
            return torch.clamp(adj_changes.data - miu, min=0, max=1)
        else:
            return torch.clamp(adj_changes.data, min=0, max=1)
    
    def get_modified_adj(self, edge_weight):
        adj = edge_weight
        complementary = torch.ones_like(adj) - 2 * adj

        modified_adj = complementary * adj_changes + adj

        return modified_adj
    
    def bisection(a: float, b: float, adj_changes: torch.Tensor, n_perturbations: int, epsilon: float):
        def func(x):
            return torch.clamp(adj_changes - x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            if (func(miu) == 0.0):
                break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu
    
    def train(self, adversarial=False, mimicIP=False, mintA=False, clp=0.03, alpha=2, beta=1, gamma=1):
        outer_batches = self.define_batch(self.val_index_tensor)
        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        if clp == 0:
            adversarial = False
            mimicIP = False
        eps = clp/4
        skip = False
        epoch_range = trange(self.epoch)
        for epoch in epoch_range:
            model.train()
            optimizer.zero_grad()

            for batch in outer_batches:
                batch = batch.to(self.device)
                batch_outputs = []

                # Train with clean data
                clean_out = model(batch.x, batch.edge_index)
                clean_loss = alpha * F.nll_loss(clean_out[batch.validation_mask], batch.y[batch.validation_mask])
                loss = clean_loss

                if adversarial:
                    # Apply PGD attack
                    adv_x = self.getAdvX(model, batch, batch.validation_mask, epsilon=eps, clipLmt=clp)
                    adv_edge_weight = self.pgdEdge(model,batch,batch.validation_mask, epsilon=eps,clipLmt=clp)
                    
                    model.train()
                    adv_out = model(adv_x, batch.edge_index, adv_edge_weight)
                    adv_loss = beta * F.nll_loss(adv_out[batch.validation_mask], batch.y[batch.validation_mask]) 
                    loss += adv_loss

                if mintA:
                    # Apply MintA attack
                    mintAEdgeIndex = self.minta_attack(model, batch, batch.validation_mask, max_group_size=5, perturbation=clp)
                    
                    model.train()
                    mintA_out = model(batch.x, mintAEdgeIndex)
                    mintA_loss = gamma * F.nll_loss(mintA_out[batch.validation_mask], batch.y[batch.validation_mask])
                    loss += mintA_loss

                if mimicIP:
                    # Apply MimicIP attack
                    mimicIPEdgeIndex = self.mimicIP(model, batch, batch.validation_mask, clp)

                    model.train()
                    mimicIP_out = model(batch.x, mimicIPEdgeIndex)
                    mimicIP_loss = gamma * F.nll_loss(mimicIP_out[batch.validation_mask], batch.y[batch.validation_mask])
                    loss += mimicIP_loss

                loss.backward()

            optimizer.step()

            val_loss, val_acc = self.__test(model, data, data.test_mask, getloss=True)
            epoch_range.set_description(f'''Epoch: {epoch:03d}, Loss {val_loss:.4f}, Val: {val_acc:.4f}''')
            if self.earlystop(val_loss.cpu(), val_acc, model): 
                print('Early stopping epoch:', epoch)
                break
            
            # if epoch % 100 == 0:
            #     val_acc, test_acc = self.__test(model, data, data.validation_mask), self.__test(model, data, data.test_mask)
            #     print(f'''Epoch: {epoch:03d}, Loss {loss:.4f}, Val: {val_acc['acc']:.4f}, Test: {test_acc['acc']:.4f}''')
        self.earlystop.load_checkpoint(model)
        return model
    
    @torch.no_grad()
    def __test(self, model, data_test, mask, getloss=False):
        model.eval()
        with torch.no_grad():
            pred_raw = model(data_test.x, data_test.edge_index, data_test.edge_weight)
        pred = pred_raw.argmax(dim=1)
        if getloss: 
            return F.nll_loss(pred_raw[mask], data_test.y[mask]), score(pred[mask], data_test.y[mask])['acc']
        return score(pred[mask], data_test.y[mask])
    
    def advtest(self, model, data_test, mask, getloss=False, pdg_step=10,epsilon=0.01,clipLmt=0.03, perturbation=0.05):
        model.eval()
        adv_x = self.getAdvX(model,data_test,mask, pdg_step=pdg_step,epsilon=epsilon,clipLmt=clipLmt)
        adv_edge_weight = self.pgdEdge(model,data_test,mask, pdg_step=pdg_step,epsilon=epsilon,clipLmt=clipLmt, perturbation=perturbation)
        with torch.no_grad():
            pred_raw = model(adv_x, data_test.edge_index, adv_edge_weight)
        pred = pred_raw.argmax(dim=1)
        if getloss: 
            return F.nll_loss(pred_raw[mask], data_test.y[mask]), score(pred[mask], data_test.y[mask])['acc']
        return score(pred[mask], data_test.y[mask])
    
    @torch.no_grad()
    def cleantest(self, model, data_test, mask, getloss=False):
        model.eval()
        model = model.to(self.device)
        print('Inside', model, data_test.x.device)
        data_test = data_test.to(self.device)
        with torch.no_grad():
            pred_raw = model(data_test.x, data_test.edge_index)
        pred = pred_raw.argmax(dim=1)
        if getloss: 
            return F.nll_loss(pred_raw[mask], data_test.y[mask]), score(pred[mask], data_test.y[mask])['acc']
        return score(pred[mask], data_test.y[mask])
    
    def mimiciptest(self, model, data_test, mask, surrogate_model=None, mintA=False, clp=0.05):
        model.eval()
        test_batches = self.define_batch(self.test_index_tensor)
        
        with torch.no_grad():
            losses = []
            combined_pred = []
            combined_target = []

            for batch in test_batches:
                print("In mimic test clp value", clp)
                if mintA:
                    if surrogate_model:
                        mimicEdgeIndex = self.minta_attack(surrogate_model, batch, batch.test_mask, max_group_size=5, perturbation=clp)
                    else:
                        mimicEdgeIndex = self.minta_attack(model, batch, batch.test_mask, max_group_size=5, perturbation=clp)
                else:
                    mimicEdgeIndex = self.mimicIP(model, batch, batch.test_mask, clp)
                pred_raw = model(batch.x, mimicEdgeIndex)
                c_loss = F.nll_loss(pred_raw[batch.test_mask], batch.y[batch.test_mask]) 
                losses.append(c_loss)

                combined_pred.append(pred_raw[batch.test_mask])
                combined_target.append(batch.y[batch.test_mask])

        combined_pred = torch.cat(combined_pred, dim=0)
        pred = combined_pred.argmax(dim=1)
        combined_target = torch.cat(combined_target, dim=0)

        return score(pred, combined_target)
    
    # def mimiciptest(self, model, data_test, mask, clp=0.05):
    #     model.eval()
    #     adv_x = self.getAdvX(model,data_test,mask, pdg_step=pdg_step,epsilon=epsilon,clipLmt=clipLmt)
    #     adv_edge_weight = self.pgdEdge(model,data_test,mask, pdg_step=pdg_step,epsilon=epsilon,clipLmt=clipLmt, perturbation=perturbation)
    #     with torch.no_grad():
    #         pred_raw = model(adv_x, data_test.edge_index, adv_edge_weight)
    #     pred = pred_raw.argmax(dim=1)
    #     if getloss: 
    #         return F.nll_loss(pred_raw[mask], data_test.y[mask]), score(pred[mask], data_test.y[mask])['acc']
    #     return score(pred[mask], data_test.y[mask])