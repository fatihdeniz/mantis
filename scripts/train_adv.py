import os
import sys
import torch
from datetime import date, datetime
from time import time

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import importlib as imp

import torch
import torch_geometric
print('Torch', torch.__version__)
print("PyTorch Geometric Version:", torch_geometric.__version__)
print("CUDA Version:", torch.version.cuda)

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
    
from src import experiment_adv
from src import loader

SAGE_Experiment=experiment_adv.SAGE_Experiment
Loader = loader.Loader

from src.features import feature_labels
from src.score  import roc, prec_recall, score
from src import config_attack as config 
args1 = config.parse()

# Select the GPU with the lowest memory usage
from utils import gpu_util
best_gpu = gpu_util.pick_gpu_lowest_memory()
args1.gpu_id = best_gpu 
args1.experiment_id = int(args1.epsilon*100)


data_path = module_path # specify your data path 
nodes_file = os.path.join(module_path,'data/fqdn_apex_nodes2.csv')
edges_file = os.path.join(module_path,'data/fqdn_apex_edges.csv')

model_file = os.path.join(module_path,f'models/sample_sage_{args1.experiment_id}.pkl')
args1.model_file = model_file

print('Arguments:', args1)

def test(exp, loader, date, experiment_id=0):
    print('Test', model_file)
    import torch.nn.functional as F
    @torch.no_grad()
    def pred_conf1(exp, loader, confidence=0.5): 
        model = exp.prepare_model(loader.data)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        with torch.no_grad():
            pred_raw = model(loader.data.x, loader.data.edge_index)
        pred_raw = F.softmax(pred_raw, dim=1)
        pred = torch.tensor((pred_raw >= confidence).t()[1], dtype=int)
        pred, pred_raw = pred.cpu(), pred_raw.cpu()
        loader.data.y = loader.data.y.cpu()
        mask = loader.data.test_mask.cpu()
        
        roc(pred_raw[mask], loader.data.y[mask])
        prec_recall(pred_raw[mask], loader.data.y[mask])
        accs = score(pred[mask], loader.data.y[mask])
        
        unknown_mal = sum(pred[loader.unknown_mask])
        accs['unknown_mal'] = unknown_mal.item()
        accs['unknown_ben'] = (sum(loader.unknown_mask) - unknown_mal).item()
        return accs

    accs = pred_conf1(exp, loader)
    return accs

result = []


loader = Loader(nodes_file, edges_file, feature_labels, args1)
exp = SAGE_Experiment(loader.data, args1)
model = exp.train(adversarial=False, mimicIP=True, mintA=True, clp=args1.epsilon)
# model = exp.train_batches()
torch.save(model.state_dict(), model_file)

cleanAccs = exp.cleantest(model, loader.data, loader.test_mask)
cleanAccs['clp'] = args1.experiment_id
cleanAccs['type'] = 'clean'
result.append(cleanAccs)
print('Clean', cleanAccs)

advAccs = exp.mimiciptest(model, loader.data, loader.test_mask, mintA=False, clp=args1.epsilon)
advAccs['clp'] = args1.experiment_id
advAccs['type'] = 'mimicIP'
result.append(advAccs)
print('mimicIP', advAccs)

advAccs = exp.mimiciptest(model, loader.data, loader.test_mask, mintA=True, clp=args1.epsilon)
advAccs['clp'] = args1.experiment_id
advAccs['type'] = 'mintA'
result.append(advAccs)
print('mintA', advAccs)
        
result_df = pd.DataFrame(result)
result_df.to_csv(f'data/results_{args1.model_type}_adv_{args1.experiment_id}.csv', index=False)
