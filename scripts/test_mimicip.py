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

import torch.nn.functional as F

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
    
from src import experiment_attack
from src import loader

imp.reload(experiment_attack)
SAGE_Experiment=experiment_attack.SAGE_Experiment
imp.reload(loader)
Loader = loader.Loader

from src.features import feature_labels
from src.score  import roc, prec_recall, score
from src import config_attack as config 
args1 = config.parse()

sys.path.insert(0, '/export/sec02/fatih/Common')
from lib_jupyter import notebook_util
best_gpu = notebook_util.pick_gpu_lowest_memory()
# args1.gpu_id = best_gpu

path='/export/blacklist-forecast/blacklist_forecast/'
nodes_file_format = path +'graph_data/2022-{:02d}-01-2022-{:02d}-07_fqdn_apex_nodes.csv.log.norm' # pass month for this experiment
edges_file_format = path +'graph_data/2022-{:02d}-01-2022-{:02d}-07_fqdn_apex_edges.csv'

# model_file_format = "/export/sec02/fatih/paper_stats/data/attack/clean_2022-07-07.pkl" # month and experiment ID
# model_file_format = "/export/sec02/fatih/paper_stats/data/attack/attack_2022-07-{}_{}.pkl" # month and experiment ID
model_file_format = "/export/sec02/fatih/paper_stats/data/attack/mimicip1_2022-07-{}_{}.pkl" # month and experiment ID
roc_file_format = """/export/sec02/fatih/paper_stats/data/attack/attack_2022-{:02d}-07_{}.csv"""  
roc_file_format=None


def train(nodes_file, edges_file, model_file, feature_labels, coeffs):
    loader = Loader(nodes_file, edges_file, feature_labels, coeffs)

    exp = SAGE_Experiment(loader.data, coeffs)
    
    # model = exp.train(adversarial=True, mimicIP=False, clp=coeffs.epsilon)
    model = exp.train(adversarial=False, mimicIP=True, clp=coeffs.epsilon)
    # model = exp.train_batches()
    torch.save(model.state_dict(), model_file)
    return exp, model, loader

def save_roc(filename, pred, label, mask):
    df_result = pd.DataFrame()
    df_result['pred1'] = numpy.asarray(pred.t().tolist()[0])
    df_result['pred2'] = numpy.asarray(pred.t().tolist()[1])
    df_result['label'] = label
    
    df_result = df_result[mask.tolist()]
    df_result.to_csv(filename, index=False)
        
def test(exp, loader, date, experiment_id=0):
    print('Test', model_file)
    import torch.nn.functional as F
    @torch.no_grad()
    def pred_conf1(exp, loader, confidence=0.5): 
        model = exp.prepare_model(loader.data)
        model.load_state_dict(torch.load(model_file))

        # model = exp.get_model(model_file, loader.data)
        mask = loader.data.test_mask
        # test_indices = mask.nonzero().t().contiguous().tolist()[0]

        model.eval()
        with torch.no_grad():
            print(model.device, oader.data.x.device, loader.data.edge_index.device)
            pred_raw = model(loader.data.x, loader.data.edge_index)
            pred_raw = F.softmax(pred_raw, dim=1)
            # print(pred_raw)
            
            pred = torch.tensor((pred_raw >= confidence).t()[1], dtype=int)
        
        pred, pred_raw = pred.cpu(), pred_raw.cpu()
        loader.data.y = loader.data.y.cpu()
        mask = mask.cpu()
        
        if roc_file_format is not None:
            roc_file=roc_file_format.format(date, i)
            save_roc(roc_file, pred_raw, loader.data.y, mask)
            roc_df = pd.read_csv(roc_file)
            roc(roc_df[['pred1', 'pred2']].to_numpy(), roc_df['label'].to_numpy())
        else:
            roc(pred_raw[mask], loader.data.y[mask])

        prec_recall(pred_raw[mask], loader.data.y[mask])
        accs = score(pred[mask], loader.data.y[mask])
        
        unknown_mal = sum(pred[loader.unknown_mask])
        accs['unknown_mal'] = unknown_mal.item()
        accs['unknown_ben'] = (sum(loader.unknown_mask) - unknown_mal).item()
        # unknown_mal = sum(pred) - sum(pred[mask])

        return accs

    accs = pred_conf1(exp, loader)#exp.test(model_file, data, show_curve=False)
    print(accs)
    # accs['date'] = date
    # accs['id'] = i
    
    # result.append(accs)
    return accs


result = []
dates = [7]
rate=0.01
curr = 7
# SET EPSILON

# k-fold tests 
i = int(args1.epsilon*100)
args1.experiment_id = i

# args1.epsilon = clp
nodes_file = nodes_file_format.format(curr, curr)
edges_file = edges_file_format.format(curr, curr)
model_file = model_file_format.format(curr, i)
args1.model_file = model_file

print('Arguments:', args1)
loader = Loader(nodes_file, edges_file, feature_labels, args1)
exp = SAGE_Experiment(loader.data, args1)

model = exp.prepare_model(loader.data)
model.load_state_dict(torch.load(model_file))

cleanAccs = exp.cleantest(model, loader.data, loader.test_mask)
cleanAccs['clp'] = i
cleanAccs['type'] = 'clean'
result.append(cleanAccs)

advAccs = exp.mimiciptest(model, loader.data, loader.test_mask, clp=args1.epsilon)
advAccs['clp'] = i
advAccs['type'] = 'adv'
result.append(advAccs)

print('Adversarial', advAccs)
        
result_df = pd.DataFrame(result)
result_df.to_csv(f'/export/sec02/fatih/paper_stats/data/attack/mimicip_test_{i}.csv', index=False)
