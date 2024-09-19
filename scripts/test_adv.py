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
    
from src import experiment_adv
from src import loader

imp.reload(experiment_adv)
SAGE_Experiment=experiment_adv.SAGE_Experiment
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

surrogate_model_file = "/export/sec02/fatih/paper_stats/data/attack/clean_2022-07-7_0.pkl" # month and experiment ID
model_file_format = "/export/sec02/fatih/paper_stats/data/attack/adv_both_2022-07-{}_{}.pkl" # month and experiment ID

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

surrogate_exp = SAGE_Experiment(loader.data, args1)
surrogate_model = surrogate_exp.prepare_model(loader.data)
surrogate_model.load_state_dict(torch.load(surrogate_model_file))

cleanAccs = exp.cleantest(model, loader.data, loader.test_mask)
cleanAccs['clp'] = i
cleanAccs['type'] = 'clean'
result.append(cleanAccs)
print('Clean', cleanAccs)

advAccs = exp.mimiciptest(model, loader.data, loader.test_mask, mintA=False, clp=args1.epsilon)
advAccs['clp'] = i
advAccs['type'] = 'mimicIP'
result.append(advAccs)
print('mimicIP', advAccs)

advAccs = exp.mimiciptest(model, loader.data, loader.test_mask, surrogate_model=surrogate_model, mintA=True, clp=args1.epsilon)
advAccs['clp'] = i
advAccs['type'] = 'mintA'
result.append(advAccs)
print('mintA', advAccs)
        
result_df = pd.DataFrame(result)
result_df.to_csv(f'/export/sec02/fatih/paper_stats/data/attack/adv_bothattack_{i}.csv', index=False)
