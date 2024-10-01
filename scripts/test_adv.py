import os
import sys
import torch

import pandas as pd
import importlib as imp

import torch_geometric

# Print versions for debugging purposes
print('Torch', torch.__version__)
print("PyTorch Geometric Version:", torch_geometric.__version__)
print("CUDA Version:", torch.version.cuda)

# Add the parent directory to the system path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
    
from src import experiment_adv
from src import loader
SAGE_Experiment = experiment_adv.SAGE_Experiment
Loader = loader.Loader

from src.features import feature_labels
from src import config_attack as config
args1 = config.parse()

# Select the GPU with the lowest memory usage
from utils import gpu_util
best_gpu = gpu_util.pick_gpu_lowest_memory()
args1.gpu_id = best_gpu

nodes_file = "../data/fqdn_apex_nodes.csv"
edges_file = "../data/fqdn_apex_edges.csv"

surrogate_model_file = "../models/clean_surrogate.pkl" 
model_file = "..models/test_adv.pkl" 
args1.model_file = model_file

result = []
dates = [7]
rate=0.01
curr = 7
# SET EPSILON

# k-fold tests 
i = int(args1.epsilon*100)
args1.experiment_id = i
# args1.epsilon = clp

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
result_df.to_csv(f'../data/adv_bothattack_{i}.csv', index=False)
