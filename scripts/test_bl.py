import os
import sys
import torch
from datetime import date
import numpy
import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from src.sage_experiment import SAGE_Experiment
from src.features import feature_labels
from src.loader import Loader
from src.experiment_utils import pred_conf2 as pred_conf
from src import config
args1 = config.parse()
args1.gpu_id = -1

def test(nodes_file, edges_file, model_file, **args):
    raw_directory='/export/sec02/fatih/output/stack' if 'raw_directory' not in args else args['raw_directory']

    loader = Loader(nodes_file, edges_file, feature_labels, args1)
    exp = SAGE_Experiment(loader.data, args1)
    
    args['output'] = nodes_file + '.pred'
    args['save'] = True
    accs = pred_conf(exp, loader, model_file, **args)
    
    return accs
# test('/export/sec02/fatih/singlequery/twitter.com/original_nodes.csv.log.norm', 
#      '/export/sec02/fatih/singlequery/twitter.com/original_edges.csv', 
#      '/export/sec02/fatih/output/2022-06-07.pkl'
#     )

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage: python {} nodes_file edges_file model_file".format(sys.argv[0]))
        sys.exit()
    try:
        nodes_file = sys.argv[1]
        edges_file = sys.argv[2]

        model_file = '/export/sec02/fatih/output/{}.pkl'.format(sys.argv[3])
        

        if os.path.exists(nodes_file) and os.path.exists(model_file):
            test(nodes_file, edges_file, model_file, raw_directory)
        else:
            print('Model training failure:', os.path.exists(nodes_file), os.path.exists(model_file))
    except Exception as e:
        print('Exception occurred', e)
