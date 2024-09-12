import os
import sys
import torch
from datetime import date
import numpy
import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from src.sage_experiment_20220326 import SAGE_Experiment
from src.features import feature_labels
from src.loader import Loader
from src.experiment_utils import pred_conf2 as pred_conf
from src import config_jupyter as config

sys.path.insert(0, '/export/sec02/fatih/Common')
from lib_jupyter import notebook_util
best_gpu = notebook_util.pick_gpu_lowest_memory()

def train(nodes_file, edges_file, model_file):
    args = config.parse()
    args.epoch = 2001
    args.dim = 256
    args.gpu_id = best_gpu
    
    args.label_source = ['alexa', 'tranco', 'crux', 'umbrella', 'edugov', 'benign4',
                          'vt_seed5_3months', 'vt_active5_3months','vt_active5_12months','vt_seed5_12months']
    loader = Loader(nodes_file, edges_file, feature_labels, args)
    exp = SAGE_Experiment(loader.data, args)
    model = exp.train()
    torch.save(model.state_dict(), model_file)
    return exp, loader

def test(exp, loader, model_file, **args):
    date = '2022-01-01' if 'date' not in args else args['date']
    args['output'] = date + '.csv'
    args['save'] = True
    accs = pred_conf(exp, loader, model_file, **args)
    accs['date'] = date
    accs['#mal'] = loader.data.y[loader.data.test_mask].sum().tolist()
    accs['#ben'] = loader.data.y[loader.data.test_mask].size(0) - accs['#mal']
    pred_ninety = pred_conf(exp, loader, model_file, date=date, output=args['output'], confidence=0.9)
    accs['0.9 fpr'] = pred_ninety['fpr']
    accs['0.9 recall'] = pred_ninety['recall']
    accs['0.9 prec'] = pred_ninety['prec']
    accs['0.9 unknown_mal'] = pred_ninety['unknown_mal']
    accs['0.9 unknown_apexmal'] = pred_ninety['unknown_apexmal']
    
    return accs

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("usage: python {} start_date end_date".format(sys.argv[0]))
        sys.exit()
    date_range = '{}-{}'.format(sys.argv[1], sys.argv[2])

    print('Model training dates:{} '.format(date_range))
    try:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'blacklist_forecast', 'graph_data'))
        nodes_file = os.path.join(path, '{}_fqdn_apex_nodes.csv.log.norm'.format(date_range))
        edges_file = os.path.join(path, '{}_fqdn_apex_edges.csv'.format(date_range))

        model_file = '/export/sec02/fatih/output/{}.pkl'.format(sys.argv[2])
        raw_directory='/export/sec02/fatih/output/'

        if os.path.exists(nodes_file) and not os.path.exists(model_file):
            exp, loader = train(nodes_file, edges_file, model_file)
            accs = test(exp, loader, model_file, date=sys.argv[2], raw_directory=raw_directory)
            
        elif os.path.exists(nodes_file) and os.path.exists(model_file):
            if len(sys.argv) > 3 and sys.argv[3] == 'test':
                args = config.parse()
                args.gpu_id = -1
                loader = Loader(nodes_file, edges_file, feature_labels, args)
                exp = SAGE_Experiment(loader.data, args)
                accs = test(exp, loader, model_file, date=sys.argv[2], raw_directory=raw_directory)
            else:
                print('Error. You can only test. Model already trained!!')
        else:
            print('Model training failure:', os.path.exists(nodes_file), os.path.exists(model_file))
            
        if accs is not None:
            print(accs)
            accs_df = pd.DataFrame([accs])
            
            stats = "/export/sec02/fatih/performance.csv"
            if not os.path.exists(stats):
                accs_df.to_csv(stats, index=False)
            else:
                p_df = pd.read_csv(stats)
                p_df = pd.concat([p_df, accs_df], axis=0).to_csv(stats, index=False)
    except Exception as e:
        print('Exception occurred', repr(e))