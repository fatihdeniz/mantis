import argparse

def parse():
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument('--model_type', 
        default= 'sage'
    )
    
    parser.add_argument('--epoch', 
        type= int, 
        default= 801,
        help= 'epoch value',
    )
    
    parser.add_argument('--num_layers', 
        type= int, 
        default= 3,
        help= "number of layers"
    )
    
    parser.add_argument('--num_features', 
        type= int, 
        default= 62,
        help= "number of features"
    )
    
    parser.add_argument('--dim', 
        type= int, 
        default= 256,
        help= 'hidden dimension'
    )

    parser.add_argument("--fanout", 
        default= [-1,25,10],
        help= "samples per layer",
    )
    
    parser.add_argument('--outer_batch_size', 
        type= int, 
        default= 500,
        help= 'outer batch size'
    )
    
    parser.add_argument('--inner_batch_size', 
        type= int, 
        default= 40,
        help= 'inner batch size'
    )
    
    parser.add_argument('--train_percentage', 
        type=float, 
        default=0.8,
        help= 'train percentage'
    )
    
    parser.add_argument('--seed', 
        type=int, 
        default=42,
        help= 'seed'
    )

    parser.add_argument('--identifier', 
        type=str, 
        default='None',
        help= 'identifier'
    )
    
    parser.add_argument('--experiment_id', 
        type=int, 
        default=0,
        help= 'experiment id'
    )
    
    parser.add_argument('--gpu_id', 
        type=int, 
        default=5,
        help= 'gpu id'
    )

    parser.add_argument('--edge_type', 
        default=[0,1,2],
        help= 'edge types'
    )
    
    
    parser.add_argument('--extra', 
        default='s1',
        help= 'extra'
    )
    
    parser.add_argument('--lstm', 
        type=bool,
        default=False,
        help= 'concat all embeddings'
    )
    
    parser.add_argument('--weighted', 
        type=bool,
        default=False,
        help= 'Use edge weights during sampling!'
    )
    
    parser.add_argument('--hetero', 
        default=False,
        help= 'heterogeneous graph'
    )
    
    parser.add_argument('--use_syn', 
        default=False,
        help= 'create synthetic features'
    )
    
    parser.add_argument('--syn_file', 
        help= 'load synthetic features'
    )
    
    parser.add_argument('--syn_labels', 
        help= 'load synthetic'
    )
    
    parser.add_argument('--balance_labels', 
        default=False,
        help= 'balance labels'
    )
    
    parser.add_argument('--labelfeature_names', 
        default=['feat_label_ben','feat_label_mal','feat_label_unknown'],
        help= 'labelfeature names'
    )
    
    parser.add_argument('--label_source', 
        default=['popular', 'alexa', 'tranco', 'umbrella', 'edugov', 'benign4',
                'vt_seed5_3months', 'vt_active5_3months','vt_active5_12months', 'vt_seed5_12months', 'vt_mixed'],
        help= 'label sources'
    )
    
    parser.add_argument('--popularity_lists', 
        default=['alexa', 'tranco', 'umbrella', 'crux'],
        help= 'popularity lists'
    )
    
    parser.add_argument('--lr', 
        type=float, 
        default=0.01,
        help= 'learning rate'
    )
    
    parser.add_argument('--weight_decay', 
        type=float, 
        default=5e-4,
        help= 'weight decay'
    )
    
    parser.add_argument('--model_file', 
        type=str, 
        default=None,
        help= 'weight decay'
    )
    
    parser.add_argument('--balance_label_source', 
        type=str, 
        default='benign4',
        help= 'which label to remove for balancing'
    )
    
    parser.add_argument('--nodes_file', 
        type=str, 
        default=None,
        help= 'nodes file'
    )
    
    parser.add_argument('--edges_file', 
        type=str, 
        default=None,
        help= 'edges file'
    )
    
    parser.add_argument('--epsilon', 
        type=float, 
        default=0.8,
        help= 'for PGD attack'
    )

    return parser.parse_args('')