from cmath import nan
import random
import torch
import itertools 
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def score(pred, labels):
    # tp, fp, tn, fn = confusion(labels, pred)
    # print('tn, fp, fn, tp', tn, fp, fn, tp)
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    accuracy = (pred == labels).sum() / len(pred)

    try:
        micro_f1 = f1_score(labels, pred, average='micro')
        macro_f1 = f1_score(labels, pred, average='macro')
        f1 = f1_score(labels, pred)
    except:
        print('Exception occurred while calculating F1 Score', labels, pred)
        f1 = nan
    try:
        auc = roc_auc_score(labels, pred)
    except:
        auc = nan

    try:
        prec, recall = precision_score(labels, pred), recall_score(labels,pred)
    except:
        prec,recall = nan,nan

    try:
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        # print('tn, fp, fn, tp', tn, fp, fn, tp)
        fpr = fp/(fp+tn)
    except:
        fpr = nan # print('SCORE', accuracy, f1, auc)

    return {'acc':accuracy, 'f1':f1, 'auc':auc, 'prec':prec, 'recall':recall, 'fpr':fpr}

def confusion(truth, prediction):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def gen_xavier(xavier_file = './data/city/city_xavier.csv',
                         node_count=263148, feature_count=32):
    x = torch.empty(node_count, feature_count)
    torch.nn.init.xavier_normal_(x)

    x = x.t()
    df_feat = pd.DataFrame()

    for i in range (0,feature_count):
        df_feat[f'x{i}'] = x[i].tolist()

    df_feat.to_csv(xavier_file, index=False)


def __random_bool(percent):
    return random.randrange(100) > percent

def randomize_bool(nodes_df, column_name ,groupby='label'):
    groups = nodes_df.groupby(groupby).size().reset_index(name='counts')[groupby].tolist()

    results = []
    for group in groups:
        stats = nodes_df[nodes_df[groupby] == group].groupby(column_name).size().reset_index(name='count')
        results.append(100 * stats['count'][0]/(stats['count'][0] + stats['count'][1]))

    print(results)

    randomlist = nodes_df.apply(lambda x: __random_bool(results[x[groupby]]), axis=1).tolist()

    randomlist = list(map(int, randomlist))
    nodes_df[column_name] = randomlist
    return nodes_df


def subsets(s):
    for cardinality in range(len(s) + 1):
        yield from itertools.combinations(s, cardinality)