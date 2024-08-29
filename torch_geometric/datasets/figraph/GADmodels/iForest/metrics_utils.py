import copy
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score,average_precision_score
from sklearn.utils import shuffle
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np








def KS(y_true, y_proba):
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic


def GM(y_true, y_pred):
    gmean = 1.0
    labels = sorted(list(set(y_true)))
    for label in labels:
        recall = (y_pred[y_true == label] == label).mean()
        gmean = gmean * recall
    return gmean ** (1 / len(labels))


def evaluate(labels, y_preds, y_probs, epo, loss, params={}):
    '''y_preds 是一维向量，y_probs是二维矩阵'''
    accuracy_list = []
    recall_list = []
    precision_list = []
    fpr_list = []
    f1_list = []
    roc_auc_list = []
    auprc_list = []
    ks_list = []

    balanced_accuracy_list = []
    recall_macro_list = []
    precision_macro_list = []
    f1_macro_arithmetic_list = []
    f1_macro_harmonic_list = []
    mauc_list = []
    gm_list = []

    params_list = []

    accuracy_list.append(accuracy_score(labels, y_preds))
    if len(np.unique(labels))<=2:
        recall_list.append(recall_score(labels, y_preds, average='binary', pos_label=1))
        precision_list.append(precision_score(labels, y_preds, average='binary', pos_label=1))
        fpr_list.append((y_preds[labels == 0] == 1).mean())
        f1_list.append(f1_score(labels, y_preds, average='binary', pos_label=1))
        roc_auc_list.append(roc_auc_score(labels, y_probs))
        auprc_list.append(average_precision_score(labels, y_probs, pos_label=1))
        ks_list.append(KS(labels, y_probs))

    elif len(np.unique(labels))>2:
        recall_list.append(None)
        precision_list.append(None)
        fpr_list.append(None)
        f1_list.append(None)
        roc_auc_list.append(None)
        auprc_list.append(None)
        ks_list.append(None)

    balanced_accuracy_list.append(balanced_accuracy_score(labels, y_preds))

    recall_macro = recall_score(labels, y_preds, average='macro')
    recall_macro_list.append(recall_macro)

    precision_macro = precision_score(labels, y_preds, average='macro')
    precision_macro_list.append(precision_macro)

    f1_macro_arithmetic_list.append(f1_score(labels, y_preds, average='macro'))
    f1_macro_harmonic = 2 * recall_macro * precision_macro / (recall_macro + precision_macro)
    f1_macro_harmonic_list.append(f1_macro_harmonic)

    mauc_list.append(roc_auc_score(labels, y_probs, average='macro', multi_class='ovo'))
    gm_list.append(GM(labels, y_preds))

    epoch_list = [epo]
    loss_list = [loss]
    params_list.append(params)

    indicator = np.vstack(
        [np.array(accuracy_list), np.array(recall_list),
         np.array(precision_list), np.array(fpr_list),
         np.array(f1_list), np.array(roc_auc_list),
         np.array(auprc_list),
         np.array(ks_list),

         np.array(balanced_accuracy_list), np.array(recall_macro_list),
         np.array(precision_macro_list), np.array(f1_macro_arithmetic_list),
         np.array(f1_macro_harmonic_list), np.array(mauc_list),
         np.array(gm_list),

         np.array(epoch_list), np.array(loss_list), np.array(params_list)
         ])

    scores = pd.DataFrame(indicator.T,
                          columns=['Accuracy', 'Recall', 'Precision',
                                   'FPR', 'F1', 'ROC_AUC', 'AUPRC', 'KS',

                                   'Balanced_Accuracy', 'Recall_macro',
                                   'precision_macro', 'F1_macro_arithmetic',
                                   'F1_macro_harmonic', 'MAUC', 'GM',

                                   'epoch','Loss', 'Parmmeters'])

    return scores




