import copy as cp
import pickle
import random
from collections import defaultdict

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data, prefix='data/'):
    """Load graph, feature, and label given dataset name
    :returns: home and single-relation graphs, feature, label
    """
    if data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
    elif data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)

    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
    """Row-normalize sparse matrix
    Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename=''):
    """Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    # with open(filename, 'wb') as file:
    #     pickle.dump(adj_lists, file)
    # file.close()
    return adj_lists


def pos_neg_split(nodes, labels):
    """Find positive and negative nodes given a list of nodes and their labels
    :param nodes: a list of nodes
    :param labels: a list of node labels
    :returns: the spited positive and negative nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def pick_step(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
    """Test the performance of GraphSAGE
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    gnn_pred_list = []
    gnn_prob_list = []
    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob = model.to_prob(batch_nodes)

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1,
                               average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0,
                               average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        +
        f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn


def prob2pred(y_prob, thres=0.5):
    """Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


def test_pcgnn(test_cases, labels, model, batch_size, adj_lists, features,
               intra_list, train_pos, epoch, params, thres=0.5):
    """Test the performance of PC-GNN and its variants
    :param test_cases: a list of testing node
    :param labels: a list of testing node labels
    :param model: the GNN model
    :param batch_size: number nodes in a batch
    :returns: the AUC and Recall of GNN and Simi modules
    """
    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1_label1 = 0.0
    acc_label1 = 0.00
    recall_label1 = 0.0
    gnn_pred_list = []
    gnn_prob_list = []
    label_list1 = []
    loss_sum = 0

    for iteration in range(test_batch_num):
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label,
                                              adj_lists, intra_list, features,
                                              train_pos, train_flag=False)
        loss = model.loss(batch_nodes, batch_label, adj_lists, intra_list,
                          features, train_pos, train_flag=False)
        loss_sum += loss.item()

        gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
        gnn_pred = prob2pred(gnn_prob_arr, thres)

        f1_label1 += f1_score(batch_label,
                              label_prob1.data.cpu().numpy().argmax(axis=1),
                              average="macro")
        acc_label1 += accuracy_score(
            batch_label,
            label_prob1.data.cpu().numpy().argmax(axis=1))
        recall_label1 += recall_score(
            batch_label,
            label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

        gnn_pred_list.extend(gnn_pred.tolist())
        gnn_prob_list.extend(gnn_prob_arr.tolist())
        label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

    auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
    average_precision_score(labels, np.array(gnn_prob_list))
    auc_label1 = roc_auc_score(labels, np.array(label_list1))
    ap_label1 = average_precision_score(labels, np.array(label_list1))

    f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1,
                               average='binary')
    f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0,
                               average='binary')
    f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
    f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
    conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
    tn, fp, fn, tp = conf_gnn.ravel()
    gmean_gnn = conf_gmean(conf_gnn)

    print(
        f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"
        +
        f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}"
    )
    print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
    print(
        f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tAccuracy: {acc_label1 / test_batch_num:.4f}"
        +
        f"\tRecall: {recall_label1 / test_batch_num:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}"
    )
    a = evaluate(loss=loss_sum / test_batch_num, labels=labels,
                 y_probs=np.array(gnn_pred_list), epo=epoch, params=params)

    return a


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp)))**0.5


def read_edges_from_files(file_paths):
    all_edges = []

    for file_path in file_paths:
        with open(file_path) as edge_file:
            edges = edge_file.readlines()
            # 处理每个文件中的边信息
            for edge in edges:
                start_node, end_node, edge_attr = edge.strip().split()
                all_edges.append((start_node, end_node, edge_attr))

    return all_edges


def map_nodes_to_indices(all_edges, feature_label_df, year_list):
    # 读取特征和标签文件
    feature_label_df = feature_label_df

    # 建立节点标识到索引的映射
    node_index_map = {}
    current_index = 0

    # 处理特征和标签文件
    node_features_labels = []
    node_feature_columns = feature_label_df.columns.difference(
        ['StkcdYear', 'label'])

    for index, row in feature_label_df.iterrows():
        node = int(row['StkcdYear'])  # 从 CSV 中读取的节点 id 可能是浮点数，需要转换为整数
        features = row[node_feature_columns]  # 前面的特征列
        label = row.iloc[-1]  # 最后一列是标签

        if node not in node_index_map:
            node_index_map[node] = current_index
            current_index += 1
        node_features_labels.append((node_index_map[node], features, label))

    # 处理连接关系文件
    mapped_edges = []
    for edge in all_edges:
        start_node, end_node, edge_attr = edge
        start_node = int(start_node)
        end_node = int(end_node)
        # if start_node not in node_index_map:
        #     node_index_map[start_node] = current_index
        #     current_index += 1
        # if end_node not in node_index_map:
        #     node_index_map[end_node] = current_index
        #     current_index += 1
        if start_node in node_index_map and end_node in node_index_map:
            mapped_edges.append((node_index_map[start_node],
                                 node_index_map[end_node], edge_attr))

    # train_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) in year_list[:7]]
    # val_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) == year_list[-2]]
    # test_mask = [int(value) for key, value in node_index_map.items() if int(str(key)[-4:]) == year_list[-1]]

    # return mapped_edges, node_features_labels , train_mask, val_mask, test_mask
    return mapped_edges, node_features_labels


def load_all_data():
    features_all = []
    y_all = []
    dgl_all = []
    adj_all = []
    array_all = []
    idx_train_all = []
    idx_test_all = []
    idx_valid_all = []
    y_train_all = []
    y_valid_all = []
    y_test_all = []
    for i in range(2014, 2023, 1):
        st = i
        end = i
        data_feature = load_data_new(list(range(st, end + 1, 1)))
        # 用法示例
        edge_file_paths = []

        for ii in range(st, end + 1):
            # tmp = 'data/newEdge/graph_edges_listed' + str(ii) + '_new.txt'
            tmp = 'data/newEdge-Nips/graph_edges_listed' + str(ii) + '_new.txt'
            edge_file_paths.append(tmp)

        # 读取所有边
        all_edges = read_edges_from_files(edge_file_paths)

        mapped_edges, mapped_node_features_labels = \
            map_nodes_to_indices(all_edges, data_feature, list(range(st, end + 1, 1)))

        # 构建 Data 对象
        edge_index_010 = torch.tensor(
            [(edge[0], edge[1]) for edge in mapped_edges if edge[2] == '010'],
            dtype=torch.long).t().contiguous()
        edge_index_001 = torch.tensor(
            [(edge[0], edge[1]) for edge in mapped_edges if edge[2] == '001'],
            dtype=torch.long).t().contiguous()
        edge_index_100 = torch.tensor(
            [(edge[0], edge[1]) for edge in mapped_edges if edge[2] == '100'],
            dtype=torch.long).t().contiguous()

        x = torch.tensor([node[1] for node in mapped_node_features_labels],
                         dtype=torch.float)
        y = torch.tensor([node[2] for node in mapped_node_features_labels],
                         dtype=torch.long)

        index = list(range(len(y)))
        labels = y
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels, stratify=labels, train_size=0.4, random_state=None,
            shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest, test_size=0.67,
            random_state=None, shuffle=True)

        csc = csc_matrix((np.ones_like(edge_index_010[0]),
                          (edge_index_010[0], edge_index_010[1])),
                         shape=(len(y), len(y)))
        graph_010 = dgl.from_scipy(csc)
        graph_010.ndata['feat'] = x
        csc.toarray()
        relation_010 = sparse_to_adjlist(csc)
        # eigen_adj = alpha * inv((sp.eye(adj.shape[0]) - (1 - alpha) * adj_normalize(adj)).toarray())  # 计算扩散矩阵
        # for p in range(adj.shape[0]):
        #     eigen_adj[p, p] = 0.
        # eigen_adj = PPR_normalize(eigen_adj)
        # eigen_adjs.append(eigen_adj)

        csc = csc_matrix((np.ones_like(edge_index_001[0]),
                          (edge_index_001[0], edge_index_001[1])),
                         shape=(len(y), len(y)))
        graph_001 = dgl.from_scipy(csc)
        graph_001.ndata['feat'] = x
        csc.toarray()
        relation_001 = sparse_to_adjlist(csc)
        # eigen_adj = alpha * inv((sp.eye(adj.shape[0]) - (1 - alpha) * adj_normalize(adj)).toarray())  # 计算扩散矩阵
        # for p in range(adj.shape[0]):
        #     eigen_adj[p, p] = 0.
        # eigen_adj = PPR_normalize(eigen_adj)
        # eigen_adjs.append(eigen_adj)

        csc = csc_matrix((np.ones_like(edge_index_100[0]),
                          (edge_index_100[0], edge_index_100[1])),
                         shape=(len(y), len(y)))
        graph_100 = dgl.from_scipy(csc)
        graph_100.ndata['feat'] = x
        csc.toarray()
        relation_100 = sparse_to_adjlist(csc)
        # eigen_adj = alpha * inv((sp.eye(adj.shape[0]) - (1 - alpha) * adj_normalize(adj)).toarray())  # 计算扩散矩阵
        # for p in range(adj.shape[0]):
        #     eigen_adj[p, p] = 0.
        # eigen_adj = PPR_normalize(eigen_adj)
        # eigen_adjs.append(eigen_adj)

        # average_array = np.mean([eigen_adjs[0], eigen_adjs[1], eigen_adjs[2]], axis=0)
        # #
        # print(np.count_nonzero(average_array))

        if torch.cuda.is_available():
            graph_001 = graph_001.to('cuda:0')
            graph_010 = graph_010.to('cuda:0')
            graph_100 = graph_100.to('cuda:0')
            graph_100.ndata['feat'] = x.to('cuda:0')
            graph_010.ndata['feat'] = x.to('cuda:0')
            graph_001.ndata['feat'] = x.to('cuda:0')

        dgl_lists = [graph_001, graph_010, graph_100]
        adj_lists = [relation_001, relation_010, relation_100]

        features_all.append(x)
        y_all.append(y)
        dgl_all.append(dgl_lists)
        adj_all.append(adj_lists)
        # array_all.append(average_array)

        idx_train_all.append(idx_train)
        idx_valid_all.append(idx_valid)
        idx_test_all.append(idx_test)

        y_train_all.append(y_train)
        y_valid_all.append(y_valid)
        y_test_all.append(y_test)
    return features_all, y_all, dgl_all, adj_all, idx_train_all, idx_valid_all, idx_test_all, y_train_all, y_valid_all, y_test_all, array_all


def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def PPR_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_new(year_list):
    # data = pd.read_csv("data/feature_new.csv")
    data = pd.read_csv("data/ListedCompanyFeatures.csv")
    # print( data.columns.tolist())
    data = data[data['Year'].isin(year_list)]
    StkcdYear = data['StkcdYear']

    data.drop(columns=['Stkcd'], inplace=True)
    dis_cols = [
        'Audittyp', 'excessDebt_PropertyRightsNature', 'ProfitForecastTypeID',
        'Audittyp_1', 'Audittyp_2', 'Audittyp_3', 'Audittyp_4', 'Audittyp_5',
        'Audittyp_6', 'excessDebt_PropertyRightsNature_0',
        'excessDebt_PropertyRightsNature_1', 'ProfitForecastTypeID_2',
        'ProfitForecastTypeID_3', 'ProfitForecastTypeID_4',
        'ProfitForecastTypeID_5', 'ProfitForecastTypeID_7',
        'ProfitForecastTypeID_8', 'ProfitForecastTypeID_9',
        'ProfitForecastTypeID_12', 'ProfitForecastTypeID_13'
    ]
    num_cols = [
        col for col in list(data.columns)
        if col not in ['Year', 'label'] + dis_cols
    ]
    # data = pd.get_dummies(data, columns=dis_cols)

    allcols = list(data.columns)
    allcols.remove('label')
    allcols.extend(['label'])
    data = data[allcols]

    trainset = pd.DataFrame.copy(data, deep=True)
    trainset.drop(columns=['Year'], axis=1, inplace=True)
    trainset[num_cols] = StandardScaler().fit_transform(trainset[num_cols])
    trainset['StkcdYear'] = StkcdYear
    # print(trainset.columns.tolist())
    return trainset


def KS(y_true, y_proba):
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic


def GM(y_true, y_pred):
    gmean = 1.0
    labels = sorted(list(set(y_true)))
    for label in labels:
        recall = (y_pred[y_true == label]).mean()
        gmean = gmean * recall
    return gmean**(1 / len(labels))


def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp)))**0.5


# def evaluate(labels, y_probs, epo, loss, params):
#     accuracy_list = []
#     recall_list = []
#     precision_list = []
#     fpr_list = []
#     f1_list = []
#     roc_auc_list = []
#     ks_list = []
#     auprc_list = []
#     balanced_accuracy_list = []
#     recall_macro_list = []
#     precision_macro_list = []
#     f1_macro_arithmetic_list = []
#     f1_macro_harmonic_list = []
#     mauc_list = []
#     gm_list = []
#     GMean_list = []
#
#     params_list = []
#
#     y_preds = np.array([1 if i > 0.5 else 0 for i in y_probs.squeeze()])
#
#     accuracy_list.append(accuracy_score(labels, y_preds))
#     recall_list.append(recall_score(labels, y_preds, average='binary', pos_label=1))
#     precision_list.append(precision_score(labels, y_preds, average='binary', pos_label=1))
#     fpr_list.append((y_preds[labels == 0] == 1).mean())
#     f1_list.append(f1_score(labels, y_preds, average='binary', pos_label=1))
#     roc_auc_list.append(roc_auc_score(labels, y_probs))
#
#     auprc_list.append(average_precision_score(labels, y_probs, pos_label=1))
#
#     ks_list.append(KS(labels, y_probs))
#
#     balanced_accuracy_list.append(balanced_accuracy_score(labels, y_preds))
#
#     recall_macro = recall_score(labels, y_preds, average='macro')
#     recall_macro_list.append(recall_macro)
#
#     precision_macro = precision_score(labels, y_preds, average='macro')
#     precision_macro_list.append(precision_macro)
#
#     f1_macro_arithmetic_list.append(f1_score(labels, y_preds, average='macro'))
#     f1_macro_harmonic = 2 * recall_macro * precision_macro / (recall_macro + precision_macro)
#     f1_macro_harmonic_list.append(f1_macro_harmonic)
#
#     mauc_list.append(roc_auc_score(labels, y_probs, average='macro', multi_class='ovo'))
#     gm_list.append(GM(labels, y_preds))
#
#     conf_gnn = confusion_matrix(labels, np.array(y_preds))
#     gmean_gnn = conf_gmean(conf_gnn)
#     GMean_list = [gmean_gnn]
#
#     epoch_list = [epo]
#     loss_list = [loss]
#     params_list.append(params)
#
#     indicator = np.vstack(
#         [np.array(accuracy_list), np.array(recall_list),         np.array(precision_list), np.array(fpr_list),         np.array(f1_list),
#          np.array(roc_auc_list),         np.array(auprc_list),         np.array(ks_list),
#
#          np.array(balanced_accuracy_list), np.array(recall_macro_list),
#          np.array(precision_macro_list), np.array(f1_macro_arithmetic_list),
#          np.array(f1_macro_harmonic_list), np.array(mauc_list),
#          np.array(gm_list), np.array(GMean_list),
#
#          np.array(epoch_list), np.array(loss_list), np.array(params_list)
#          ])
#
#     scores = pd.DataFrame(indicator.T,
#                           columns=['Accuracy', 'Recall', 'Precision',                                   'FPR', 'F1',
#                                    'ROC_AUC', 'AUPRC', 'KS',
#
#                                    'Balanced_Accuracy', 'Recall_macro',
#                                    'precision_macro', 'F1_macro_arithmetic',
#                                    'F1_macro_harmonic', 'MAUC', 'GM', 'GMean',
#
#                                    'epoch', 'Loss', 'Parmmeters'])
#
#     return scores


def evaluate(labels, y_probs, epo, loss, params):
    accuracy_list = []
    recall_list = []
    precision_list = []
    fpr_list = []
    f1_list = []
    roc_auc_list = []
    ks_list = []

    balanced_accuracy_list = []
    recall_macro_list = []
    precision_macro_list = []
    f1_macro_arithmetic_list = []
    f1_macro_harmonic_list = []
    mauc_list = []
    gm_list = []
    GMean_list = []
    auprc_list = []

    params_list = []

    y_preds = np.array([1 if i > 0.5 else 0 for i in y_probs.squeeze()])

    accuracy_list.append(accuracy_score(labels, y_preds))
    recall_list.append(
        recall_score(labels, y_preds, average='binary', pos_label=1))
    precision_list.append(
        precision_score(labels, y_preds, average='binary', pos_label=1))
    fpr_list.append((y_preds[labels == 0] == 1).mean())
    f1_list.append(f1_score(labels, y_preds, average='binary', pos_label=1))
    roc_auc_list.append(roc_auc_score(labels, y_probs))
    auprc_list.append(average_precision_score(labels, y_probs, pos_label=1))
    ks_list.append(KS(labels, y_probs))

    balanced_accuracy_list.append(balanced_accuracy_score(labels, y_preds))

    recall_macro = recall_score(labels, y_preds, average='macro')
    recall_macro_list.append(recall_macro)

    precision_macro = precision_score(labels, y_preds, average='macro')
    precision_macro_list.append(precision_macro)

    f1_macro_arithmetic_list.append(f1_score(labels, y_preds, average='macro'))
    f1_macro_harmonic = 2 * recall_macro * precision_macro / (recall_macro +
                                                              precision_macro)
    f1_macro_harmonic_list.append(f1_macro_harmonic)

    mauc_list.append(
        roc_auc_score(labels, y_probs, average='macro', multi_class='ovo'))
    gm_list.append(GM(labels, y_preds))

    conf_gnn = confusion_matrix(labels, np.array(y_preds))
    gmean_gnn = conf_gmean(conf_gnn)
    GMean_list = [gmean_gnn]

    epoch_list = [epo]
    loss_list = [loss]
    params_list.append(params)

    indicator = np.vstack([
        np.array(accuracy_list),
        np.array(recall_list),
        np.array(precision_list),
        np.array(fpr_list),
        np.array(f1_list),
        np.array(roc_auc_list),
        np.array(auprc_list),
        np.array(ks_list),
        np.array(balanced_accuracy_list),
        np.array(recall_macro_list),
        np.array(precision_macro_list),
        np.array(f1_macro_arithmetic_list),
        np.array(f1_macro_harmonic_list),
        np.array(mauc_list),
        np.array(gm_list),
        np.array(GMean_list),
        np.array(epoch_list),
        np.array(loss_list),
        np.array(params_list)
    ])

    scores = pd.DataFrame(
        indicator.T, columns=[
            'Accuracy', 'Recall', 'Precision', 'FPR', 'F1', 'ROC_AUC', 'AUPRC',
            'KS', 'Balanced_Accuracy', 'Recall_macro', 'precision_macro',
            'F1_macro_arithmetic', 'F1_macro_harmonic', 'MAUC', 'GM', 'GMean',
            'epoch', 'Loss', 'Parmmeters'
        ])

    return scores
