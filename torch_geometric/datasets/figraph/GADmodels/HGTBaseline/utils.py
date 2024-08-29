import gc

import dgl
import networkx as nx
import pandas as pd
import numpy as np
import torch
from scipy.sparse import load_npz, coo_matrix
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch_sparse import SparseTensor, set_diag

def load_feature(feature):
    # 按照上市公司的名称进行升序排序
    feature = feature.sort_values(by='Stkcd')
    # 标签
    labels = feature['label']
    # 特征
    node_features = feature.drop(columns={'StkcdYear', 'Stkcd', 'Year', 'label'})

    return torch.from_numpy(node_features.values.astype(np.float32)), labels.values

def feature_norm(feature_path, year):
    # 特征和标签数据
    feature = pd.read_csv(feature_path)
    feature_year = pd.DataFrame.copy(feature[(feature['Year'] == year)])

    # 特征归一化
    # 离散特征
    dis_cols = ['Audittyp', 'excessDebt_PropertyRightsNature', 'ProfitForecastTypeID']
    # 连续特征提取
    num_cols = [col for col in list(feature_year.columns) if
                col not in ['StkcdYear', 'Stkcd', 'Year', 'label'] + dis_cols]
    feature_year[num_cols] = StandardScaler().fit_transform(feature_year[num_cols])

    # 离散特征onehot
    feature_year = pd.get_dummies(feature_year, columns=dis_cols, dtype=int)
    # 特征列名整理
    allcols = list(feature_year.columns)
    feature_year = feature_year[allcols]

    return feature_year

def load_dataset(year, graph_type, graph_type_etypes, feature_path, device):
    # 特征和标签数据
    feature = pd.read_csv("feature/feature.csv")
    # 特征归一化
    num_cols = [col for col in list(feature.columns) if col not in ['StkcdYear', 'Stkcd', 'Year', 'label']]
    # 开始连续特征归一化
    feature[num_cols] = StandardScaler().fit_transform(feature[num_cols])

    # 边
    new_edges = {}
    # 节点
    ntypes = set()
    # 边类型
    etypes = set()

    for graph in graph_type:
        for etype in graph_type_etypes[graph]:
            stype, rtype, dtype = etype


            adj = load_npz('adjs/' + rtype + '_' + str(year) + '_' + graph + '.npz')
            # 增加L-L的自环
            if rtype == 'L-L':
                new_rows = np.concatenate([adj.row, np.arange(adj.shape[0])])
                new_cols = np.concatenate([adj.col, np.arange(adj.shape[0])])
                new_data = np.concatenate([adj.data, np.ones(adj.shape[0])])
                adj = coo_matrix((new_data, (new_rows, new_cols)), shape=adj.shape)

            src = adj.row
            dst = adj.col
            new_edges[(stype, rtype + '-' + graph, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
            etypes.add(rtype + '-' + graph)

    g = dgl.heterograph(new_edges)

    feature = feature_norm(feature_path, year)

    # 设置节点属性
    g.nodes['L'].data['inp'], label = load_feature(feature)
    dim = g.nodes['L'].data['inp'].shape[1]
    g.nodes['U'].data['inp'] = torch.eye(g.number_of_nodes('U'), dim)
    g.nodes['P'].data['inp'] = torch.eye(g.number_of_nodes('P'), dim)
    g.nodes['R'].data['inp'] = torch.eye(g.number_of_nodes('R'), dim)
    g.nodes['A'].data['inp'] = torch.eye(g.number_of_nodes('A'), dim)

    g.node_dict = {}
    g.edge_dict = {}
    for ntype in ntypes:
        g.node_dict[ntype] = len(g.node_dict)
    for etype in etypes:
        g.edge_dict[etype] = len(g.edge_dict)
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * g.edge_dict[etype]

    g = g.to(device)
    label = torch.tensor(label).long().to(device)
    return g, ntypes, etypes, label

def split_dataset(total, train, val, test):
    ids = [i for i in range(total.shape[0])]
    # 划分训练集和测试集id
    split_ratio = [train, val, test]
    # 计算总数
    total_ratios = sum(split_ratio)
    splits = [sum(split_ratio[:i + 1]) * len(ids) // total_ratios for i in range(len(split_ratio))]

    # 打乱
    np.random.shuffle(ids)
    # 验证集为测试集的前split个
    train_id = ids[:splits[0]]
    val_id = ids[splits[0]:splits[1]]
    test_id = ids[splits[1]:]
    # 节点序号排序
    train_id = np.sort(train_id)
    val_id = np.sort(val_id)
    test_id = np.sort(test_id)

    return train_id, val_id, test_id