import numpy as np
import pandas as pd
from scipy.sparse import load_npz, coo_matrix
from sklearn.preprocessing import StandardScaler
import dgl

def create_dgl_graph(year):
    # 根据L-U-L、L-A-L、L-R-L、L-P-L、L-L创建同构图
    g = []
    adjs = {}
    etypes = [  # src->tgt
        ('L', 'L-U', 'U'),
        ('U', 'U-L', 'L'),
        ('L', 'L-P', 'P'),
        ('P', 'P-L', 'L'),
        ('L', 'L-R', 'R'),
        ('R', 'R-L', 'L'),
        ('L', 'L-L', 'L'),
        ('L', 'L-A', 'A'),
        ('A', 'A-L', 'L'),
    ]
    # 计算L-U-L
    # 邻接矩阵
    # 边
    new_edges = {}
    for etype in etypes:
        stype, rtype, dtype = etype

        adj = load_npz('adjs/' + rtype + '_' + str(year) + '.npz')
        # 增加L-L的自环
        # if rtype == 'L-L':
        #     adj = set_diag(adj)

        # 邻接矩阵添加
        adjs[rtype.replace('-', '')] = adj

        src = adj.row
        dst = adj.col
        new_edges[(stype, rtype, dtype)] = (src, dst)

    g_hetero = dgl.heterograph(new_edges)
    # 添加同构图
    g.append(dgl.metapath_reachable_graph(g_hetero, ['L-U', 'U-L']))
    g.append(dgl.metapath_reachable_graph(g_hetero, ['L-L', 'L-L']))
    g.append(dgl.metapath_reachable_graph(g_hetero, ['L-P', 'P-L']))
    g.append(dgl.metapath_reachable_graph(g_hetero, ['L-R', 'R-L']))
    g.append(dgl.metapath_reachable_graph(g_hetero, ['L-A', 'A-L']))

    for i in range(len(g)):
        g[i] = dgl.add_self_loop(g[i])

    return g

# 给定特征表和年份，返回特征和标签
def feature_label(feature, year):
    feature_year = feature[(feature['Year'] == year)]
    # 按照上市公司的名称进行升序排序
    feature_year = feature_year.sort_values(by='Stkcd')
    # 标签
    labels = feature_year['label']
    # 特征
    node_features = feature_year.drop(columns={'StkcdYear', 'Stkcd', 'Year', 'label'})
    return node_features.values, labels.values

def new_split_data(train_start_year, train_end_year, valid_year, test_year):
    # 文件路径
    file_path = 'feature/feature.csv'
    # 节点特征
    feature = pd.read_csv(file_path)
    # 特征归一化
    # 特征提取
    num_cols = [col for col in list(feature.columns) if col not in ['StkcdYear', 'Stkcd', 'Year', 'label']]

    # 获取训练集，验证集，测试集
    trainset = pd.DataFrame.copy(
        feature[(feature['Year'] >= train_start_year) & (feature['Year'] <= train_end_year)], deep=True)
    validset = pd.DataFrame.copy(feature[(feature['Year'] == valid_year)], deep=True)
    testset = pd.DataFrame.copy(feature[(feature['Year'] == test_year)], deep=True)
    # 开始连续特征归一化
    trainset[num_cols] = StandardScaler().fit_transform(trainset[num_cols])
    validset[num_cols] = StandardScaler().fit_transform(validset[num_cols])
    testset[num_cols] = StandardScaler().fit_transform(testset[num_cols])
    trainset.index = range(trainset.shape[0])
    validset.index = range(validset.shape[0])
    testset.index = range(testset.shape[0])

    # 训练集年份
    train_g = {}
    train_feature = {}
    train_labels = {}
    for year in range(train_start_year, train_end_year + 1):
        train_g[year] = create_dgl_graph(year)
        train_feature[year], train_labels[year] = feature_label(trainset, year)

    # 验证集年份
    valid_g = create_dgl_graph(valid_year)
    valid_feature, valid_labels = feature_label(validset, valid_year)

    # 测试集年份
    test_g = create_dgl_graph(test_year)
    test_feature, test_labels = feature_label(testset, test_year)

    return train_g, train_feature, train_labels, valid_g, valid_feature, valid_labels, test_g, test_feature, test_labels

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