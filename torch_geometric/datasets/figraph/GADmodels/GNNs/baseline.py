import csv

import pandas as pd
import torch
from util import (
    GAT_1,
    GCN_1,
    GraphSAGE_1,
    evaluate,
    load_data,
    map_nodes_to_indices,
    read_edges_from_files,
    sigmoid_focal_loss,
)

from torch_geometric.data import Data

# 加载数据
x_all = []
y_all = []
data_all = []
for i in range(2014, 2023, 1):
    st = i
    end = i
    data_feature = load_data(list(range(st, end + 1, 1)))
    # 用法示例
    edge_file_paths = []

    for ii in range(st, end + 1):
        tmp = 'data/newEdge-Nips/graph_edges_listed' + str(ii) + '_new.txt'
        edge_file_paths.append(tmp)

    # 读取所有边
    all_edges = read_edges_from_files(edge_file_paths)

    mapped_edges, mapped_node_features_labels, train_mask, val_mask, test_mask = \
        map_nodes_to_indices(all_edges, data_feature, list(range(st, end + 1, 1)))

    # 构建 Data 对象
    edge_index = torch.tensor([(edge[0], edge[1]) for edge in mapped_edges],
                              dtype=torch.long).t().contiguous()
    x = torch.tensor([node[1] for node in mapped_node_features_labels],
                     dtype=torch.float)
    y = torch.tensor([node[2] for node in mapped_node_features_labels],
                     dtype=torch.long)
    x_all.append(x)
    y_all.append(y)
    data = Data(x=x, edge_index=edge_index, y=y)
    data_all.append(data)

# criterion = FocalLoss()

# split_dicts = {0: {'train': [2014, 2016], 'valid': 2017, 'test': 2018},
#                1: {'train': [2014, 2017], 'valid': 2018, 'test': 2019},
#                2: {'train': [2014, 2018], 'valid': 2019, 'test': 2020},
#                3: {'train': [2014, 2019], 'valid': 2020, 'test': 2021},
#                4: {'train': [2014, 2020], 'valid': 2021, 'test': 2022}}
split_dicts = {
    0: {
        'train': [2014, 2016],
        'valid': 2017,
        'test': 2018
    },
    1: {
        'train': [2014, 2017],
        'valid': 2018,
        'test': 2019
    },
    2: {
        'train': [2014, 2018],
        'valid': 2019,
        'test': 2020
    },
    3: {
        'train': [2014, 2019],
        'valid': 2020,
        'test': 2021
    },
    4: {
        'train': [2014, 2020],
        'valid': 2021,
        'test': 2022
    }
}

model_name = "GAT"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for model_name in ["SAGE", 'GAT', "GCN"]:
    for time in range(5):
        if model_name == "SAGE":
            model = GraphSAGE_1(247, 2).to(device)
        elif model_name == "GCN":
            model = GCN_1(247, 2).to(device)
        elif model_name == "GAT":
            model = GAT_1(247, 2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                     weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        result = []
        loss_train = []
        for epoch in range(500):
            loss_sum = 0
            for year in range(split_dicts[time]['train'][0],
                              split_dicts[time]['train'][1] + 1):
                labels = data_all[year - 2014].y.to(device)
                # 现在，data 包含了节点特征、连接关系和标签信息
                model.train()
                optimizer.zero_grad()
                out = model(data_all[year - 2014].to(device))
                loss = criterion(out, labels)
                # loss = sigmoid_focal_loss(gnn_loss_xent, out, labels, alpha=0.82, gamma=0)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                # print('Epoch:',epoch,'      Train')
                # evaluate(loss.item(), labels, out, epoch, params=1)
            loss_train.append(loss_sum / (split_dicts[time]['train'][1] -
                                          split_dicts[time]['train'][0] + 1))
            print(loss_sum / (split_dicts[time]['train'][1] -
                              split_dicts[time]['train'][0] + 1))

            # 验证
            model.eval()
            labels = data_all[split_dicts[time]['valid'] - 2014].y.to(device)
            with torch.no_grad():
                out = model(data_all[split_dicts[time]['valid'] - 2014])
            gnn_loss_xent = criterion(out, labels)
            loss = sigmoid_focal_loss(gnn_loss_xent, out, labels, alpha=0.82,
                                      gamma=0)
            print('Epoch:', epoch, '      Valid       ', loss.item())

            if epoch == 0:
                result = evaluate(loss.item(), labels, out, epoch, params=1)
            else:
                result = pd.concat([
                    result,
                    evaluate(loss.item(), labels, out, epoch, params=1)
                ], axis=0)

            # 测试
            model.eval()
            labels = data_all[split_dicts[time]['test'] - 2014].y.to(device)
            with torch.no_grad():
                out = model(data_all[split_dicts[time]['test'] - 2014])
            gnn_loss_xent = criterion(out, labels)
            loss = sigmoid_focal_loss(gnn_loss_xent, out, labels, alpha=0.82,
                                      gamma=0)
            print('Epoch:', epoch, '      Test       ', loss.item())
            result = pd.concat(
                [result,
                 evaluate(loss.item(), labels, out, epoch, params=2)], axis=0)
        file_name = 'result-Nips/result_Findata_Nips_' + str(
            time) + '_' + model_name + '.csv'
        result.to_csv(file_name)

        # 将列表写入CSV文件
        file_name = 'result-Nips/result_Findata_Nips_train_loss_' + str(
            time) + '_' + model_name + '.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train_loss_average'])
            for i, value in enumerate(loss_train):
                writer.writerow([i, value])
