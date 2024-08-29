import random

import numpy as np
import pandas as pd
import torch
from metrics_utils import evaluate  # 导入评估函数

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

# 确保使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载边数据
edge_files = {
    2014: '../data/FiGraph/edges2014.csv',
    2015: '../data/FiGraph/edges2015.csv',
    2016: '../data/FiGraph/edges2016.csv',
    2017: '../data/FiGraph/edges2017.csv',
    2018: '../data/FiGraph/edges2018.csv',
    2019: '../data/FiGraph/edges2019.csv',
    2020: '../data/FiGraph/edges2020.csv',
    2021: '../data/FiGraph/edges2021.csv',
    2022: '../data/FiGraph/edges2022.csv',
}

features = pd.read_csv('../data/FiGraph/ListedCompanyFeatures.csv')


def remove_background_nodes(edges, removal_percentage, node_labels):
    background_nodes = [
        i for i, label in enumerate(node_labels) if label == -1
    ]
    num_remove = int(len(background_nodes) * removal_percentage / 100)
    remove_nodes = random.sample(background_nodes, num_remove)
    keep_nodes = list(set(range(len(node_labels))) - set(remove_nodes))
    return keep_nodes


def create_data_list(removal_percentage):
    data_list = []
    all_edges = pd.concat([
        pd.read_csv(edge_file, names=['source', 'target', 'relation'])
        for edge_file in edge_files.values()
    ])
    all_node_ids = list(
        set(all_edges['source']).union(set(all_edges['target'])))
    all_node_index = {node: idx for idx, node in enumerate(all_node_ids)}

    all_node_labels = np.full(len(all_node_index), -1)  # 初始化为 -1，表示背景节点没有标签
    for _, row in features.iterrows():
        if row['nodeID'] in all_node_index:
            idx = all_node_index[row['nodeID']]
            all_node_labels[idx] = row['Label']

    keep_nodes = remove_background_nodes(all_edges, removal_percentage,
                                         all_node_labels)
    keep_nodes_set = set(keep_nodes)

    for year, edge_file in edge_files.items():
        edges = pd.read_csv(edge_file, names=['source', 'target', 'relation'])
        node_ids = list(set(edges['source']).union(set(edges['target'])))
        node_index = {node: idx for idx, node in enumerate(node_ids)}

        edge_index = {
            edge_type: [[], []]
            for edge_type in edges['relation'].unique()
        }
        for _, row in edges.iterrows():
            src = node_index[row['source']]
            tgt = node_index[row['target']]
            if src in keep_nodes_set and tgt in keep_nodes_set:
                edge_index[row['relation']][0].append(src)
                edge_index[row['relation']][1].append(tgt)

        num_nodes = len(node_index)
        node_features = np.zeros((num_nodes, features.shape[1] - 3))
        node_labels = np.full(num_nodes, -1)  # 初始化为 -1，表示背景节点没有标签

        for _, row in features[features['Year'] == year].iterrows():
            if row['nodeID'] in node_index:
                idx = node_index[row['nodeID']]
                node_features[idx] = row.iloc[2:-1]  # 假设特征从第二列到倒数第二列
                node_labels[idx] = row['Label']

        node_features = torch.tensor(node_features, dtype=torch.float)
        node_labels = torch.tensor(node_labels, dtype=torch.long)

        data = HeteroData()
        data['company'].x = node_features
        data['company'].y = node_labels

        for edge_type, (src, tgt) in edge_index.items():
            data[('company', edge_type,
                  'company')].edge_index = torch.tensor([src, tgt],
                                                        dtype=torch.long)

        data = ToUndirected()(data)
        data_list.append(data)

    return data_list


import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class DyHGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations):
        super().__init__()
        self.num_layers = len(hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.relation_embedding = torch.nn.Embedding(num_relations,
                                                     in_channels)

        # 创建每种关系的多层 GCN
        for _ in range(num_relations):
            conv_layers = torch.nn.ModuleList()
            for i in range(self.num_layers):
                in_dim = hidden_channels[i - 1] if i > 0 else in_channels
                out_dim = hidden_channels[i]
                conv_layers.append(GCNConv(in_dim, out_dim))
            self.convs.append(conv_layers)

        self.lstm = torch.nn.LSTM(hidden_channels[-1], hidden_channels[-1],
                                  batch_first=True)
        self.out_conv = GCNConv(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index_dict):
        out_list = []
        for i, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
            h = x
            for conv in self.convs[i]:
                h = conv(h, edge_index)
                h = F.relu(h)
            out_list.append(h)

        out = torch.stack(out_list, dim=1).sum(dim=1)
        out, _ = self.lstm(out.unsqueeze(0))
        out = out.squeeze(0)
        out = F.relu(out)
        out = self.out_conv(out,
                            torch.cat(list(edge_index_dict.values()), dim=1))
        return F.log_softmax(out, dim=1)


def train_and_evaluate(model, optimizer, criterion, data_list, train_years,
                       val_year, test_year, params):
    all_results = []
    model.to(device)  # 将模型移到 GPU 上
    for epoch in range(250):
        model.train()
        for year in train_years:
            data = data_list[year - 2014].to(device)  # 将数据移到 GPU 上
            optimizer.zero_grad()
            out = model(data['company'].x, data.edge_index_dict)
            mask = data['company'].y != -1  # 只计算目标节点的损失
            loss = criterion(out[mask], data['company'].y[mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for year, settype in [(train_years[-1], 'train'),
                                  (val_year, 'valid'), (test_year, 'test')]:
                data = data_list[year - 2014].to(device)  # 将数据移到 GPU 上
                out = model(data['company'].x, data.edge_index_dict)
                mask = data['company'].y != -1  # 只评估目标节点
                y_probs = torch.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
                labels = data['company'].y[mask].cpu().numpy()
                results = evaluate(labels, y_probs, epoch, loss.item(),
                                   params=str(params))
                results['settype'] = settype  # 添加 settype 列
                all_results.append(results)

    return pd.concat(all_results)


if __name__ == '__main__':
    removal_percentages = [75]
    for removal_percentage in removal_percentages:
        results = []
        data_list = create_data_list(removal_percentage)

        for i in range(5):
            train_years = list(range(2014, 2017 + i))
            val_year = 2017 + i
            test_year = 2018 + i

            in_channels = data_list[0]['company'].x.shape[1]
            out_channels = 2
            num_relations = len(data_list[0].edge_types)

            hidden_channels = [64, 16]
            weight_decay = 0.01

            model = DyHGCN(in_channels=in_channels,
                           hidden_channels=hidden_channels,
                           out_channels=out_channels,
                           num_relations=num_relations)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                         weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            params = {
                'hidden_channels': hidden_channels,
                'weight_decay': weight_decay,
                'removal_percentage': removal_percentage
            }

            print(
                f"Training on years {train_years}, validating on {val_year}, testing on {test_year}, removal_percentage={removal_percentage}"
            )
            df_results = train_and_evaluate(model, optimizer, criterion,
                                            data_list, train_years, val_year,
                                            test_year, params)

            df_results['repeat'] = i
            df_results.to_csv(
                './result/DyHGCN_result_' + str(removal_percentage) + '_' +
                str(i) + '.csv', index=False, encoding='utf-8')
            results.append(df_results)

        final_results = pd.concat(results, axis=0)
        final_results.to_csv(
            './result/DyHGCN_result_' + str(removal_percentage) + '.csv',
            index=False, encoding='utf-8')
