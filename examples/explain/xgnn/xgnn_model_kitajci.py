import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    paper: Semi-Supervised Classification with Graph Convolutional Networks
    """
    # 模型的参数包括weight和bias
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    # 权重初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    # 类似于tostring
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    # 计算A~ X W(0)
    def forward(self, input, adj):
        # input.shape = [max_node, features] = X
        # adj.shape = [max_node, max_node] = A~
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    # feature的个数；最终的分类数

    def __init__(self, nfeat, nclass, dropout):
        """ As per paper """
        """ 3 layers of GCNs with output dimensions equal to 32, 48, 64 respectively and average all node features """
        """ Final classifier with 2 fully connected layers and hidden dimension set to 32 """
        """ Activation function - ReLu (Mutag) """
        super(GCN, self).__init__()

        self.dropout = dropout

        self.gc1 = GraphConvolution(nfeat, 32)
        self.gc2 = GraphConvolution(32, 48)
        self.gc3 = GraphConvolution(48, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, nclass)

    def forward(self, x, adj):
        # x.shape = [max_node, features]
        # adj.shape = [max_node, max_node]
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))


        y = torch.mean(x, 0)  # 采用mean作为聚合函数聚合所有结点的特征
        y = F.relu(self.fc1(y))
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.softmax(self.fc2(y), dim=0)

        return y


if __name__ == '__main__':
    input = torch.rand(29, 7)
    adj = torch.rand(29, 29)

    model = GCN(nfeat=7,  # nfeat = 7
                nclass=2,  # nclass = 7
                dropout=0.1)

    output = model(input, adj)
    print(output.size())