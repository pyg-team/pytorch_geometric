import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


class FocalLoss(nn.Module):
    # gamma=0,alpha=0.86,gcn层数为1，应该最优，可以平衡欺诈和非欺诈之间的识别
    def __init__(self, gamma=0, alpha=0.82, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * pt ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * (1 - pt) ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# meta之间的注意力机制
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)
class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # 一个GAT层对应一个metapath
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class HAN(nn.Module):
    # num_meta_paths：数据集的边关系的数量
    # in_size：特征维度
    # hidden_size：隐藏层维度 / num_head
    # out_size：输出类别数
    # num_head：HANLayer层的多头注意力的头数
    # dropout
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        # 多层HANLayer
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        # 分类，全连接层
        self.predict = nn.Sequential(
            nn.Linear(hidden_size * num_heads[-1], out_size),
            nn.Sigmoid()
        )

    def forward(self, g, h):
        # 遍历每个HANLayers
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)