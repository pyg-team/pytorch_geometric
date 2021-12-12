import torch
from torch import nn, Tensor
from torch_geometric.typing import Adj

from torch_geometric.nn.conv import LGConv


class LightGCN(nn.Module):
    """ LightGCN implementation in PyG. From :
    "LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation"
    Adapted from the original author's repo:
    https://github.com/gusye1234/LightGCN-PyTorch
    """
    def __init__(self,
                 config: dict,
                 device=None,
                 **kwargs):
        super().__init__()

        self.num_users = config["n_users"]
        self.num_items = config["m_items"]
        self.embedding_size = config["embedding_size"]
        self.in_channels = self.embedding_size
        self.out_channels = self.embedding_size
        self.num_layers = config["num_layers"]

        # 0-th layer embedding.
        self.embedding_user_item = torch.nn.Embedding(
            num_embeddings=self.num_users + self.num_items,
            embedding_dim=self.embedding_size)
        self.alpha = None

        # random normal init seems to be a better choice
        # when lightGCN actually don't use any non-linear activation function
        nn.init.normal_(self.embedding_user_item.weight, std=0.1)
        print('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.convs.append(LGConv(**kwargs))

        for _ in range(1, self.num_layers):
            self.convs.append(LGConv(**kwargs))

        self.device = None
        if device is not None:
            self.convs.to(device)
            self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        alpha = 1 / (1 + self.num_layers)
        out = alpha * x
        for conv in self.convs:
            x = conv(x, edge_index)
            out = out + alpha * x
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
