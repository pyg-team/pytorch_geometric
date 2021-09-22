import torch
from torch import Tensor
from torch.nn import Embedding


class KEModel(torch.nn.Module):
    def __init__(self, num_nodes: int, num_relations: int,
                 hidden_channels: int, sparse: bool = False):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = Embedding(num_nodes, hidden_channels, sparse=sparse)
        self.rel_emb = Embedding(num_relations, hidden_channels, sparse=sparse)

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """"""
        raise NotImplementedError

    def loader(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'hidden_channels={self.hidden_channels})')
