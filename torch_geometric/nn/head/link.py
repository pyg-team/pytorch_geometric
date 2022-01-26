import torch
from torch import Tensor


class LinkHead(torch.nn.Module):
    def forward(self, x: Tensor, edge_label_index: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class ConcatLinkHead(LinkHead):
    def __init__(self, in_channels: int, combine: str = "concat"):
        super().__init__()
        assert combine in ['concat', 'average', 'hadamard', 'l1', '2']
        self.combine = combine

    def forward(self, x: Tensor, edge_label_index: Tensor) -> Tensor:
        """"""
        row, col = edge_label_index
        return self.MLP(torch.cat([x[row], x[col]], dim=-1))


class InnerProductLinkHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, edge_label_index: Tensor) -> Tensor:
        """"""
        row, col = edge_label_index
        return (x[row] * x[col]).sum(dim=-1, keepdim=True)
