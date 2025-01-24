from typing import List, Optional, Tuple

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.utils import sort_edge_index


def test_sort_edge_index():
    edge_index = torch.tensor([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = torch.tensor([[1], [2], [3], [4]])

    out = sort_edge_index(edge_index)
    assert out.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    if torch_geometric.typing.WITH_PT113:
        torch_geometric.typing.MAX_INT64 = 1
        out = sort_edge_index(edge_index)
        torch_geometric.typing.MAX_INT64 = torch.iinfo(torch.int64).max
        assert out.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]

    out = sort_edge_index((edge_index[0], edge_index[1]))
    assert isinstance(out, tuple)
    assert out[0].tolist() == [0, 1, 1, 2]
    assert out[1].tolist() == [1, 0, 2, 1]

    out = sort_edge_index(edge_index, None)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1] is None

    out = sort_edge_index(edge_index, edge_attr)
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1].tolist() == [[4], [3], [2], [1]]

    out = sort_edge_index(edge_index, [edge_attr, edge_attr.view(-1)])
    assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert out[1][0].tolist() == [[4], [3], [2], [1]]
    assert out[1][1].tolist() == [4, 3, 2, 1]


def test_sort_edge_index_jit():
    @torch.jit.script
    def wrapper1(edge_index: Tensor) -> Tensor:
        return sort_edge_index(edge_index)

    @torch.jit.script
    def wrapper2(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return sort_edge_index(edge_index, edge_attr)

    @torch.jit.script
    def wrapper3(
        edge_index: Tensor,
        edge_attr: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        return sort_edge_index(edge_index, edge_attr)

    edge_index = torch.tensor([[2, 1, 1, 0], [1, 2, 0, 1]])
    edge_attr = torch.tensor([[1], [2], [3], [4]])

    out = wrapper1(edge_index)
    assert out.size() == edge_index.size()

    out = wrapper2(edge_index, None)
    assert out[0].size() == edge_index.size()
    assert out[1] is None

    out = wrapper2(edge_index, edge_attr)
    assert out[0].size() == edge_index.size()
    assert out[1].size() == edge_attr.size()

    out = wrapper3(edge_index, [edge_attr, edge_attr.view(-1)])
    assert out[0].size() == edge_index.size()
    assert len(out[1]) == 2
    assert out[1][0].size() == edge_attr.size()
    assert out[1][1].size() == edge_attr.view(-1).size()
