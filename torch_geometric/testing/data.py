from typing import Callable, Optional

import torch
from torch import Tensor

from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.utils import coalesce as coalesce_fn


def get_random_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    num_edges: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    coalesce: bool = False,
) -> Tensor:
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=dtype,
                        device=device)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=dtype,
                        device=device)
    edge_index = torch.stack([row, col], dim=0)

    if coalesce:
        edge_index = coalesce_fn(edge_index)

    return edge_index


def get_random_tensor_frame(
    num_rows: int,
    device: Optional[torch.device] = None,
) -> TensorFrame:

    feat_dict = {
        torch_frame.categorical:
        torch.randint(0, 3, size=(num_rows, 3), device=device),
        torch_frame.numerical:
        torch.randn(size=(num_rows, 2), device=device),
    }
    col_names_dict = {
        torch_frame.categorical: ['a', 'b', 'c'],
        torch_frame.numerical: ['x', 'y'],
    }
    y = torch.randn(num_rows, device=device)

    return torch_frame.TensorFrame(
        feat_dict=feat_dict,
        col_names_dict=col_names_dict,
        y=y,
    )


class FakeHeteroDataset(InMemoryDataset):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__(transform=transform)

        data = HeteroData()

        num_papers = 100
        num_authors = 10

        data['paper'].x = torch.randn(num_papers, 16)
        data['author'].x = torch.randn(num_authors, 8)

        edge_index = get_random_edge_index(
            num_src_nodes=num_papers,
            num_dst_nodes=num_authors,
            num_edges=300,
        )
        data['paper', 'author'].edge_index = edge_index
        data['author', 'paper'].edge_index = edge_index.flip([0])

        data['paper'].y = torch.randint(0, 4, (num_papers, ))

        perm = torch.randperm(num_papers)
        data['paper'].train_mask = torch.zeros(num_papers, dtype=torch.bool)
        data['paper'].train_mask[perm[0:60]] = True
        data['paper'].val_mask = torch.zeros(num_papers, dtype=torch.bool)
        data['paper'].val_mask[perm[60:80]] = True
        data['paper'].test_mask = torch.zeros(num_papers, dtype=torch.bool)
        data['paper'].test_mask[perm[80:100]] = True

        self.data, self.slices = self.collate([data])
