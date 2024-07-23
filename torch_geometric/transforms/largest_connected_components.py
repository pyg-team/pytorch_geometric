import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix


@functional_transform('largest_connected_components')
class LargestConnectedComponents(BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph
    (functional name: :obj:`largest_connected_components`).

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
        connection (str, optional): Type of connection to use for directed
            graphs, can be either :obj:`'strong'` or :obj:`'weak'`.
            Nodes `i` and `j` are strongly connected if a path
            exists both from `i` to `j` and from `j` to `i`. A directed graph
            is weakly connected if replacing all of its directed edges with
            undirected edges produces a connected (undirected) graph.
            (default: :obj:`'weak'`)
    """
    def __init__(
        self,
        num_components: int = 1,
        connection: str = 'weak',
    ) -> None:
        assert connection in ['strong', 'weak'], 'Unknown connection type'
        self.num_components = num_components
        self.connection = connection

    def forward(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        assert data.edge_index is not None

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(
            adj, connection=self.connection)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset_np = np.in1d(component, count.argsort()[-self.num_components:])
        subset = torch.from_numpy(subset_np)
        subset = subset.to(data.edge_index.device, torch.bool)

        return data.subgraph(subset)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'
