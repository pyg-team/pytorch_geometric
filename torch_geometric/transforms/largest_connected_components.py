import numpy as np
import scipy.sparse as sp
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph


class LargestConnectedComponents(BaseTransform):
    r"""Selects subgraph corresponding to the K largest connected components in the graph

    Args:
        num_components (int): Number of largest components to keep (default: :obj:`1`)
    """

    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data):
        adj = to_scipy_sparse_matrix(data.edge_index, data.edge_attr, data.num_nodes)

        _, components = sp.csgraph.connected_components(adj)
        values, counts = np.unique(components, return_counts=True)

        subset = np.in1d(components, values[counts.argsort()[-self.num_components:]])
        subset = torch.from_numpy(subset)

        data.edge_index, data.edge_attr = subgraph(subset=subset,
                                                   edge_index=data.edge_index,
                                                   edge_attr=data.edge_attr,
                                                   relabel_nodes=True,
                                                   num_nodes=data.num_nodes)

        if data.x is not None:
            data.x = data.x[subset]

        if data.y is not None:
            data.y = data.y[subset]

        data.num_nodes = int(subset.sum())

        return data
