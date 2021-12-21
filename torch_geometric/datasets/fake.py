from typing import Optional, Callable

import random
from itertools import product
from collections import defaultdict

import torch

from torch_geometric.data import InMemoryDataset, Data, HeteroData
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops


class FakeDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool): Whether the graphs to generate are undirected.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: int = 10,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)

        data = Data()

        data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)

        if self.num_channels > 0:
            data.x = torch.randn(num_nodes, self.num_channels)
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        return data


class FakeHeteroDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.HeteroData` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        num_node_types (int, optional): The number of node types.
            (default: :obj:`3`)
        num_edge_types (int, optional): The number of edge types.
            (default: :obj:`6`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        avg_num_channels (int, optional): The average number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_graphs: int = 1,
        num_node_types: int = 3,
        num_edge_types: int = 6,
        avg_num_nodes: int = 1000,
        avg_degree: int = 10,
        avg_num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform)

        if task == 'auto':
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.node_types = [f'v{i}' for i in range(max(num_node_types, 1))]

        edge_types = []
        edge_type_product = list(product(self.node_types, self.node_types))
        while len(edge_types) < max(num_edge_types, 1):
            edge_types.extend(edge_type_product)
        random.shuffle(edge_types)

        self.edge_types = []
        count = defaultdict(lambda: 0)
        for edge_type in edge_types[:max(num_edge_types, 1)]:
            rel = f'e{count[edge_type]}'
            count[edge_type] += 1
            self.edge_types.append((edge_type[0], rel, edge_type[1]))

        self.avg_num_nodes = max(avg_num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = [
            get_num_channels(avg_num_channels) for _ in self.node_types
        ]
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> HeteroData:
        data = HeteroData()

        iterator = zip(self.node_types, self.num_channels)
        for i, (node_type, num_channels) in enumerate(iterator):
            num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)

            store = data[node_type]

            if num_channels > 0:
                store.x = torch.randn(num_nodes, num_channels)
            else:
                store.num_nodes = num_nodes

            if self._num_classes > 0 and self.task == 'node' and i == 0:
                store.y = torch.randint(self._num_classes, (num_nodes, ))

        for (src, rel, dst) in self.edge_types:
            store = data[(src, rel, dst)]

            store.edge_index = get_edge_index(
                data[src].num_nodes,
                data[dst].num_nodes,
                self.avg_degree,
                is_undirected=False,
                remove_loops=False,
            )

            if self.edge_dim > 1:
                store.edge_attr = torch.rand(store.num_edges, self.edge_dim)
            elif self.edge_dim == 1:
                store.edge_weight = torch.rand(store.num_edges)

            pass

        if self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        return data


###############################################################################


def get_num_nodes(avg_num_nodes: int, avg_degree: int) -> int:
    min_num_nodes = max(3 * avg_num_nodes // 4, avg_degree)
    max_num_nodes = 5 * avg_num_nodes // 4
    return random.randint(min_num_nodes, max_num_nodes)


def get_num_channels(num_channels) -> int:
    min_num_channels = 3 * num_channels // 4
    max_num_channels = 5 * num_channels // 4
    return random.randint(min_num_channels, max_num_channels)


def get_edge_index(num_src_nodes: int, num_dst_nodes: int, avg_degree: int,
                   is_undirected: bool = False,
                   remove_loops: bool = False) -> torch.Tensor:
    num_edges = num_src_nodes * avg_degree
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.int64)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.int64)
    edge_index = torch.stack([row, col], dim=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index
