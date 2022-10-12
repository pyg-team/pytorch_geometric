import random
from collections import defaultdict
from itertools import product
from typing import Callable, Optional

import torch
import numpy as np
from torch_geometric.utils.mask import index_to_mask

from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.graphgym.config import cfg

from ben_utils import get_k_hop_adjacencies


class RingTransferDataset(InMemoryDataset):
    r"""A synthetic dataset that returns a Ring Transfer dataset.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        num_classes (int, optional): The number of node features.
            (default: :obj:`64`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """
    def __init__(
        self,
        num_graphs: int = 100000,
        num_nodes: int = 10,
        num_classes: int = 5,
        # task: str = "auto",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__('.', transform)

        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self._num_classes = num_classes
        self.kwargs = kwargs
        if cfg.gnn.layers_mp == 1: # the default - otherwise use specified no.
            cfg.gnn.layers_mp = num_nodes//2
        
        split = (self.num_graphs * torch.tensor(cfg.dataset.split)).long()
        data_list, split = self.load_ring_transfer_dataset(self.num_nodes, 
                                                           split=split,
                                                           classes=self._num_classes)        
        
        self.data, self.slices = self.collate(data_list)
        
        # add train/val split masks
        self.data.train_mask = index_to_mask(torch.tensor(split[0]), size=len(self.data.x))
        self.data.val_mask = index_to_mask(torch.tensor(split[1]), size=len(self.data.x))
        self.data.test_mask = index_to_mask(torch.tensor(split[2]), size=len(self.data.x))
        

    def load_ring_transfer_dataset(self, nodes=10, split=[5000, 500, 500], classes=5):
        train = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[0])
        val = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[1])
        test = self.generate_ring_transfer_graph_dataset(nodes, classes=classes, samples=split[2])
        dataset = train + val + test
        return dataset, [list(range(int(split[i]))) for i in range(3)]

    def generate_ring_transfer_graph_dataset(self, nodes, classes=5, samples=10000):
        # Generate the dataset
        dataset = []
        samples_per_class = torch.div(samples, classes, rounding_mode="floor")
        for i in range(samples):
            label = torch.div(i, samples_per_class, rounding_mode="floor")
            target_class = np.zeros(classes)
            target_class[label] = 1.0
            graph = self.generate_ring_transfer_graph(nodes, target_class)
            dataset.append(graph)
        return dataset

    def generate_ring_transfer_graph(self, nodes, target_label):
        opposite_node = nodes // 2

        # Initialise the feature matrix with a constant feature vector
        # TODO: Modify the experiment to use another random constant feature per graph
        x = np.ones((nodes, len(target_label)))

        x[0, :] = 0.0
        x[opposite_node, :] = target_label
        x = torch.tensor(x, dtype=torch.float32)

        edge_index = []
        for i in range(nodes-1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

        # Add the edges that close the ring
        edge_index.append([0, nodes - 1])
        edge_index.append([nodes - 1, 0])

        edge_index = np.array(edge_index, dtype=np.long).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Create a mask for the target node of the graph
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1

        # Add the label of the graph as a graph label
        y = torch.tensor([np.argmax(target_label)], dtype=torch.long)
        
        k_hop_edges, _ = get_k_hop_adjacencies(edge_index, cfg.delay.max_k)
        assert torch.mean((k_hop_edges[0] == edge_index).float())==1.0
        cutoffs = torch.tensor([v.shape[-1] for v in k_hop_edges])
        k_hop_edges = torch.cat(k_hop_edges, dim=1)
        # cutoffs = [int(sum(cutoffs[:i])) for i in range(len(cutoffs))] 
        # cutoffs += [int(k_hop_edges.shape[-1])]
            
        
        # make edge labels for k-hops
        k_hop_labels = []
        for i in range(len(cutoffs)):
            k = i + 1
            k_hop_labels.append(k * torch.ones(cutoffs[i]))
        k_hop_labels = torch.cat(k_hop_labels)
            
        
        return Data(x=x, edge_index=k_hop_edges, 
                    edge_attr=k_hop_labels,
                    mask=mask, y=y,

                    # k_hop_edges=k_hop_edges, 
                    # test_edge_idx=torch.arange(22).reshape((2,-1)),
                    # test_list=[torch.arange(4).reshape((2,-1)) for _ in range(5)],
                    )
