Hierarchical Neighborhood Sampling
==================================

One of the design principles of :pyg:`PyG` is that models and data loading routines should be exchangeable to allow for flexible GNN and data loading experimentation.
As such, models can usually be written in a data loading agnostic fashion, independent of whether one applies full-batch or mini-batch training strategies via, *e.g.*, :class:`~torch_geometric.loader.DataLoader`, :class:`~torch_geometric.loader.NeighborLoader` or :class:`~torch_geometric.loader.ClusterLoader`.
However, in some scenarios, this flexibility comes at the cost of performance, as the model cannot exploit special characteristics of the underyling data routine.
One such limitation is that a GNN trained with the :class:`~torch_geometric.loader.NeighborLoader` routine iteratively builds representations for *all* nodes at *all* depths of the network, although nodes sampled in later hops do not contribute to the node representations of seed nodes in later GNN layers anymore, thus performing useless computation.

*Hierarchical Neighborhood Sampling* or *Hierarchical Graph Adjacency Matrix (HGAM)* is a technique available in :pyg:`PyG` to eliminate this overhead and speeds up training and inference in mini-batch GNNs.
Its main idea is to progressively trim the adjacency matrix of the returned subgraph before inputting it to each GNN layer.
It works seamlessly across several models, basically reducing the amount of compute necessary to generate the representations for the seed node of the given mini-batch.

Crucially, HGAM recognizes that the computation of the final node representations is only necessary for the seed nodes (which are the real target of the batch computation).
Thus, HGAM allows for every layer of the GNN to compute only the representations of the nodes that are necessary for that layer, leading to a reduction of the computation and a speed up of the training process that grows with the depth of the GNN being considered.
In practice, this is achieved **trimming the adjacency matrix** and the various **features matrices** as the computation proceeds throughout the GNN layers.
This is in line with the fact that in order to compute the representation for the seed/target nodes (from which the batch subgraph was build via sampling methods), the depth of the relevant neighborhood shrinks as we proceed through the layers of the GNN.
The trimming applied by HGAM is possible as the nodes of the subgraph built via sampling are ordered according to a *Breadth First Search (BFS)* strategy, meaning that the rows and columns of the adjacency matrix refer to a node ordering that starts with the seed nodes (in any order) followed by the 1-hop neighbors of the first seed node, followed by the 1-hop sampled neighbors of the second seed node and so on.
The BFS ordering of nodes in a mini-batch allows for incremental trimming (reduction) of the adjacency matrix of the subgraph.
This progressive trimming is done in a computational convenient manner thanks to the BFS ordering, that causes the nodes more distant from the seed nodes to be appear farther away in the list of ordered nodes.

To support this trimming and implement it effectively, the :class:`~torch_geometric.loaderl.NeighborLoader` implementation in :pyg:`PyG` and in :pyg-lib:`pyg-lib` additionally return the number of nodes and edges sampled in hop.
This information allows for fast manipulation of the adjacency matrix, which in turns lead to great computation reduction.
The :class:`~torch_geometric.loader.NeighborLoader` prepares this metadata via the dedicated attributes :obj:`num_sampled_nodes` and :obj:`num_sampled_edges`.
It can be accessed from the :class:`~torch_geometric.data.Batch` object returned for both homogenous and heterogeneous graphs.

To sum up, HGAM is special data structure that enables efficient message passing computation in :class:`~torch_geometric.loader.NeighborLoader` scenarios.
HGAM is implemented in :pyg:`PyG` and can be utilized via the special `:meth:`~torch_geometric.utils.trim_to_layer` functionality.
HGAM is currently an option that :pyg:`PyG` users are free to switch on, or leave it off (current default).

Usage
-----

Here below some examples of how to use the NeighborLoader along with a fully functional example of HGAM utilization are provided.

* Homogenous data example:

.. code-block:: python

    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import NeighborLoader

    data = Planetoid(path, name='Cora')[0]

    loader = NeighborLoader(
        data,
        # Sample 10 neighbors for each node for 3 iterations
        num_neighbors=[10] * 3,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=data.train_mask,
    )

    sampled_data = next(iter(loader))
    print(sampled_data)
    >>> Data(x=[1883, 1433], edge_index=[2, 5441], y=[1883], train_mask=[1883], val_mask=[1883], test_mask=[1883], n_id=[1883], e_id=[5441], num_sampled_nodes=[4], num_sampled_edges=[3], input_id=[128], batch_size=128)

    print(sampled_data.num_sampled_nodes)
    >>> [128, 425, 702, 628] # Number of sampled nodes per iteration/layer
    print(sampled_data.num_sampled_edges)
    >>> [520, 2036, 2885] # Number of sampled edges per iteration/layer


* Heterogeneous data example:

.. code-block:: python

    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.loader import NeighborLoader

    hetero_data = OGB_MAG(root='../data')[0]

    loader = NeighborLoader(
        hetero_data,
        # Sample 10 neighbors for each node and edge type for 2 iterations
        num_neighbors={key: [10] * 2 for key in hetero_data.edge_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=128,
        input_nodes=('paper', hetero_data['paper'].train_mask),
    )

    sampled_hetero_data = next(iter(loader))
    print(sampled_hetero_data)
    >>> HeteroData(
        paper={
            x=[2275, 128],
            year=[2275],
            y=[2275],
            train_mask=[2275],
            val_mask=[2275],
            test_mask=[2275],
            n_id=[2275],
            num_sampled_nodes=[3],
            input_id=[128],
            batch_size=128,
        },
        author={
            num_nodes=2541,
            n_id=[2541],
            num_sampled_nodes=[3],
        },
        institution={
            num_nodes=0,
            n_id=[0],
            num_sampled_nodes=[3],
        },
        field_of_study={
            num_nodes=0,
            n_id=[0],
            num_sampled_nodes=[3],
        },
        (author, affiliated_with, institution)={
            edge_index=[2, 0],
            e_id=[0],
            num_sampled_edges=[2],
        },
        (author, writes, paper)={
            edge_index=[2, 3255],
            e_id=[3255],
            num_sampled_edges=[2],
        },
        (paper, cites, paper)={
            edge_index=[2, 2691],
            e_id=[2691],
            num_sampled_edges=[2],
        },
        (paper, has_topic, field_of_study)={
            edge_index=[2, 0],
            e_id=[0],
            num_sampled_edges=[2],
        }
        )
    print(sampled_hetero_data['paper'].num_sampled_nodes)
    >>> [128, 508, 1598] # Number of sampled nodes per iteration/layer for 'paper' node type

    print(sampled_hetero_data['author', 'writes', 'paper'].num_sampled_edges)
    >>>> [629, 2621] # Number of sampled edges per iteration/layer for 'author_writes_paper' edge type


The returned by NeighborLoader :obj:`num_sampled_nodes` and :obj:`num_sampled_edges` fields can be used by :obj:`trim_to_layer` utils function.
The class :class:`~torch_geometric.utils.trim_to_layer.TrimToLayer` can be used as a layer that trims the adjacency matrix as needed using the function :obj:`trim_to_layer`.
Please see below how this can be done.

.. code-block::  python

    import os.path as osp
    from typing import List, Optional

    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn import Linear, ModuleList
    from tqdm import tqdm

    from torch_geometric.datasets import Reddit
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.typing import Adj, OptTensor
    from torch_geometric.utils.trim_to_layer import TrimToLayer
    from torch_geometric.nn.conv import GCNConv as GCNconv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)

    data = dataset[0].to(device, 'x', 'y')
    kwargs = {'batch_size': 8, 'num_workers': 4, 'persistent_workers': True}
    loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 5, 5], shuffle=True, **kwargs)

    class myGCN(torch.nn.Module):
        def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                num_layers: int = 3
                ):

            super().__init__()
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.num_layers = num_layers

            self.convs = ModuleList()
            self.convs.append(GCNconv(in_channels, hidden_channels))
            for _ in range(num_layers-1):
                self.convs.append(GCNconv(hidden_channels, hidden_channels))

            self.Lin = Linear(hidden_channels, out_channels)
            self._trim = TrimToLayer()

        def forward(self, x: Tensor, edge_index: Adj,
            *, edge_weight: Tensor = None,
            edge_attr: Tensor = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None) -> Tensor:

            for i in range(self.num_layers):
                if num_sampled_nodes_per_hop is not None:
                    x, edge_index, value = self._trim(
                        i,
                        num_sampled_nodes_per_hop,
                        num_sampled_edges_per_hop,
                        x,
                        edge_index,
                        edge_weight if edge_weight is not None else edge_attr,
                        )
                x = self.convs[i](x, edge_index)

            x = self.Lin(x)
            return x

    def train(trim=False):
        for batch in tqdm(loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            if not trim:
                out = model(batch.x, batch.edge_index)
            else:
                out = model(
                batch.x,
                batch.edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
                )

            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    model = myGCN(dataset.num_features, hidden_channels=32, out_channels=dataset.num_classes)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(trim=True)
