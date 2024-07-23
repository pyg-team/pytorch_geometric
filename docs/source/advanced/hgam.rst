Hierarchical Neighborhood Sampling
==================================

One of the design principles of :pyg:`PyG` is that models and data loading routines should be exchangeable to allow for flexible GNN and data loading experimentation.
As such, models can usually be written in a data loading agnostic fashion, independent of whether one applies full-batch or mini-batch training strategies via, *e.g.*, :class:`~torch_geometric.loader.DataLoader`, :class:`~torch_geometric.loader.NeighborLoader` or :class:`~torch_geometric.loader.ClusterLoader`.
However, in some scenarios, this flexibility comes at the cost of performance, as the model cannot exploit special characteristics of the underlying data loading routine.
One such limitation is that a GNN trained with the :class:`~torch_geometric.loader.NeighborLoader` routine iteratively builds representations for *all* nodes at *all* depths of the network, although nodes sampled in later hops do not contribute to the node representations of seed nodes in later GNN layers anymore, thus performing useless computation.

*Hierarchical Neighborhood Sampling* or *Hierarchical Graph Adjacency Matrix (HGAM)* is a technique available in :pyg:`PyG` to eliminate this overhead and speeds up training and inference in mini-batch GNNs.
Its main idea is to progressively trim the adjacency matrix of the returned subgraph before inputting it to each GNN layer.
It works seamlessly across several models, basically reducing the amount of compute necessary to generate the representations for the seed node of the given mini-batch.

Crucially, HGAM recognizes that the computation of the final node representations is only necessary for the seed nodes (which are the real target of the batch computation).
Thus, HGAM allows for every layer of the GNN to compute only the representations of the nodes that are necessary for that layer, leading to a reduction of the computation and a speed up of the training process that grows with the depth of the GNN being considered.
In practice, this is achieved by **trimming the adjacency matrix** and the various **features matrices** as the computation proceeds throughout the GNN layers.
This is in line with the fact that in order to compute the representation for the seed/target nodes (from which the mini-batch was build via sampling methods), the depth of the relevant neighborhood shrinks as we proceed through the layers of the GNN.
The trimming applied by HGAM is possible as the nodes of the subgraph built via sampling are ordered according to a *Breadth First Search (BFS)* strategy, meaning that the rows and columns of the adjacency matrix refer to a node ordering that starts with the seed nodes (in any order) followed by the 1-hop neighbors of the first seed node, followed by the 1-hop sampled neighbors of the second seed node and so on.
The BFS ordering of nodes in a mini-batch allows for incremental trimming (reduction) of the adjacency matrix of the subgraph.
This progressive trimming is done in a computational convenient manner thanks to the BFS ordering that causes the nodes more distant from the seed nodes to be appear farther away in the list of ordered nodes.

To support this trimming and implement it effectively, the :class:`~torch_geometric.loader.NeighborLoader` implementation in :pyg:`PyG` and in :pyg:`pyg-lib` additionally return the number of nodes and edges sampled in hop.
This information allows for fast manipulation of the adjacency matrix, which in turns lead to great computation reduction.
The :class:`~torch_geometric.loader.NeighborLoader` prepares this metadata via the dedicated attributes :obj:`num_sampled_nodes` and :obj:`num_sampled_edges`.
It can be accessed from the :class:`~torch_geometric.data.Batch` object returned for both homogeneous and heterogeneous graphs.

To sum up, HGAM is special data structure that enables efficient message passing computation in :class:`~torch_geometric.loader.NeighborLoader` scenarios.
HGAM is implemented in :pyg:`PyG` and can be utilized via the special :meth:`~torch_geometric.utils.trim_to_layer` functionality.
HGAM is currently an option that :pyg:`PyG` users are free to switch on, or leave it off *(current default)*.

Usage
-----

Here, we show examples of how to use the HGAM functionality in combination with :class:`~torch_geometric.loader.NeighborLoader`:

* **Homogeneous data example:**

  .. code-block:: python

      from torch_geometric.datasets import Planetoid
      from torch_geometric.loader import NeighborLoader

      data = Planetoid(path, name='Cora')[0]

      loader = NeighborLoader(
          data,
          num_neighbors=[10] * 3,
          batch_size=128,
      )

      batch = next(iter(loader))
      print(batch)
      >>> Data(x=[1883, 1433], edge_index=[2, 5441], y=[1883], train_mask=[1883],
               val_mask=[1883], test_mask=[1883], batch_size=128,
               num_sampled_nodes=[4], num_sampled_edges=[3])

      print(batch.num_sampled_nodes)
      >>> [128, 425, 702, 628]  # Number of sampled nodes per hop/layer.
      print(batch.num_sampled_edges)
      >>> [520, 2036, 2885]  # Number of sampled edges per hop/layer.

* **Heterogeneous data example:**

  .. code-block:: python

      from torch_geometric.datasets import OGB_MAG
      from torch_geometric.loader import NeighborLoader

      data = OGB_MAG(path)[0]

      loader = NeighborLoader(
          data,
          num_neighbors=[10] * 3,
          batch_size=128,
          input_nodes='paper',
      )

      batch = next(iter(loader))
      print(batch)
      >>> HeteroData(
          paper={
              x=[2275, 128],
              num_sampled_nodes=[3],
              batch_size=128,
          },
          author={
              num_nodes=2541,
              num_sampled_nodes=[3],
          },
          institution={
              num_nodes=0,
              num_sampled_nodes=[3],
          },
          field_of_study={
              num_nodes=0,
              num_sampled_nodes=[3],
          },
          (author, affiliated_with, institution)={
              edge_index=[2, 0],
              num_sampled_edges=[2],
          },
          (author, writes, paper)={
              edge_index=[2, 3255],
              num_sampled_edges=[2],
          },
          (paper, cites, paper)={
              edge_index=[2, 2691],
              num_sampled_edges=[2],
          },
          (paper, has_topic, field_of_study)={
              edge_index=[2, 0],
              num_sampled_edges=[2],
          }
          )
      print(batch['paper'].num_sampled_nodes)
      >>> [128, 508, 1598]  # Number of sampled paper nodes per hop/layer.

      print(batch['author', 'writes', 'paper'].num_sampled_edges)
      >>>> [629, 2621]  # Number of sampled autor<>paper edges per hop/layer.

The attributes :obj:`num_sampled_nodes` and :obj:`num_sampled_edges` can be used by the :meth:`~torch_geometric.utils.trim_to_layer` function inside the GNN:

.. code-block::  python

    from torch_geometric.datasets import Reddit
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import trim_to_layer

    dataset = Reddit(path)
    loader = NeighborLoader(data, num_neighbors=[10, 5, 5], ...)

    class GNN(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, num_layers: int):
            super().__init__()

            self.convs = ModuleList([SAGEConv(in_channels, 64)])
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.lin = Linear(hidden_channels, out_channels)

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            num_sampled_nodes_per_hop: List[int],
            num_sampled_edges_per_hop: List[int],
        ) -> Tensor:

            for i, conv in enumerate(self.convs):
                # Trim edge and node information to the current layer `i`.
                x, edge_index, _ = trim_to_layer(
                    i, num_sampled_nodes_per_hop, num_sampled_edges_per_hop,
                    x, edge_index)

                x = conv(x, edge_index).relu()

            return self.lin(x)

Examples
--------

We provide full examples of HGAM in the :pyg:`PyG` :obj:`examples/` folder:

* :obj:`examples/hierarchical_sampling.py`: An `example <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hierarchical_sampling.py>`__ to show-case the basic usage of HGAM.
* :obj:`examples/hetero/hierarchical_sage.py`: An `example <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hierarchical_sage.py>`__ of HGAM on heterogeneous graphs.
