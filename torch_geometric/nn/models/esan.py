import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter


def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([
        torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
        torch.cumsum(num_subgraphs, dim=0)
    ])
    graph_offset = tmp[batched_data.batch]
    subgraph_id = batched_data.subgraph_batch + graph_offset

    return pool(h_node, subgraph_id)


class DSnetwork(torch.nn.Module):
    r"""DeepSets (DS) Network from the `"Equivariant Subgraph Aggregation
    Networks" <https://arxiv.org/abs/2110.02910>`_ paper.
    :class:`DSnetwork` outputs a graph representation based on the aggregation
    of the subgraph representations.

    Both :class:`DSnetwork` and :class:`DSSnetwork` models comprise of
    three components:

    1. An equivariant feature encoder consisting of H-equivariant layers
    2. A subgraph readout layer
    3. A set encoder.

    .. note::

        For an example of using a pretrained DSnetwork variant, see
        `examples/esan.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        esan.py>`_.

    Args:
        num_layers(int): number of graph neural network (GNN) layers
        in_dim(int): input node feature dimensionality
        emb_dim(int): hidden node feature dimensionality
        num_tasks(int): number of prediction tasks
        eature_encoder(torch.nn.Module): node feature encoder module
        GNNConv(torch.nn.Module): graph neural network convolution module
     """
    def __init__(self, num_layers: int, in_dim: int, emb_dim: int,
                 num_tasks: int, feature_encoder: torch.nn.Module,
                 GNNConv: torch.nn.Module):
        super(DSnetwork, self).__init__()
        self.emb_dim = emb_dim
        self.feature_encoder = feature_encoder

        gnn_list = []
        bn_list = []

        # Create num_layers layers of GNNs and batch normalization (BN) layers
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

        # Save the GNNs and BN layers as module lists
        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.bn_list = torch.nn.ModuleList(bn_list)

        # Final layers to produce output predictions
        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks))

    def reset_parameters(self):
        reset(self.gnn_list)
        reset(self.bn_list)

    def forward(self, batched_data):
        # Unpack input batch data
        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr

        # Encode node features
        x = self.feature_encoder(x)

        # Apply GNN layers
        for gnn, bn in zip(self.gnn_list, self.bn_list):
            h1 = bn(gnn(x, edge_index, edge_attr))
            x = F.relu(h1)

        # Pool node features across subgraphs to obtain
        # subgraph representations
        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)

        # Pool subgraph representations to obtain graph representation
        h_graph = scatter(src=h_subgraph, index=batched_data.subgraph_id_batch,
                          dim=0, reduce="mean")

        # Apply final layers and return output
        return self.final_layers(h_graph)


class DSSnetwork(DSnetwork):
    r"""Deep Sets for Symmetric elements (DSS) Network from the
    `"Equivariant Subgraph Aggregation Networks"
    <https://arxiv.org/abs/2110.02910>`_
    paper. The key additional functionality of :class:`DSSnetwork` compared to
    :class:`DSnetwork` is an additional information-sharing component in
    the H-equivariant layers in the DSS-GNN model.

    .. note::

        For an example of using a pretrained DSSnetwork variant, see
        `examples/esan.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        esan.py>`_.

    Args:
        num_layers(int): number of graph neural network (GNN) layers
        in_dim(int): input node feature dimensionality
        emb_dim(int): hidden node feature dimensionality
        num_tasks(int): number of prediction tasks
        eature_encoder(torch.nn.Module): node feature encoder module
        GNNConv(torch.nn.Module): graph neural network convolution module
    """
    def __init__(self, num_layers: int, in_dim: int, emb_dim: int,
                 num_tasks: int, feature_encoder: torch.nn.Module,
                 GNNConv: torch.nn.Module):
        super().__init__(num_layers=num_layers, in_dim=in_dim, emb_dim=emb_dim,
                         num_tasks=num_tasks, feature_encoder=feature_encoder,
                         GNNConv=GNNConv)

        gnn_sum_list = []
        bn_sum_list = []

        # Initialize GNNs for data sharing module
        for i in range(num_layers):
            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim,
                                        emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

    def reset_parameters(self):
        reset(self.gnn_list)
        reset(self.bn_list)
        reset(self.gnn_sum_list)
        reset(self.bn_sum_list)

    def forward(self, batched_data):
        # Unpack input batch data
        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch

        # Encode node features
        x = self.feature_encoder(x)

        # Apply GNN layers
        for i in range(len(self.gnn_list)):
            # Unpack GNN layer and batch norm layer for this iteration
            gnn = self.gnn_list[i]
            bn = self.bn_list[i]
            gnn_sum = self.gnn_sum_list[i]
            bn_sum = self.bn_sum_list[i]

            # Apply GNN and batch norm layer
            h1 = bn(gnn(x, edge_index, edge_attr))

            # Compute graph offset and node indices
            # for summing node features across subgraphs
            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat([
                torch.zeros(1, device=num_nodes_per_subgraph.device,
                            dtype=num_nodes_per_subgraph.dtype),
                torch.cumsum(num_nodes_per_subgraph, dim=0)
            ])
            graph_offset = tmp[batch]
            node_idx = graph_offset + batched_data.subgraph_n_id

            # Sum node features across subgraphs
            x_sum = scatter(src=x, index=node_idx, dim=0, reduce="mean")

            # Information sharing component
            h2 = bn_sum(
                gnn_sum(
                    x_sum, batched_data.orig_edge_index,
                    batched_data.orig_edge_attr
                    if edge_attr is not None else edge_attr))

            # Apply activation function and update node features
            # for next iteration
            x = F.relu(h1 + h2[node_idx])

        # Pool node features across subgraphs to
        # obtain subgraph representations
        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)

        # Pool subgraph representations to obtain graph representation
        h_graph = scatter(src=h_subgraph, index=batched_data.subgraph_id_batch,
                          dim=0, reduce="mean")

        # Apply final layers and return output
        return self.final_layers(h_graph)
