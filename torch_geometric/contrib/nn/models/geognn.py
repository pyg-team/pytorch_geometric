import torch

from torch_geometric.contrib.nn.conv.geognn_conv import GeoGNNConv
from torch_geometric.nn import global_mean_pool


class GeoGNN(torch.nn.Module):
    """Modified version of GeoGNN from the
    paper `Geometry-enhanced molecular
    representation learning for property prediction`
    <https://www.nature.com/articles/s42256-021-00438-4>`_.

    Args:
        embedding_size (int): The size of the node embeddings.
        layer_num (int): The number of GNN layers.
        dropout (float, optional): Dropout rate for regularization.
            Defaults to 0.2.
    """
    def __init__(self, embedding_size, layer_num, dropout=0.2):
        super(GeoGNN, self).__init__()
        self.layer_num = layer_num
        self.embedding_size = embedding_size

        self.layers = torch.nn.ModuleList()
        for layer_id in range(layer_num):
            self.layers.append(
                GeoGNNConv(embedding_size, dropout,
                           last_act=(layer_id != self.layer_num - 1)))

        # Readout for graph-level embedding
        self.graph_pool = global_mean_pool

    def forward(self, ab_graph, ba_graph):
        """Forward pass of GeoGNN.

        Args:
            ab_graph (torch_geometric.data.Data): Atom bond graph.
            ba_graph (torch_geometric.data.Data): Bond atom graph.

        Returns:
            ab_graph_repr (torch.Tensor): Graph-level representation of the
                atom bond graph.
            ba_graph_repr (torch.Tensor): Graph-level representation of the
                bond atom graph.
            node_repr (torch.Tensor): Node-level representation of the atom
                bond graph.
            edge_repr (torch.Tensor): Node-level representation of the bond
                atom graph.
        """
        # Atom bond graph
        ab_x, ab_edge_index, ab_batch = \
            ab_graph.x, ab_graph.edge_index, ab_graph.batch
        # Bond atom graph
        ba_x, ba_edge_index, ba_edge_attr, ba_batch = \
            ba_graph.x, ba_graph.edge_index, ba_graph.edge_attr, ba_graph.batch

        node_hidden_list = [ab_x]
        edge_hidden_list = [ba_x]

        for layer_id in range(self.layer_num):

            node_hidden = self.layers[layer_id](node_hidden_list[layer_id],
                                                ab_edge_index,
                                                edge_hidden_list[layer_id],
                                                ab_batch)

            edge_hidden = self.layers[layer_id](edge_hidden_list[layer_id],
                                                ba_edge_index, ba_edge_attr,
                                                ba_batch)

            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]

        ab_graph_repr = self.graph_pool(node_repr, ab_batch)
        ba_graph_repr = self.graph_pool(edge_repr, ba_batch)
        return ab_graph_repr, ba_graph_repr, node_repr, edge_repr
