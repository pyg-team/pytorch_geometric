import pytest
import torch

from torch_geometric.nn import GATConv, GCNConv, GraphConv
from torch_geometric.nn.aggr import GraphMultisetTransformer


@pytest.mark.parametrize('layer_norm', [False, True])
def test_graph_multiset_transformer(layer_norm):
    num_avg_nodes = 4
    in_channels, hidden_channels, out_channels = 32, 64, 16

    x = torch.randn((6, in_channels))
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    index = torch.tensor([0, 0, 0, 0, 1, 1])

    for GNN in [GraphConv, GCNConv, GATConv]:
        gmt = GraphMultisetTransformer(
            in_channels,
            hidden_channels,
            out_channels,
            Conv=GNN,
            num_nodes=num_avg_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_I'],
            num_heads=4,
            layer_norm=layer_norm,
        )
        gmt.reset_parameters()
        assert str(gmt) == ("GraphMultisetTransformer(32, 16, "
                            "pool_sequences=['GMPool_I'])")
        assert gmt(x, index, edge_index=edge_index).size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels,
            hidden_channels,
            out_channels,
            Conv=GNN,
            num_nodes=num_avg_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G'],
            num_heads=4,
            layer_norm=layer_norm,
        )
        gmt.reset_parameters()
        assert str(gmt) == ("GraphMultisetTransformer(32, 16, "
                            "pool_sequences=['GMPool_G'])")
        assert gmt(x, index, edge_index=edge_index).size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels,
            hidden_channels,
            out_channels,
            Conv=GNN,
            num_nodes=num_avg_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'GMPool_I'],
            num_heads=4,
            layer_norm=layer_norm,
        )
        gmt.reset_parameters()
        assert str(gmt) == ("GraphMultisetTransformer(32, 16, "
                            "pool_sequences=['GMPool_G', 'GMPool_I'])")
        assert gmt(x, index, edge_index=edge_index).size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels,
            hidden_channels,
            out_channels,
            Conv=GNN,
            num_nodes=num_avg_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=layer_norm,
        )
        gmt.reset_parameters()
        assert str(gmt) == ("GraphMultisetTransformer(32, 16, pool_sequences="
                            "['GMPool_G', 'SelfAtt', 'GMPool_I'])")
        assert gmt(x, index, edge_index=edge_index).size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels,
            hidden_channels,
            out_channels,
            Conv=GNN,
            num_nodes=num_avg_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=layer_norm,
        )
        gmt.reset_parameters()
        assert str(gmt) == ("GraphMultisetTransformer(32, 16, pool_sequences="
                            "['GMPool_G', 'SelfAtt', 'SelfAtt', 'GMPool_I'])")
        assert gmt(x, index, edge_index=edge_index).size() == (2, out_channels)
