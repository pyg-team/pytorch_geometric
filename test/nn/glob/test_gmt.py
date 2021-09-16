import torch
from torch_geometric.nn import GraphMultisetTransformer
from torch_geometric.nn import GraphConv, GCNConv, GATConv


def test_graph_multiset_transformer():
    num_avg_nodes = 4
    in_channels, hidden_channels, out_channels = (32, 64, 16)

    x = torch.randn((6, in_channels))
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    for GNN in [GraphConv, GCNConv, GATConv]:
        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_I'],
            num_heads=4, ln=False
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_I'],
            num_heads=4, ln=True
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_G'],
            num_heads=4, ln=False
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'GMPool_I'],
            num_heads=4, ln=False
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4, ln=False
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)

        gmt = GraphMultisetTransformer(
            in_channels, hidden_channels, out_channels,
            conv=GNN, num_nodes=num_avg_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'SelfAtt', 'GMPool_I'],
            num_heads=4, ln=False
        )
        out = gmt(x, batch, edge_index)
        assert out.size() == (2, out_channels)
