from torch_geometric.nn.conv import utils


def test_gnn_cheatsheet():
    assert utils.paper_title('GCNConv') == ('Semi-supervised Classification '
                                            'with Graph Convolutional '
                                            'Networks')
    assert utils.paper_link('GCNConv') == 'https://arxiv.org/abs/1609.02907'

    assert utils.supports_sparse_tensor('GCNConv')
    assert not utils.supports_sparse_tensor('ChebConv')

    assert utils.supports_edge_weights('GraphConv')
    assert not utils.supports_edge_weights('SAGEConv')

    assert utils.supports_edge_features('GATConv')
    assert not utils.supports_edge_features('SimpleConv')

    assert utils.supports_bipartite_graphs('SAGEConv')
    assert not utils.supports_bipartite_graphs('GCNConv')
    assert not utils.supports_bipartite_graphs('FAConv')

    assert utils.supports_static_graphs('GCNConv')
    assert not utils.supports_static_graphs('GATConv')

    assert utils.supports_lazy_initialization('SAGEConv')
    assert not utils.supports_lazy_initialization('GatedGraphConv')

    assert utils.processes_heterogeneous_graphs('RGCNConv')
    assert utils.processes_heterogeneous_graphs('HeteroConv')
    assert not utils.processes_heterogeneous_graphs('GCNConv')

    assert utils.processes_hypergraphs('HypergraphConv')
    assert not utils.processes_hypergraphs('SAGEConv')

    assert utils.processes_point_clouds('DynamicEdgeConv')
    assert utils.processes_point_clouds('XConv')
    assert not utils.processes_point_clouds('CuGraphSAGEConv')


def test_egnn_cheatsheet():
    assert utils.paper_title('EGNNConv') == (
        'Exploiting Edge Features in Graph Neural Networks')
    assert utils.paper_link('EGNNConv') == 'https://arxiv.org/abs/1809.02709'

    assert utils.supports_sparse_tensor('EGNNConv')
    assert not utils.supports_edge_weights('EGNNConv')
    assert utils.supports_edge_features('EGNNConv')
    assert not utils.supports_bipartite_graphs('EGNNConv')
    assert not utils.supports_static_graphs('EGNNConv')
    assert utils.supports_lazy_initialization('EGNNConv')
