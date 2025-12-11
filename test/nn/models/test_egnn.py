import pytest 

import torch 

from torch_geometric.nn.models import EGNN

"""
in_channels = [2, 3]
hidden_channels = [4, 5]
num_layers = [1, 5]
pos_dim = [3, 6]
edge_dim = [0, 1]
update_pos = [True, False]
act = ["SiLU", "ReLU"]
skip_connection = [True, False]
"""
in_channels = [2]
hidden_channels = [4, 5]
num_layers = [1, 5]
pos_dim = [3, 6]
update_pos = [True, False]
act = ["SiLU", "ReLU", torch.nn.SiLU()]
skip_connection = [False]
@pytest.mark.parametrize('in_channels', in_channels)
@pytest.mark.parametrize('hidden_channels', hidden_channels)
@pytest.mark.parametrize('num_layers', num_layers)
@pytest.mark.parametrize('pos_dim', pos_dim)
@pytest.mark.parametrize('update_pos', update_pos)
@pytest.mark.parametrize('act', act)
@pytest.mark.parametrize('skip_connection', skip_connection)
def test_egnn_parameters(in_channels, hidden_channels, num_layers, pos_dim, update_pos, act, skip_connection):
    model = EGNN(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 num_layers=num_layers,
                 pos_dim=pos_dim,
                 update_pos=update_pos,
                 act=act,
                 skip_connection=skip_connection)
    nb_nodes = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.randn((nb_nodes, in_channels))
    pos = torch.randn((nb_nodes, pos_dim))
    out_node, out_pos = model(x, pos, edge_index)
    assert out_node.size() == (nb_nodes, hidden_channels)
    assert out_pos.size() == (nb_nodes, pos_dim)

@pytest.mark.parametrize('pos_dim', pos_dim)
def test_egnn_equivariance(pos_dim):
    in_channels = 4
    hidden_channels = 4
    num_layers = 2
    model = EGNN(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 num_layers=num_layers,
                 pos_dim=pos_dim,
                 skip_connection=True)
    nb_nodes = 5
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 0],
                               [1, 0, 2, 1, 4, 3, 4]])
    x = torch.randn((nb_nodes, in_channels))
    pos = torch.randn((nb_nodes, pos_dim))

    # Original output
    out_node_orig, out_pos_orig = model(x, pos, edge_index)

    # Apply random rotation and translation
    R = torch.randn((pos_dim, pos_dim))
    U, _, Vt = torch.linalg.svd(R)
    R_ortho = U @ Vt  # Ensure it's a proper rotation matrix
    t = torch.randn((1, pos_dim))

    pos_transformed = (pos @ R_ortho) + t

    # Output after transformation
    out_node_transformed, out_pos_transformed = model(x, pos_transformed, edge_index)

    # Transform the original output positions
    out_pos_orig_transformed = (out_pos_orig @ R_ortho) + t

    # Check if the transformed outputs match
    assert torch.allclose(out_pos_transformed, out_pos_orig_transformed, atol=1e-5), "Position outputs are not equivariant"
    assert torch.allclose(out_node_transformed, out_node_orig, atol=1e-5), "Node features should remain invariant"
    