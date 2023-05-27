import pytest
import torch

from torch_geometric.datasets import graph_generator
from torch_geometric.testing import withPackage
from torch_geometric.transforms import NodePropertySplit


@withPackage('networkx')
@pytest.mark.parametrize('property_name', [
    'popularity',
    'locality',
    'density',
])
def test_node_property_split(property_name):
    ratios = [0.3, 0.1, 0.1, 0.2, 0.3]

    transform = NodePropertySplit(property_name, ratios)
    assert str(transform) == f'NodePropertySplit({property_name})'

    data = graph_generator.ERGraph(num_nodes=100, edge_prob=0.4)()
    data = transform(data)

    node_ids = []
    for name, ratio in zip([
            'id_train_mask',
            'id_val_mask',
            'id_test_mask',
            'ood_val_mask',
            'ood_test_mask',
    ], ratios):
        assert data[name].dtype == torch.bool
        assert data[name].size() == (100, )
        assert int(data[name].sum()) == 100 * ratio
        node_ids.extend(data[name].nonzero().view(-1).tolist())

    # Check that masks are non-intersecting and cover all nodes:
    node_ids = torch.tensor(node_ids)
    assert node_ids.numel() == torch.unique(node_ids).numel() == 100
