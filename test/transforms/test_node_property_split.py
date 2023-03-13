from functools import reduce
from itertools import combinations

import numpy as np
import torch

from torch_geometric.datasets import graph_generator
from torch_geometric.testing import withPackage
from torch_geometric.transforms import NodePropertySplit


@withPackage('networkx')
def test_node_property_split():
    property_names = ["popularity", "locality", "density"]

    num_nodes = 1000
    edge_prob = 0.4
    data = graph_generator.ERGraph(num_nodes, edge_prob)()
    part_ratios = [0.3, 0.1, 0.1, 0.2, 0.3]

    for property_name in property_names:
        transform = NodePropertySplit(property_name, part_ratios)
        assert (str(transform) ==
                f"NodePropertySplit(property_name={property_name})")
        data = transform(data)

        for store in data.node_stores:
            assert "split_masks" in store
            split_masks = store.split_masks
            _test_split_masks(split_masks, num_nodes, part_ratios)


def test_mask_nodes_by_property():
    num_nodes = 1000
    property_values = np.random.uniform(size=num_nodes)
    part_ratios = [0.3, 0.1, 0.1, 0.2, 0.3]

    split_masks = NodePropertySplit.mask_nodes_by_property(
        property_values, part_ratios)
    _test_split_masks(split_masks, num_nodes, part_ratios)


def _test_split_masks(split_masks, num_nodes, part_ratios):
    mask_names = [
        "in_train_mask",
        "in_valid_mask",
        "in_test_mask",
        "out_valid_mask",
        "out_test_mask",
    ]

    for mask_name, part_ratio in zip(mask_names, part_ratios):
        assert mask_name in split_masks
        split_mask = split_masks[mask_name]
        assert len(split_mask) == num_nodes

        # check that each mask covers necessary number of nodes
        assert (torch.sum(split_mask.long()).item() == int(num_nodes *
                                                           part_ratio))

    # check that masks are non-intersecting
    for mask_pair in combinations(split_masks.values(), 2):
        assert not torch.any(torch.logical_and(*mask_pair)).item()

    # check that masks cover all nodes
    assert torch.all(reduce(torch.logical_or, split_masks.values())).item()
