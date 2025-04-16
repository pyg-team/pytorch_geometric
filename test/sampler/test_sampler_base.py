import pytest
import torch

from torch_geometric.sampler.base import (
    HeteroSamplerOutput,
    NumNeighbors,
    SamplerOutput,
)
from torch_geometric.sampler.utils import local_to_global_node_idx, global_to_local_node_idx
from torch_geometric.testing import get_random_edge_index
from torch_geometric.utils import is_undirected


def test_homogeneous_num_neighbors():
    with pytest.raises(ValueError, match="'default' must be set to 'None'"):
        num_neighbors = NumNeighbors([25, 10], default=[-1, -1])

    num_neighbors = NumNeighbors([25, 10])
    assert str(num_neighbors) == 'NumNeighbors(values=[25, 10], default=None)'

    assert num_neighbors.get_values() == [25, 10]
    assert num_neighbors.__dict__['_values'] == [25, 10]
    assert num_neighbors.get_values() == [25, 10]  # Test caching.

    assert num_neighbors.get_mapped_values() == [25, 10]
    assert num_neighbors.__dict__['_mapped_values'] == [25, 10]
    assert num_neighbors.get_mapped_values() == [25, 10]  # Test caching.

    assert num_neighbors.num_hops == 2
    assert num_neighbors.__dict__['_num_hops'] == 2
    assert num_neighbors.num_hops == 2  # Test caching.


def test_heterogeneous_num_neighbors_list():
    num_neighbors = NumNeighbors([25, 10])

    values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])
    assert values == {('A', 'B'): [25, 10], ('B', 'A'): [25, 10]}

    values = num_neighbors.get_mapped_values([('A', 'B'), ('B', 'A')])
    assert values == {'A__to__B': [25, 10], 'B__to__A': [25, 10]}

    assert num_neighbors.num_hops == 2


def test_heterogeneous_num_neighbors_dict_and_default():
    num_neighbors = NumNeighbors({('A', 'B'): [25, 10]}, default=[-1])
    with pytest.raises(ValueError, match="hops must be the same across all"):
        values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])

    num_neighbors = NumNeighbors({('A', 'B'): [25, 10]}, default=[-1, -1])

    with pytest.raises(ValueError, match="Not all edge types"):
        num_neighbors.get_values([('A', 'C'), ('B', 'A')])

    values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])
    assert values == {('A', 'B'): [25, 10], ('B', 'A'): [-1, -1]}

    values = num_neighbors.get_mapped_values([('A', 'B'), ('B', 'A')])
    assert values == {'A__to__B': [25, 10], 'B__to__A': [-1, -1]}

    assert num_neighbors.num_hops == 2


def test_heterogeneous_num_neighbors_empty_dict():
    num_neighbors = NumNeighbors({}, default=[25, 10])

    values = num_neighbors.get_values([('A', 'B'), ('B', 'A')])
    assert values == {('A', 'B'): [25, 10], ('B', 'A'): [25, 10]}

    values = num_neighbors.get_mapped_values([('A', 'B'), ('B', 'A')])
    assert values == {'A__to__B': [25, 10], 'B__to__A': [25, 10]}

    assert num_neighbors.num_hops == 2


def test_homogeneous_to_bidirectional():
    edge_index = get_random_edge_index(10, 10, num_edges=20)

    obj = SamplerOutput(
        node=torch.arange(10),
        row=edge_index[0],
        col=edge_index[0],
        edge=torch.arange(edge_index.size(1)),
    ).to_bidirectional()

    assert is_undirected(torch.stack([obj.row, obj.col], dim=0))


def test_heterogeneous_to_bidirectional():
    edge_index1 = get_random_edge_index(10, 5, num_edges=20)
    edge_index2 = get_random_edge_index(5, 10, num_edges=20)
    edge_index3 = get_random_edge_index(10, 10, num_edges=20)

    obj = HeteroSamplerOutput(
        node={
            'v1': torch.arange(10),
            'v2': torch.arange(5)
        },
        row={
            ('v1', 'to', 'v2'): edge_index1[0],
            ('v2', 'rev_to', 'v1'): edge_index2[0],
            ('v1', 'to', 'v1'): edge_index3[0],
        },
        col={
            ('v1', 'to', 'v2'): edge_index1[1],
            ('v2', 'rev_to', 'v1'): edge_index2[1],
            ('v1', 'to', 'v1'): edge_index3[1],
        },
        edge={},
    ).to_bidirectional()

    assert torch.equal(
        obj.row['v1', 'to', 'v2'].sort().values,
        obj.col['v2', 'rev_to', 'v1'].sort().values,
    )
    assert torch.equal(
        obj.col['v1', 'to', 'v2'].sort().values,
        obj.row['v2', 'rev_to', 'v1'].sort().values,
    )
    assert is_undirected(
        torch.stack([obj.row['v1', 'to', 'v1'], obj.col['v1', 'to', 'v1']], 0))

def test_homogeneous_sampler_output_global_fields():
    output = SamplerOutput(
        node=torch.tensor([0, 2, 3]),
        row=torch.tensor([0, 1]),
        col=torch.tensor([1, 2]),
        edge=torch.tensor([1, 2]),
        batch=torch.tensor([0, 0, 0]),
        num_sampled_nodes=[1, 1, 1],
        num_sampled_edges=[1, 1],
        orig_row=None,
        orig_col=None,
        metadata=(None, None),
    )

    local_values = []
    global_values = []

    global_row, global_col = output.global_row, output.global_col
    assert torch.equal(global_row, torch.tensor([0, 2]))
    assert torch.equal(global_col, torch.tensor([2, 3]))
    local_values.append(output.row)
    local_values.append(output.col)
    global_values.append(global_row)
    global_values.append(global_col)

    seed_node = output.seed_node
    assert torch.equal(seed_node, torch.tensor([0, 0, 0]))
    local_values.append(output.batch)
    global_values.append(seed_node)

    output_bidirectional = output.to_bidirectional(keep_orig_edges=True)
    global_bidir_row, global_bidir_col = output_bidirectional.global_row, output_bidirectional.global_col
    assert torch.equal(global_bidir_row, torch.tensor([2, 0, 3, 2]))
    assert torch.equal(global_bidir_col, torch.tensor([0, 2, 2, 3]))
    local_values.append(output_bidirectional.row)
    local_values.append(output_bidirectional.col)
    global_values.append(global_bidir_row)
    global_values.append(global_bidir_col)

    assert torch.equal(output.global_row, output_bidirectional.global_orig_row)
    assert torch.equal(output.global_col, output_bidirectional.global_orig_col)
    
    # Make sure reverse mapping is correct
    for local_value, global_value in zip(local_values, global_values):
        assert torch.equal(global_to_local_node_idx(output.node, global_value), local_value)

def test_heterogeneous_sampler_output_global_fields():
    def _tensor_dict_equal(dict1, dict2):
        is_equal = True
        is_equal &= dict1.keys() == dict2.keys()
        for key in dict1.keys():
            is_equal &= torch.equal(dict1[key], dict2[key])
        return is_equal

    output = HeteroSamplerOutput(
        node={"person": torch.tensor([0, 2, 3])},
        row={("person", "works_with", "person"): torch.tensor([1]), ("person", "leads", "person"): torch.tensor([0])},
        col={("person", "works_with", "person"): torch.tensor([2]), ("person", "leads", "person"): torch.tensor([1])},
        edge={("person", "works_with", "person"): torch.tensor([1]), ("person", "leads", "person"): torch.tensor([0])},
        batch={"person": torch.tensor([0, 0, 0])},
        num_sampled_nodes={"person": torch.tensor([1, 1, 1])},
        num_sampled_edges={("person", "works_with", "person"): torch.tensor([1]), ("person", "leads", "person"): torch.tensor([1])},
        orig_row=None,
        orig_col=None,
        metadata=(None, None),
    )

    local_values = []
    global_values = []

    global_row, global_col = output.global_row, output.global_col
    assert _tensor_dict_equal(global_row, {("person", "works_with", "person"): torch.tensor([2]), ("person", "leads", "person"): torch.tensor([0])})
    assert _tensor_dict_equal(global_col, {("person", "works_with", "person"): torch.tensor([3]), ("person", "leads", "person"): torch.tensor([2])})

    local_row_dict = {k: global_to_local_node_idx(output.node[k[0]], v) for k, v in global_row.items()}
    assert _tensor_dict_equal(local_row_dict, output.row)

    local_col_dict = {k: global_to_local_node_idx(output.node[k[2]], v) for k, v in global_col.items()}
    assert _tensor_dict_equal(local_col_dict, output.col)

    seed_node = output.seed_node
    assert _tensor_dict_equal(seed_node, {"person": torch.tensor([0, 0, 0])})

    local_batch_dict = {k: global_to_local_node_idx(output.node[k], v) for k, v in seed_node.items()}
    assert _tensor_dict_equal(local_batch_dict, output.batch)

    output_bidirectional = output.to_bidirectional(keep_orig_edges=True)
    global_bidir_row, global_bidir_col = output_bidirectional.global_row, output_bidirectional.global_col
    assert _tensor_dict_equal(global_bidir_row, {("person", "works_with", "person"): torch.tensor([3, 2]), ("person", "leads", "person"): torch.tensor([2, 0])})
    assert _tensor_dict_equal(global_bidir_col, {("person", "works_with", "person"): torch.tensor([2, 3]), ("person", "leads", "person"): torch.tensor([0, 2])})

    local_bidir_row_dict = {k: global_to_local_node_idx(output_bidirectional.node[k[0]], v) for k, v in global_bidir_row.items()}
    assert _tensor_dict_equal(local_bidir_row_dict, output_bidirectional.row)

    local_bidir_col_dict = {k: global_to_local_node_idx(output_bidirectional.node[k[2]], v) for k, v in global_bidir_col.items()}
    assert _tensor_dict_equal(local_bidir_col_dict, output_bidirectional.col)

    assert _tensor_dict_equal(output.global_row, output_bidirectional.global_orig_row)
    assert _tensor_dict_equal(output.global_col, output_bidirectional.global_orig_col)