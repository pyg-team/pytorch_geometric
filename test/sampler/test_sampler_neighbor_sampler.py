import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler.base import NodeSamplerInput, SamplerOutput
from torch_geometric.sampler.neighbor_sampler import (
    BidirectionalNeighborSampler,
    NeighborSampler,
)
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    onlyNeighborSampler,
)


def _init_sample_graph(hetero=False):
    """Initializes the following graph.

    #############                    ###########
    # Alice (0) # -> "works with" -> # Bob (1) #
    #############                    ###########
         |
         v
      "leads"
         |
         v
    #############                    ############
    # Carol (2) # -> "works with" -> # Dave (3) #
    #############                    ############
    """
    sample_attr = None
    sample_edge_attr = None
    sample_edge_indices = None
    if not hetero:
        sample_attr = torch.tensor([[0], [1], [2], [3]])
        sample_edge_attr = torch.tensor([[1], [2], [3]])
        sample_edge_indices = torch.tensor([[0, 0, 2], [1, 2, 3]])
    else:
        sample_attr = dict({
            "person": dict({"x": torch.tensor([[1], [2], [3]])}),
            "manager": dict({"x": torch.tensor([[0]])})
        })
        sample_edge_attr = dict({
            ('person', 'works_with', 'person'):
            dict({"edge_attr": torch.tensor([[3]])}),
            ('manager', 'leads', 'person'):
            dict({"edge_attr": torch.tensor([[1]])}),
            ('manager', 'works_with', 'person'):
            dict({"edge_attr": torch.tensor([[2]])})
        })
        sample_edge_indices = dict({
            ('person', 'works_with', 'person'):
            dict({"edge_index": torch.tensor([[1], [2]])}),
            ('manager', 'leads', 'person'):
            dict({"edge_index": torch.tensor([[0], [1]])}),
            ('manager', 'works_with', 'person'):
            dict({"edge_index": torch.tensor([[0], [0]])})
        })
    return sample_attr, sample_edge_attr, sample_edge_indices


def _init_graph_to_sample(graph_dtype, hetero=False, reverse=False):
    sample_attr, sample_edge_attr, sample_edge_indices = _init_sample_graph(
        hetero)
    if reverse:
        if not hetero:
            sample_edge_indices = sample_edge_indices.flip(0)
        else:
            reversed_edge_indices = dict()
            reversed_edge_attr = dict()
            for edge_type, edge_index in sample_edge_indices.items():
                edge_index = edge_index["edge_index"]
                edge_attr = sample_edge_attr[edge_type]["edge_attr"]
                flipped_edge_index = edge_index.flip(0)
                flipped_edge_type = (edge_type[2], edge_type[1], edge_type[0])
                reversed_edge_indices[flipped_edge_type] = dict(
                    {"edge_index": flipped_edge_index})
                reversed_edge_attr[flipped_edge_type] = dict(
                    {"edge_attr": edge_attr})
            sample_edge_indices = reversed_edge_indices
            sample_edge_attr = reversed_edge_attr
    graph_to_sample = None
    if graph_dtype == 'data' and not hetero:
        graph_to_sample = Data(edge_index=sample_edge_indices, x=sample_attr,
                               time=sample_attr.squeeze(-1),
                               edge_attr=sample_edge_attr.squeeze(-1))
    elif graph_dtype == 'remote' and not hetero:
        graph_store = MyGraphStore()
        graph_store.put_edge_index(sample_edge_indices, edge_type=None,
                                   layout='coo', is_sorted=True, size=(4, 4))
        feature_store = MyFeatureStore()
        feature_store.put_tensor(sample_attr, group_name='default',
                                 attr_name='x', index=None)
        # temporal node sampling on (fs, gs) needs 'time' attr
        feature_store.put_tensor(sample_attr.squeeze(-1), group_name='default',
                                 attr_name='time', index=None)
        feature_store.put_tensor(sample_edge_attr.squeeze(-1),
                                 group_name='default', attr_name='edge_attr',
                                 index=None)
        graph_to_sample = (feature_store, graph_store)
    elif graph_dtype == 'data' and hetero:
        graph_to_sample = HeteroData()
        for node_type, node_attr in sample_attr.items():
            graph_to_sample[node_type].x = node_attr['x']
            graph_to_sample[node_type].time = node_attr['x'].squeeze(-1)
        for edge_type in sample_edge_indices.keys():
            graph_to_sample[edge_type].edge_index = sample_edge_indices[
                edge_type]["edge_index"]
            graph_to_sample[edge_type].edge_attr = sample_edge_attr[edge_type][
                "edge_attr"].squeeze(-1)
    elif graph_dtype == 'remote' and hetero:
        graph_store = MyGraphStore()
        for edge_type, edge_index in sample_edge_indices.items():
            edge_index = edge_index["edge_index"]
            graph_store.put_edge_index(
                edge_index, edge_type=edge_type, layout='coo', is_sorted=True,
                size=(len(sample_attr[edge_type[0]]["x"]),
                      len(sample_attr[edge_type[2]]["x"])))
        feature_store = MyFeatureStore()
        for node_type, node_attr in sample_attr.items():
            feature_store.put_tensor(node_attr["x"], group_name=node_type,
                                     attr_name='x', index=None)
            # temporal node sampling on (fs, gs) needs 'time' attr
            feature_store.put_tensor(node_attr["x"].squeeze(-1),
                                     group_name=node_type, attr_name='time',
                                     index=None)
        for edge_type, edge_attr in sample_edge_attr.items():
            feature_store.put_tensor(edge_attr["edge_attr"].squeeze(-1),
                                     group_name=edge_type,
                                     attr_name='edge_attr', index=None)
        graph_to_sample = (feature_store, graph_store)
    return graph_to_sample


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_homogeneous_neighbor_sampler_basic(input_type):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=False)

    # NeighborSampler default parameters 1 node
    # disjoint = False
    # replace = False
    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
    }

    # Sampling from Bob should yield only Alice
    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([1]))
    expected_output = SamplerOutput(
        node=torch.tensor([1, 0]), row=torch.tensor([1]),
        col=torch.tensor([0]), edge=torch.tensor([0]), batch=None,
        num_sampled_nodes=[1, 1], num_sampled_edges=[1], orig_row=None,
        orig_col=None, metadata=(None, None))
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)
    assert str(sampler_output) == str(expected_output)

    # Sampling Alice should yield no edges
    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([0]))
    expected_output = SamplerOutput(node=torch.tensor([0]), row=torch.empty(
        0, dtype=torch.int64), col=torch.empty(0, dtype=torch.int64),
                                    edge=torch.empty(0, dtype=torch.int64),
                                    batch=None, num_sampled_nodes=[1, 0],
                                    num_sampled_edges=[0], orig_row=None,
                                    orig_col=None, metadata=(None, None))
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)
    assert str(sampler_output) == str(expected_output)


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_heterogeneous_neighbor_sampler_basic(input_type):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=True)

    # NeighborSampler default parameters 1 node
    # disjoint = False
    # replace = False
    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
    }

    # Sampling from Bob should yield only Alice
    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([0]),
                                          input_type="person")
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    assert sampler_output.node['person'].tolist() == [0]
    assert sampler_output.node['manager'].tolist() == [0]

    assert sampler_output.row[('manager', 'works_with',
                               'person')] == torch.tensor([0])
    assert sampler_output.row[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.row[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.col[('manager', 'works_with',
                               'person')] == torch.tensor([0])
    assert sampler_output.col[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.col[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.edge[('manager', 'works_with',
                                'person')] == torch.tensor([0])
    assert sampler_output.edge[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.edge[('person', 'works_with', 'person')].numel() == 0

    # Sampling Alice should yield no edges
    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([0]),
                                          input_type="manager")
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    assert sampler_output.node['manager'].tolist() == [0]
    assert sampler_output.node['person'].numel() == 0

    assert sampler_output.row[('manager', 'works_with', 'person')].numel() == 0
    assert sampler_output.row[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.row[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.col[('manager', 'works_with', 'person')].numel() == 0
    assert sampler_output.col[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.col[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.edge[('manager', 'works_with',
                                'person')].numel() == 0
    assert sampler_output.edge[('manager', 'leads', 'person')].numel() == 0
    assert sampler_output.edge[('person', 'works_with', 'person')].numel() == 0


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_homogeneous_neighbor_sampler_backwards(input_type):

    graph_to_sample = _init_graph_to_sample(input_type, hetero=False)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
    }

    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([2]))

    sampler = NeighborSampler(**sampler_kwargs)
    # This output should have Carol and Alice
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    backward_sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
        'sample_direction': 'backward',
    }
    backward_sampler = NeighborSampler(**backward_sampler_kwargs)
    # This output should have Carol and Dave
    backward_sampler_output = backward_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_graph_to_sample = _init_graph_to_sample(input_type, hetero=False,
                                                    reverse=True)

    reverse_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [1],
    }

    reverse_sampler = NeighborSampler(**reverse_sampler_kwargs)
    # This output should have Carol and Dave
    reverse_sampler_output = reverse_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_backward_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [1],
        'sample_direction': 'backward',
    }

    reverse_backward_sampler = NeighborSampler(
        **reverse_backward_sampler_kwargs)
    # This output should have Carol and Alice
    reverse_backward_sampler_output = \
        reverse_backward_sampler.sample_from_nodes(node_sampler_input)

    assert torch.equal(sampler_output.node,
                       reverse_backward_sampler_output.node)
    assert torch.equal(sampler_output.row, reverse_backward_sampler_output.col)
    assert torch.equal(sampler_output.col, reverse_backward_sampler_output.row)
    assert torch.equal(sampler_output.edge,
                       reverse_backward_sampler_output.edge)

    assert torch.equal(backward_sampler_output.node,
                       reverse_sampler_output.node)
    assert torch.equal(backward_sampler_output.row, reverse_sampler_output.col)
    assert torch.equal(backward_sampler_output.col, reverse_sampler_output.row)
    assert torch.equal(backward_sampler_output.edge,
                       reverse_sampler_output.edge)


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_homogeneous_neighbor_sampler_weighted_backwards(input_type):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=False)
    reverse_graph_to_sample = _init_graph_to_sample(input_type, hetero=False,
                                                    reverse=True)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1, 1],
        'weight_attr': 'weight',
        'sample_direction': 'backward'
    }
    reverse_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [1, 1],
        'weight_attr': 'weight',
        'sample_direction': 'forward',
    }

    if input_type == 'remote':
        with pytest.raises(NotImplementedError):
            NeighborSampler(**sampler_kwargs)
        return

    graph_to_sample['weight'] = torch.tensor([1.0, 0.0, 1.0])
    reverse_graph_to_sample['weight'] = torch.tensor([1.0, 0.0, 1.0])

    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([0]))

    # Sampling from Alice should yield Bob
    backward_sampler = NeighborSampler(**sampler_kwargs)
    backward_sampler_output = backward_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_sampler = NeighborSampler(**reverse_sampler_kwargs)
    reverse_sampler_output = reverse_sampler.sample_from_nodes(
        node_sampler_input)

    assert torch.equal(backward_sampler_output.node,
                       reverse_sampler_output.node)
    assert torch.equal(backward_sampler_output.row, reverse_sampler_output.col)
    assert torch.equal(backward_sampler_output.col, reverse_sampler_output.row)
    assert torch.equal(backward_sampler_output.edge,
                       reverse_sampler_output.edge)

    graph_to_sample['weight'] = torch.tensor([0.0, 1.0, 1.0])
    reverse_graph_to_sample['weight'] = torch.tensor([0.0, 1.0, 1.0])

    # Sampling from Alice should yield Carol and Dave
    backward_sampler = NeighborSampler(**sampler_kwargs)
    backward_sampler_output = backward_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_sampler = NeighborSampler(**reverse_sampler_kwargs)
    reverse_sampler_output = reverse_sampler.sample_from_nodes(
        node_sampler_input)

    assert torch.equal(backward_sampler_output.node,
                       reverse_sampler_output.node)
    assert torch.equal(backward_sampler_output.row, reverse_sampler_output.col)
    assert torch.equal(backward_sampler_output.col, reverse_sampler_output.row)
    assert torch.equal(backward_sampler_output.edge,
                       reverse_sampler_output.edge)


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
@pytest.mark.parametrize('time_attr', ['time', 'edge_attr'])
def test_homogeneous_neighbor_sampler_temporal_backwards(
        input_type, time_attr):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=False)
    reverse_graph_to_sample = _init_graph_to_sample(input_type, hetero=False,
                                                    reverse=True)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [2, 2],
        'time_attr': time_attr,
    }
    reverse_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [2, 2],
        'time_attr': time_attr,
    }

    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([1]),
                                          time=torch.tensor([1]))
    reverse_node_sampler_input = NodeSamplerInput(input_id=None,
                                                  node=torch.tensor([0]),
                                                  time=torch.tensor([1]))

    # sampling from Dave should yield Carol, Alice
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    reverse_sampler = NeighborSampler(**reverse_sampler_kwargs)
    reverse_sampler_output = reverse_sampler.sample_from_nodes(
        reverse_node_sampler_input)

    assert torch.equal(sampler_output.node, torch.tensor([1, 0]))
    assert torch.equal(reverse_sampler_output.node, torch.tensor([0, 1]))
    """
    TODO (zaristei) Negative cases for temporal sampling,
    then verify that the output is correct for backwards sampling.
    """
    pytest.skip("still TODO")


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_heterogeneous_neighbor_sampler_backwards(input_type):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=True)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
    }

    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([1]),
                                          input_type="person")

    sampler = NeighborSampler(**sampler_kwargs)
    # This output should have Carol and Alice
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    backward_sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
        'sample_direction': 'backward',
    }
    backward_sampler = NeighborSampler(**backward_sampler_kwargs)
    # This output should have Carol and Dave
    backward_sampler_output = backward_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_graph_to_sample = _init_graph_to_sample(input_type, hetero=True,
                                                    reverse=True)

    reverse_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [1],
    }

    reverse_sampler = NeighborSampler(**reverse_sampler_kwargs)
    # This output should have Carol and Dave
    reverse_sampler_output = reverse_sampler.sample_from_nodes(
        node_sampler_input)

    reverse_backward_sampler_kwargs = {
        'data': reverse_graph_to_sample,
        'num_neighbors': [1],
        'sample_direction': 'backward',
    }

    reverse_backward_sampler = NeighborSampler(
        **reverse_backward_sampler_kwargs)
    # This output should have Carol and Alice
    reverse_backward_sampler_output = \
        reverse_backward_sampler.sample_from_nodes(node_sampler_input)

    def reverse_key(key):
        return (key[2], key[1], key[0])

    assert sampler_output.node.keys(
    ) == reverse_backward_sampler_output.node.keys()
    assert reverse_sampler_output.node.keys(
    ) == backward_sampler_output.node.keys()
    for key in sampler_output.node.keys():
        assert torch.equal(sampler_output.node[key],
                           reverse_backward_sampler_output.node[key])
    for key in reverse_sampler_output.node.keys():
        assert torch.equal(reverse_sampler_output.node[key],
                           backward_sampler_output.node[key])

    assert len(sampler_output.row.keys()) == len(
        reverse_backward_sampler_output.row.keys())
    for key in sampler_output.row.keys():
        assert reverse_key(key) in reverse_backward_sampler_output.row.keys()

        assert torch.equal(
            sampler_output.row[key],
            reverse_backward_sampler_output.col[reverse_key(key)])
        assert torch.equal(
            sampler_output.col[key],
            reverse_backward_sampler_output.row[reverse_key(key)])
        assert torch.equal(
            sampler_output.edge[key],
            reverse_backward_sampler_output.edge[reverse_key(key)])

    assert len(reverse_sampler_output.row.keys()) == len(
        backward_sampler_output.row.keys())
    for key in reverse_sampler_output.row.keys():
        assert reverse_key(key) in backward_sampler_output.row.keys()

        assert torch.equal(reverse_sampler_output.row[key],
                           backward_sampler_output.col[reverse_key(key)])
        assert torch.equal(reverse_sampler_output.col[key],
                           backward_sampler_output.row[reverse_key(key)])
        assert torch.equal(reverse_sampler_output.edge[key],
                           backward_sampler_output.edge[reverse_key(key)])


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_bidirectional_neighbor_sampler(input_type):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=False)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
    }

    node_sampler_input = NodeSamplerInput(input_id=None,
                                          node=torch.tensor([2]))
    sampler = BidirectionalNeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)

    expected_output = SamplerOutput(
        # Union between forward and backward nodes
        node=torch.tensor([2, 0, 3]),
        # Reindexed to be relative to new nodes field
        row=torch.tensor([1, 0]),
        # Reindexed to be relative to new nodes field
        col=torch.tensor([0, 2]),
        # Union between forward and backward edges
        edge=torch.tensor([1, 2]),
        # Will be part of node uid if disjoint=True
        batch=None,
        # nodes are only counted on their first sample
        num_sampled_nodes=[1, 1, 0, 1],
        # edges are only counted on their first sample
        num_sampled_edges=[1, 1],
        # Will be used as edge uid if bidirectional=True with
        # keep_orig_edges=True
        orig_row=None,
        # Will be used as edge uid if bidirectional=True with
        # keep_orig_edges=True
        orig_col=None,
        # simple concat of forward and backward metadata
        metadata=(None, None))
    assert str(sampler_output) == str(expected_output)

    adv_sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [2, 2, 2, 2],
    }

    adv_sampler_input = NodeSamplerInput(input_id=None,
                                         node=torch.tensor([1, 3]))

    adv_sampler = BidirectionalNeighborSampler(**adv_sampler_kwargs)
    adv_sampler_output = adv_sampler.sample_from_nodes(adv_sampler_input)

    adv_expected_output = SamplerOutput(
        node=torch.tensor([1, 3, 0, 2]),
        row=torch.tensor([2, 3, 2]),
        col=torch.tensor([0, 1, 3]),
        edge=torch.tensor([0, 2, 1]),
        batch=None,
        # 8 _sample calls total, each have 2 num_sampled_nodes slots
        num_sampled_nodes=[2, 2] + [0] * 14,
        num_sampled_edges=[2, 0, 1, 0, 0, 0, 0, 0],
        orig_row=None,
        orig_col=None,
        metadata=(None, None))
    assert str(adv_sampler_output) == str(adv_expected_output)

    adv_sampler_kwargs['disjoint'] = True

    adv_sampler_disjoint = BidirectionalNeighborSampler(**adv_sampler_kwargs)
    adv_sampler_disjoint_output = adv_sampler_disjoint.sample_from_nodes(
        adv_sampler_input)

    adv_expected_disjoint_output = SamplerOutput(
        node=torch.tensor([1, 3, 0, 2, 0, 2, 1, 3]),
        row=torch.tensor([2, 3, 4, 2, 4, 5]),
        col=torch.tensor([0, 1, 3, 5, 6, 7]),
        edge=torch.tensor([0, 2, 1, 1, 0, 2]),
        batch=torch.tensor([0, 1, 0, 1, 1, 0, 1, 0]),
        num_sampled_nodes=[
            # First forward iteration:
            # (Bob, Dave) -> (Alice, Carol)
            2,
            2,
            # First backward iteration:
            # (Bob (seen), Dave (seen)) -> (None, None)
            0,
            0,
            # Second forward iteration:
            # (Alice (seen), Carol (seen)) -> (None, Alice)
            0,
            1,
            # Second backward iteration:
            # (Alice (seen), Carol (seen)) ->
            #   (Bob (seen) and Carol, Alice and Dave (seen))
            0,
            1,
            # Third forward iteration:
            # (Carol (seen), Alice (seen)) -> (Alice (seen), None)
            0,
            0,
            # Third backward iteration:
            # (Alice (seen), Carol (seen)) -> (Bob and Carol (seen), Dave)
            0,
            2,
            # Fourth forward iteration:
            # (Bob (seen), Dave (seen)) -> (Alice (seen), Carol (seen))
            0,
            0,
            # Fourth backward iteration:
            # (Bob (seen), Dave (seen)) -> (None, None)
            0,
            0
        ],
        num_sampled_edges=[2, 0, 1, 1, 0, 2, 0, 0],
        orig_row=None,
        orig_col=None,
        metadata=(None, None))
    assert str(adv_sampler_disjoint_output) == str(
        adv_expected_disjoint_output)


@pytest.mark.skip(
    reason="BidirectionalSampler not implemented yet for heterogeneous graphs."
)
@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
def test_bidirectional_neighbor_sampler_hetero(input_type):
    raise NotImplementedError


@onlyNeighborSampler
@pytest.mark.parametrize('input_type', ['data', 'remote'])
@pytest.mark.parametrize('hetero', [False, True])
def test_neighbor_sampler_backwards_not_supported(input_type, hetero):
    graph_to_sample = _init_graph_to_sample(input_type, hetero=hetero)

    sampler_kwargs = {
        'data': graph_to_sample,
        'num_neighbors': [1],
        'sample_direction': 'backward',
        'time_attr': 'time'
    }

    with pytest.raises(NotImplementedError):
        NeighborSampler(**sampler_kwargs)
