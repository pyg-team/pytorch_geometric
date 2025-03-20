import pytest
import torch
from torch_geometric.testing import onlyNeighborSampler, MyGraphStore, MyFeatureStore
from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler.base import NodeSamplerInput, SamplerOutput
from torch_geometric.sampler.neighbor_sampler import NeighborSampler


def _init_sample_graph(hetero=False):
    """ Initializes the following graph.

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

    sample_x = torch.tensor([[0],[1],[2],[3]])

    if not hetero:
        sample_edge_indices = torch.tensor([
            [0,0,2],
            [1,2,3]
        ])
    else:
        sample_edge_indices = {
            ('person', 'works_with', 'person'): {"edge_index": torch.tensor([
                [0,2],
                [1,3]
            ])},
            ('person', 'leads', 'person'): {"edge_index": torch.tensor([
                [0],
                [2]
            ])}
        }
    return sample_x, sample_edge_indices

def _init_graph_to_sample(graph_dtype, hetero=False):
    sample_x, sample_edge_indices = _init_sample_graph(hetero)
    graph_to_sample = None
    if graph_dtype == 'data' and not hetero:
        graph_to_sample = Data(edge_index=sample_edge_indices, x=sample_x)
    elif graph_dtype == 'remote' and not hetero:
        graph_store = MyGraphStore()
        graph_store.put_edge_index(sample_edge_indices, edge_type=None, layout='coo', size=(4, 4))
        feature_store = MyFeatureStore()
        feature_store.put_tensor(sample_x, group_name='person', attr_name='x', index=None)
        graph_to_sample = (feature_store, graph_store)
    elif graph_dtype == 'data' and hetero:
        graph_to_sample = HeteroData(sample_edge_indices)
        graph_to_sample['person'].x = sample_x
    elif graph_dtype == 'remote' and hetero:
        graph_store = MyGraphStore()
        for edge_type, edge_index in sample_edge_indices.items():
            edge_index = edge_index["edge_index"]
            graph_store.put_edge_index(edge_index, edge_type=edge_type, layout='coo', size=(1+torch.max(edge_index, dim=1).values))
        feature_store = MyFeatureStore()
        feature_store.put_tensor(sample_x, group_name='person', attr_name='x', index=None)
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
    node_sampler_input = NodeSamplerInput(input_id=None, node=torch.tensor([1]))
    expected_output = SamplerOutput(
        node=torch.tensor([1, 0]),
        row=torch.tensor([1]),
        col=torch.tensor([0]),
        edge=torch.tensor([0]),
        batch=None,
        num_sampled_nodes=[1, 1],
        num_sampled_edges=[1],
        orig_row=None,
        orig_col=None,
        metadata=(None, None)
    )
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)
    assert str(sampler_output) == str(expected_output)


    # Sampling Alice should yield no edges
    node_sampler_input = NodeSamplerInput(input_id=None, node=torch.tensor([0]))
    expected_output = SamplerOutput(
        node=torch.tensor([0]),
        row=torch.empty(0, dtype=torch.int64),
        col=torch.empty(0, dtype=torch.int64),
        edge=torch.empty(0, dtype=torch.int64),
        batch=None,
        num_sampled_nodes=[1, 0],
        num_sampled_edges=[0],
        orig_row=None,
        orig_col=None,
        metadata=(None, None)
    )
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
    node_sampler_input = NodeSamplerInput(input_id=None, node=torch.tensor([1]), input_type="person")
    sampler = NeighborSampler(**sampler_kwargs)
    sampler_output = sampler.sample_from_nodes(node_sampler_input)
    assert set(sampler_output.node['person'].tolist()) == set([1, 0])
    assert sampler_output.row[('person', 'works_with', 'person')] == torch.tensor([1])
    assert sampler_output.row[('person', 'leads', 'person')].numel() == 0
    assert sampler_output.col[('person', 'works_with', 'person')] == torch.tensor([0])
    assert sampler_output.col[('person', 'leads', 'person')].numel() == 0
    assert sampler_output.edge[('person', 'works_with', 'person')] == torch.tensor([0])
    assert sampler_output.edge[('person', 'leads', 'person')].numel() == 0

    # Sampling Alice should yield no edges
    node_sampler_input = NodeSamplerInput(input_id=None, node=torch.tensor([0]), input_type="person")
    sampler_output = sampler.sample_from_nodes(node_sampler_input)
    assert set(sampler_output.node['person'].tolist()) == set([0])
    assert sampler_output.row[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.row[('person', 'leads', 'person')].numel() == 0
    assert sampler_output.col[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.col[('person', 'leads', 'person')].numel() == 0
    assert sampler_output.edge[('person', 'works_with', 'person')].numel() == 0
    assert sampler_output.edge[('person', 'leads', 'person')].numel() == 0
