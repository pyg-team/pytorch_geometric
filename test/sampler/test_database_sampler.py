"""Unit tests for DatabaseSampler — no real database required."""
import pytest
import torch

from torch_geometric.sampler.base import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.sampler.database_sampler import DatabaseSampler
from torch_geometric.testing import FakeDatabaseGraphStore, FakeDatabaseSampler


def _make_graph_store(nodes, edges):
    """Return FakeDatabaseGraphStore FAKE_NODE_QUERY returns nodes+edges."""
    record = {"nodes": nodes, "edges": edges}
    return FakeDatabaseGraphStore(records={
        "FAKE_NODE_QUERY": record,
        "FAKE_EDGE_QUERY": record
    })


def _node_input(nids):
    return NodeSamplerInput(input_id=None,
                            node=torch.tensor(nids, dtype=torch.long))


def test_sample_from_nodes_returns_sampler_output():
    gs = _make_graph_store(nodes=[0, 1, 2], edges=[[0, 1], [1, 2]])
    sampler = FakeDatabaseSampler(gs)
    out = sampler.sample_from_nodes(_node_input([0]))
    assert isinstance(out, SamplerOutput)


def test_sample_from_nodes_passes_correct_query():
    gs = _make_graph_store(nodes=[0, 1], edges=[[0, 1]])
    sampler = FakeDatabaseSampler(gs)
    sampler.sample_from_nodes(_node_input([0]))
    executed_query, params = gs.executed_queries[-1]
    assert executed_query == "FAKE_NODE_QUERY"
    assert params == {"seed_ids": [0]}


def test_sample_from_nodes_seed_ids_in_params():
    gs = _make_graph_store(nodes=[3, 4, 5], edges=[[3, 4]])
    sampler = FakeDatabaseSampler(gs)
    sampler.sample_from_nodes(_node_input([3, 5]))
    _, params = gs.executed_queries[-1]
    assert set(params["seed_ids"]) == {3, 5}


def test_sample_from_nodes_metadata_roundtrip():
    gs = _make_graph_store(nodes=[1], edges=[])
    sampler = FakeDatabaseSampler(gs)
    seeds = torch.tensor([1], dtype=torch.long)
    out = sampler.sample_from_nodes(_node_input([1]))
    stored_seeds, seed_time = out.metadata
    assert torch.equal(stored_seeds, seeds)
    assert seed_time is None


def test_sample_from_nodes_missing_query_raises():
    """Subclass that returns None for node query raises ValueError."""
    class NoQuerySampler(DatabaseSampler):
        def _build_node_sampling_query(self):
            return None

        def _build_edge_sampling_query(self):
            return None

        def _build_node_query_params(self, seeds, **kwargs):
            return {}

    gs = _make_graph_store([], [])
    sampler = NoQuerySampler(gs)
    with pytest.raises(ValueError, match="Node sampling query is not built"):
        sampler.sample_from_nodes(_node_input([0]))


def test_sample_from_edges_seeds_are_unique_union():
    gs = _make_graph_store(nodes=[0, 1, 2], edges=[[0, 1]])
    sampler = FakeDatabaseSampler(gs)
    edge_input = EdgeSamplerInput(
        input_id=None,
        row=torch.tensor([0, 1], dtype=torch.long),
        col=torch.tensor([1, 0], dtype=torch.long),
    )
    sampler.sample_from_edges(edge_input)
    _, params = gs.executed_queries[-1]
    # unique([0,1,1,0]) = {0,1}
    assert set(params["seed_ids"]) == {0, 1}


def test_sample_from_edges_neg_sampling_not_supported():
    gs = _make_graph_store(nodes=[0, 1], edges=[[0, 1]])
    sampler = FakeDatabaseSampler(gs)
    edge_input = EdgeSamplerInput(
        input_id=None,
        row=torch.tensor([0], dtype=torch.long),
        col=torch.tensor([1], dtype=torch.long),
    )
    with pytest.raises(NotImplementedError, match="negative sampling"):
        sampler.sample_from_edges(edge_input, neg_sampling=object())


def test_sample_from_edges_missing_query_raises():
    class NoEdgeSampler(DatabaseSampler):
        def _build_node_sampling_query(self):
            return None

        def _build_edge_sampling_query(self):
            return None

        def _build_edge_query_params(self, seeds, **kwargs):
            return {}

    gs = _make_graph_store([], [])
    sampler = NoEdgeSampler(gs)
    edge_input = EdgeSamplerInput(
        input_id=None,
        row=torch.tensor([0], dtype=torch.long),
        col=torch.tensor([1], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="Edge sampling query is not built"):
        sampler.sample_from_edges(edge_input)


def test_is_hetero_property_matches_constructor_arg():
    gs = FakeDatabaseGraphStore()
    assert not FakeDatabaseSampler(gs).is_hetero
    assert not FakeDatabaseSampler(gs, is_hetero=False).is_hetero
    assert FakeDatabaseSampler(gs, is_hetero=True).is_hetero


def test_hetero_output_when_is_hetero_true():
    """Empty hetero result produces HeteroSamplerOutput (no crash)."""
    gs = FakeDatabaseGraphStore(records={
        "FAKE_NODE_QUERY": None,
        "FAKE_EDGE_QUERY": None
    })
    sampler = FakeDatabaseSampler(gs, is_hetero=True)
    out = sampler.sample_from_nodes(_node_input([0]))
    assert isinstance(out, HeteroSamplerOutput)


def test_hetero_missing_input_type_spliced_into_node():
    """When seed's input_type absent from decoded node dict,
    seeds spliced in.
    """
    gs = FakeDatabaseGraphStore(records={
        "FAKE_NODE_QUERY": None,
        "FAKE_EDGE_QUERY": None
    })
    sampler = FakeDatabaseSampler(gs, is_hetero=True)

    seeds = torch.tensor([5, 6], dtype=torch.long)
    inp = NodeSamplerInput(input_id=None, node=seeds, input_type="paper")
    out = sampler.sample_from_nodes(inp)
    assert "paper" in out.node
    assert torch.equal(out.node["paper"], seeds)


def test_empty_result_homogeneous():
    seeds = torch.tensor([10, 20], dtype=torch.long)
    node, row, col = DatabaseSampler._empty_result(seeds, is_hetero=False)
    assert torch.equal(node, seeds)
    assert row.numel() == 0
    assert col.numel() == 0


def test_empty_result_heterogeneous():
    seeds = torch.tensor([1], dtype=torch.long)
    node, row, col = DatabaseSampler._empty_result(seeds, is_hetero=True)
    assert node == row == col == {}
