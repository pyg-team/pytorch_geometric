import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam

import torch_geometric.typing
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import get_random_edge_index
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, to_torch_csc_tensor


def test_hgt_conv_same_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 16),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(16, 16, metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 16, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 16)
    assert out_dict1['paper'].size() == (6, 16)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)

    # TODO: Test JIT functionality. We need to wait on this one until PyTorch
    # allows indexing `ParameterDict` mappings :(


def test_hgt_conv_different_dimensions():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(in_channels={
        'author': 16,
        'paper': 32
    }, out_channels=32, metadata=metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)


def test_hgt_conv_lazy():
    x_dict = {
        'author': torch.randn(4, 16),
        'paper': torch.randn(6, 32),
    }
    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    edge_index_dict = {
        ('author', 'writes', 'paper'): edge_index,
        ('paper', 'written_by', 'author'): edge_index.flip([0]),
    }

    adj_t_dict1 = {}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        adj_t_dict1[edge_type] = to_torch_csc_tensor(
            edge_index,
            size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)),
        ).t()

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = HGTConv(-1, 32, metadata, heads=2)
    assert str(conv) == 'HGTConv(-1, 32, heads=2)'
    out_dict1 = conv(x_dict, edge_index_dict)
    assert len(out_dict1) == 2
    assert out_dict1['author'].size() == (4, 32)
    assert out_dict1['paper'].size() == (6, 32)

    out_dict2 = conv(x_dict, adj_t_dict1)
    assert len(out_dict1) == len(out_dict2)
    for key in out_dict1.keys():
        assert torch.allclose(out_dict1[key], out_dict2[key], atol=1e-6)

    if False and torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict2 = {}
        for edge_type, edge_index in edge_index_dict.items():
            adj_t_dict2[edge_type] = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=adj_t_dict1[edge_type].size()[::-1],
            ).t()
        out_dict3 = conv(x_dict, adj_t_dict2)
        assert len(out_dict1) == len(out_dict3)
        for key in out_dict1.keys():
            assert torch.allclose(out_dict1[key], out_dict3[key], atol=1e-6)


def test_hgt_conv_out_of_place():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)

    edge_index = coalesce(get_random_edge_index(4, 6, num_edges=20))

    data['author', 'paper'].edge_index = edge_index
    data['paper', 'author'].edge_index = edge_index.flip([0])

    conv = HGTConv(-1, 64, data.metadata(), heads=1)

    x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
    assert x_dict['author'].size() == (4, 16)
    assert x_dict['paper'].size() == (6, 32)

    _ = conv(x_dict, edge_index_dict)

    assert x_dict['author'].size() == (4, 16)
    assert x_dict['paper'].size() == (6, 32)


def test_hgt_conv_missing_dst_node_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    data['author', 'paper'].edge_index = get_random_edge_index(4, 6, 20)
    data['paper', 'author'].edge_index = get_random_edge_index(6, 4, 20)
    data['university', 'author'].edge_index = get_random_edge_index(10, 4, 10)

    conv = HGTConv(-1, 64, data.metadata(), heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['author'].size() == (4, 64)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


def test_hgt_conv_missing_input_node_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['author', 'writes',
         'paper'].edge_index = get_random_edge_index(4, 6, 20)

    # Some nodes from metadata are missing in data.
    # This might happen while using NeighborLoader.
    metadata = (['author', 'paper',
                 'university'], [('author', 'writes', 'paper')])
    conv = HGTConv(-1, 64, metadata, heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


def test_hgt_conv_missing_edge_type():
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    data['author', 'writes',
         'paper'].edge_index = get_random_edge_index(4, 6, 20)

    metadata = (['author', 'paper',
                 'university'], [('author', 'writes', 'paper'),
                                 ('university', 'employs', 'author')])
    conv = HGTConv(-1, 64, metadata, heads=1)

    out_dict = conv(data.x_dict, data.edge_index_dict)
    assert out_dict['author'].size() == (4, 64)
    assert out_dict['paper'].size() == (6, 64)
    assert 'university' not in out_dict


def test_rte_on_vs_off():
    """Test whether RTE has an effect when enabled vs. disabled."""
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)
    awp_edge.time_diff = torch.randint(0, 100, (awp_edge.num_edges, ))

    uea_edge = data['university', 'employs', 'author']
    uea_edge.edge_index = get_random_edge_index(10, 4, 15)
    uea_edge.time_diff = torch.zeros(uea_edge.num_edges, dtype=torch.long)

    metadata = data.metadata()

    torch.manual_seed(42)
    conv_with_rte = HGTConv(-1, 64, metadata, heads=2, use_RTE=True)

    torch.manual_seed(42)
    conv_without_rte = HGTConv(-1, 64, metadata, heads=2, use_RTE=False)

    out_dict_with_rte = conv_with_rte(data.x_dict, data.edge_index_dict,
                                      data.time_diff_dict)
    out_dict_without_rte = conv_without_rte(data.x_dict, data.edge_index_dict)

    author_out_with_rte = out_dict_with_rte['author']
    author_out_without_rte = out_dict_without_rte['author']

    assert not torch.allclose(author_out_with_rte, author_out_without_rte)


def test_rte_sensitivity_to_time_values():
    """Tests the sensitivity of the HGTConv layer to its temporal inputs.

    This test ensures that when the `edge_time_diff_dict` values are
    modified, the output embeddings of the HGTConv layer with RTE enabled
    also change.
    """
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)
    awp_edge.time_diff = torch.randint(0, 100, (awp_edge.num_edges, ))

    uae_edge = data['university', 'employs', 'author']
    uae_edge.edge_index = get_random_edge_index(10, 4, 15)
    uae_edge.time_diff = torch.zeros(uae_edge.num_edges, dtype=torch.long)

    metadata = data.metadata()
    torch.manual_seed(42)
    conv = HGTConv(-1, 64, metadata, heads=2, use_RTE=True)

    out_dict_1 = conv(data.x_dict, data.edge_index_dict, data.time_diff_dict)
    author_out_1 = out_dict_1['author']

    data_alt_time = data.clone()
    for edge_type in data.edge_types:
        if 'time_diff' in data[edge_type]:
            data_alt_time[
                edge_type].time_diff = data[edge_type].time_diff + 100

    out_dict_2 = conv(data.x_dict, data.edge_index_dict,
                      data_alt_time.time_diff_dict)
    author_out_2 = out_dict_2['author']

    assert not torch.allclose(author_out_1, author_out_2)


def test_rte_zero_time_diff():
    """Tests that a zero time difference produces a different output.

    This test ensures that the output of the HGTConv layer with RTE is
    different when given zero time differences compared to when RTE is
    set to false.
    """
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    uea_edge = data['university', 'employs', 'author']
    uea_edge.edge_index = get_random_edge_index(10, 4, 15)
    uea_edge.time_diff = torch.zeros(uea_edge.num_edges, dtype=torch.long)

    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)
    awp_edge.time_diff = torch.zeros(awp_edge.num_edges, dtype=torch.long)

    metadata = data.metadata()
    torch.manual_seed(42)
    conv_with_rte = HGTConv(-1, 64, metadata, heads=2, use_RTE=True)

    out_dict_zero = conv_with_rte(data.x_dict, data.edge_index_dict,
                                  data.time_diff_dict)
    author_out_zero = out_dict_zero['author']

    torch.manual_seed(42)
    conv_without_rte = HGTConv(-1, 64, metadata, heads=2, use_RTE=False)
    out_dict_without_rte = conv_without_rte(data.x_dict, data.edge_index_dict)
    author_out_without_rte = out_dict_without_rte['author']

    assert not torch.allclose(author_out_zero, author_out_without_rte)


def test_rte_raises_error_if_time_is_missing():
    """Tests that a ValueError is raised if RTE is on but no time is given."""
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 16)

    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)

    metadata = data.metadata()
    conv = HGTConv(-1, 32, metadata, heads=2, use_RTE=True)

    with pytest.raises(ValueError, match="RTE enabled, but no"):
        conv(data.x_dict, data.edge_index_dict)


def test_rte_warns_if_time_is_provided_but_unused():
    """Tests that a warning is raised if time is given but RTE deactivated."""
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 16)
    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)
    awp_edge.time_diff = torch.randint(0, 100, (awp_edge.num_edges, ))

    metadata = data.metadata()
    conv = HGTConv(-1, 32, metadata, heads=2, use_RTE=False)

    with pytest.warns(UserWarning, match="'use_RTE' is False, but"):
        conv(data.x_dict, data.edge_index_dict, data.time_diff_dict)


def test_rte_raises_error_if_time_key_is_missing():
    """Tests ValueError is raised if time for one edge type is missing."""
    data = HeteroData()
    data['author'].x = torch.randn(4, 16)
    data['paper'].x = torch.randn(6, 32)
    data['university'].x = torch.randn(10, 32)

    uea_edge = data['university', 'employs', 'author']
    uea_edge.edge_index = get_random_edge_index(10, 4, 15)
    uea_edge.time_diff = torch.randint(0, 100, (uea_edge.num_edges, ))

    awp_edge = data['author', 'writes', 'paper']
    awp_edge.edge_index = get_random_edge_index(4, 6, 20)

    metadata = data.metadata()
    torch.manual_seed(42)

    conv = HGTConv(-1, 32, metadata, heads=2, use_RTE=True)

    with pytest.raises(ValueError, match="'time_diff' missing for edge type"):
        conv(data.x_dict, data.edge_index_dict, data.time_diff_dict)


def test_hgt_conv_rte_behavioral():
    """Tests if HGTConv with RTE can learn a purely time-dependent rule.

    Each 'source' node has two outgoing edges. The edge with the smaller
    `time_diff` is labeled as correct (1).

    The test asserts that the model with `use_RTE=True` successfully
    learns this rule (high accuracy), while the model with `use_RTE=False`
    fails (accuracy near random chance of 0.5).
    """
    num_source_nodes = 50
    data = HeteroData()
    data['source'].x = torch.randn(num_source_nodes, 16)
    data['target'].x = torch.randn(num_source_nodes * 2, 16)

    source_indices = []
    target_indices = []
    time_list = []
    label_list = []

    for i in range(num_source_nodes):
        target1 = i * 2
        target2 = i * 2 + 1

        identical_target_features = torch.randn(1, 16)
        data['target'].x[target1] = identical_target_features
        data['target'].x[target2] = identical_target_features

        # Randomly decide which target receives the "fast" edge
        if torch.rand(1) > 0.5:
            # In this case, the edge to target1 is the faster one
            time_for_target1 = 5.0
            label_for_target1 = 1

            time_for_target2 = 50.0
            label_for_target2 = 0
        else:
            # In this case, the edge to target2 is the faster one
            time_for_target1 = 50.0
            label_for_target1 = 0

            time_for_target2 = 5.0
            label_for_target2 = 1

        source_indices.extend([i, i])
        target_indices.extend([target1, target2])
        time_list.extend([time_for_target1, time_for_target2])
        label_list.extend([label_for_target1, label_for_target2])

    edge_index = torch.tensor([source_indices, target_indices])
    data['source', 'to', 'target'].edge_index = edge_index
    data['source', 'to', 'target'].time_diff = torch.tensor(time_list)
    data['source', 'to', 'target'].y = torch.tensor(label_list)

    data['target', 'rev_to', 'source'].edge_index = edge_index.flip(0)
    data['target', 'rev_to', 'source'].time_diff = torch.zeros(len(time_list))

    metadata = data.metadata()

    class HGTEdgeClassifier(torch.nn.Module):
        def __init__(self, out_channels, use_rte=True):
            super().__init__()
            self.conv = HGTConv(-1, out_channels, metadata, heads=2,
                                use_RTE=use_rte)
            self.classifier = Linear(out_channels * 2, 2)

        def forward(self, x_dict, edge_index_dict, edge_time_diff_dict,
                    edge_label_index):
            x_dict = self.conv(x_dict, edge_index_dict, edge_time_diff_dict)
            src_emb = x_dict['source'][edge_label_index[0]]
            dst_emb = x_dict['target'][edge_label_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
            return self.classifier(edge_emb)

    def train_and_test(use_rte):
        torch.manual_seed(42)
        model = HGTEdgeClassifier(out_channels=16, use_rte=use_rte)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        args = [
            data.x_dict, data.edge_index_dict, data.time_diff_dict,
            data['source', 'to', 'target'].edge_index
        ]
        edge_data = data['source', 'to', 'target']

        for _ in range(20):
            optimizer.zero_grad()

            if not use_rte:
                with pytest.warns(UserWarning, match="'use_RTE' is False"):
                    logits = model(*args)
            else:
                logits = model(*args)

            loss = criterion(logits, edge_data.y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if not use_rte:
                with pytest.warns(UserWarning, match="'use_RTE' is False"):
                    pred = model(*args).argmax(dim=-1)
            else:
                pred = model(*args).argmax(dim=-1)

            return (pred == edge_data.y).float().mean().item()

    acc_with_rte = train_and_test(use_rte=True)
    assert acc_with_rte >= 0.95

    acc_without_rte = train_and_test(use_rte=False)
    assert acc_without_rte <= 0.6


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    num_nodes, num_edges = 30_000, 300_000
    x_dict = {
        'paper': torch.randn(num_nodes, 64, device=args.device),
        'author': torch.randn(num_nodes, 64, device=args.device),
    }
    edge_index_dict = {
        ('paper', 'to', 'paper'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
        ('author', 'to', 'paper'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
        ('paper', 'to', 'author'):
        torch.randint(num_nodes, (2, num_edges), device=args.device),
    }

    conv = HGTConv(
        in_channels=64,
        out_channels=64,
        metadata=(list(x_dict.keys()), list(edge_index_dict.keys())),
        heads=4,
    ).to(args.device)

    benchmark(
        funcs=[conv],
        args=(x_dict, edge_index_dict),
        num_steps=10 if args.device == 'cpu' else 100,
        num_warmups=5 if args.device == 'cpu' else 50,
        backward=False,
    )
