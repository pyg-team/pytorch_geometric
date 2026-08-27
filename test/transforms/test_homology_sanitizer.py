import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import HomologySanitizer


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_trigger_removal():
    # Build a tree (Betti-1 = 0)
    tree_edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    # Attach a 5-clique trigger (nodes 5-9)
    clique_edges = []
    for i in range(5, 10):
        for j in range(i + 1, 10):
            clique_edges.append([i, j])
            clique_edges.append([j, i])
    clique_edges_t = torch.tensor(clique_edges).t()
    edge_index = torch.cat([tree_edges, clique_edges_t], dim=1)
    data = Data(edge_index=edge_index, num_nodes=10)

    transform = HomologySanitizer(threshold=2, ego_hops=2)
    out = transform(data)

    # Clique edges should be removed; tree edges should remain
    out_set = {tuple(e) for e in out.edge_index.t().tolist()}
    tree_set = {tuple(e) for e in tree_edges.t().tolist()}

    assert tree_set.issubset(out_set), "Tree edges were incorrectly removed"

    # None of the clique edges should remain (both endpoints anomalous)
    for i in range(5, 10):
        for j in range(i + 1, 10):
            assert (i, j) not in out_set
            assert (j, i) not in out_set


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_clean_graph():
    # Sparse ER graph with low cycle density
    torch.manual_seed(42)
    num_nodes = 50
    p = 0.05
    edge_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < p:
                edge_list.append([i, j])
    if len(edge_list) == 0:
        edge_list = [[0, 1]]
    edge_index = torch.tensor(edge_list).t()
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    transform = HomologySanitizer(threshold=2, community_cycle_ratio=0.5)
    out = transform(data)

    # With a high threshold, no edges should be removed from a sparse graph
    assert out.edge_index.shape[1] == data.edge_index.shape[1]


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_community_isolation():
    # Two disconnected components: tree + dense trigger
    tree = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    trigger = []
    for i in range(5, 10):
        for j in range(i + 1, 10):
            trigger.extend([[i, j], [j, i]])
    trigger = torch.tensor(trigger).t()
    edge_index = torch.cat([tree, trigger], dim=1)
    data = Data(edge_index=edge_index, num_nodes=10)

    transform = HomologySanitizer(threshold=2)
    out = transform(data)

    # Tree component should be untouched
    out_set = {tuple(e) for e in out.edge_index.t().tolist()}
    tree_set = {tuple(e) for e in tree.t().tolist()}
    assert tree_set.issubset(out_set)

    # Trigger component should be pruned
    for i in range(5, 10):
        for j in range(i + 1, 10):
            assert (i, j) not in out_set


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_edge_attr():
    tree_edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    clique_edges = []
    for i in range(5, 10):
        for j in range(i + 1, 10):
            clique_edges.append([i, j])
            clique_edges.append([j, i])
    clique_edges_t = torch.tensor(clique_edges).t()
    edge_index = torch.cat([tree_edges, clique_edges_t], dim=1)
    edge_attr = torch.arange(edge_index.shape[1], dtype=torch.float)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=10)

    transform = HomologySanitizer(threshold=2)
    out = transform(data)

    # Edge attr should be filtered in sync with edge_index
    assert out.edge_attr.shape[0] == out.edge_index.shape[1]

    # Verify tree edge attributes preserved by value matching
    out_edge_dict = {}
    for i in range(out.edge_index.shape[1]):
        src = int(out.edge_index[0, i])
        dst = int(out.edge_index[1, i])
        out_edge_dict[(src, dst)] = float(out.edge_attr[i])

    in_edge_dict = {}
    for i in range(data.edge_index.shape[1]):
        src = int(data.edge_index[0, i])
        dst = int(data.edge_index[1, i])
        in_edge_dict[(src, dst)] = float(data.edge_attr[i])

    for e in tree_edges.t().tolist():
        assert out_edge_dict[tuple(e)] == in_edge_dict[tuple(e)]


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_fast_path():
    # Synthetic graph larger than fast_path_limit
    num_nodes = 6000
    edge_index = torch.tensor([[0, 1], [1, 0]])
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    transform = HomologySanitizer(threshold=2, fast_path_limit=5000)
    out = transform(data)

    # Fast path should not crash; with only one edge no pruning occurs
    assert out.edge_index.shape[1] == 2


@withPackage('networkx', 'scipy')
def test_homology_sanitizer_repr():
    transform = HomologySanitizer(threshold=3, ego_hops=1,
                                  community_cycle_ratio=0.5)
    assert str(transform) == ('HomologySanitizer(threshold=3, ego_hops=1, '
                              'community_cycle_ratio=0.5)')
