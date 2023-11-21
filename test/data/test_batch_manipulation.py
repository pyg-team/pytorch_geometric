import torch

from torch_geometric.data import Batch, Data

device = torch.device("cpu")


def test_batch_set_attr():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size, ))
    x_list = [torch.rand(n, 3).to(device) for n in nodes_v]

    batch_list = [Data(x=x) for x in x_list]
    batch_truth = Batch.from_data_list(batch_list)

    batch_truth.batch

    batch = Batch.from_empty([g.num_nodes for g in batch_list])

    batch.set_attr("x", torch.vstack(x_list))

    compare(batch_truth, batch)


def test_batch_set_edge_index():
    batch_size = 4
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size, )).to(device)
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    batch_mod = Batch.from_data_list([Data(x=x) for x in x_list])
    batch_mod2 = batch_mod.clone()

    for _ in range(3):
        edges_per_graph = torch.cat([
            torch.randint(1, num_nodes, size=(1, )).to(device)
            for num_nodes in nodes_v
        ]).to(device)
        edges_list = [
            torch.vstack([
                torch.randint(0, num_nodes, size=(num_edges, )),
                torch.randint(0, num_nodes, size=(num_edges, )),
            ]).to(device)
            for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
        ]

        batch_list = [
            Data(x=x, edge_index=edges)
            for x, edges in zip(x_list, edges_list)
        ]
        batch_truth = Batch.from_data_list(batch_list)

        batchidx_per_edge = torch.cat([
            torch.ones(num_edges).long().to(device) * igraph
            for igraph, num_edges in enumerate(edges_per_graph)
        ])
        batch_mod.set_edge_index(torch.hstack(edges_list), batchidx_per_edge)
        batch_mod2.set_edge_index(edges_list)

        compare(batch_truth, batch_mod)
        for dt, dn in zip(batch_truth.to_data_list(),
                          batch_mod.to_data_list()):
            assert torch.allclose(dt.x, dn.x)
            assert torch.allclose(dt.edge_index, dn.edge_index)

        compare(batch_truth, batch_mod2)
        for dt, dn in zip(batch_truth.to_data_list(),
                          batch_mod2.to_data_list()):
            assert torch.allclose(dt.x, dn.x)
            assert torch.allclose(dt.edge_index, dn.edge_index)


def test_batch_set_edge_attr():
    batch_size = 4
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size, )).to(device)
    edges_per_graph = torch.cat([
        torch.randint(1, num_nodes, size=(1, )).to(device)
        for num_nodes in nodes_v
    ])
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack([
            torch.randint(0, num_nodes, size=(num_edges, )),
            torch.randint(0, num_nodes, size=(num_edges, )),
        ]).to(device) for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]
    edge_attr_list = [
        torch.rand(num_edges).to(device) for num_edges in edges_per_graph
    ]

    batch_list = [
        Data(x=x, edge_index=edges, edge_attr=ea)
        for x, edges, ea in zip(x_list, edges_list, edge_attr_list)
    ]
    batch_truth = Batch.from_data_list(batch_list)

    batch_truth.batch
    batch = Batch.from_empty([g.num_nodes for g in batch_list])
    batch.set_attr("x", torch.vstack(x_list))

    batchidx_per_edge = torch.cat([
        torch.ones(num_edges).to(device).long() * igraph
        for igraph, num_edges in enumerate(edges_per_graph)
    ])
    batch.set_edge_index(torch.hstack(edges_list), batchidx_per_edge)
    batch.set_attr("edge_attr", torch.hstack(edge_attr_list), "edge")
    compare(batch_truth, batch)


def test_batch_add_graph_attr():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size, )).to(device)
    x_list = [torch.rand(n, 3).to(device) for n in nodes_v]
    graph_attr_list = torch.rand(batch_size).to(device)

    batch_list = [Data(x=x, ga=ga) for x, ga in zip(x_list, graph_attr_list)]
    batch_truth = Batch.from_data_list(batch_list)

    batch_truth.batch

    batch = Batch.from_empty([g.num_nodes for g in batch_list])

    batch.set_attr("x", torch.vstack(x_list))
    batch.set_attr("ga", graph_attr_list, "graph")
    compare(batch_truth, batch)


def test_from_batch_list():
    batch_size = 12
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size, )).to(device)
    edges_per_graph = torch.cat([
        torch.randint(1, num_nodes, size=(1, )).to(device)
        for num_nodes in nodes_v
    ])
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack([
            torch.randint(0, num_nodes, size=(num_edges, )),
            torch.randint(0, num_nodes, size=(num_edges, )),
        ]).to(device) for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]
    edge_attr_list = [
        torch.rand(num_edges).to(device) for num_edges in edges_per_graph
    ]
    graph_attr_list = torch.rand(batch_size).to(device)

    batch_list = [
        Data(x=x, edge_index=edges,
             edge_attr=ea, ga=ga) for x, edges, ea, ga in zip(
                 x_list, edges_list, edge_attr_list, graph_attr_list)
    ]
    batch_truth = Batch.from_data_list(batch_list)
    batch = Batch.from_batch_list([
        Batch.from_data_list(batch_list[:3]),
        Batch.from_data_list(batch_list[3:5]),
        Batch.from_data_list(batch_list[5:7]),
        Batch.from_data_list(batch_list[7:]),
    ])

    compare(batch_truth, batch)


def test_batch_slice():
    batch_size = 9
    node_range = (3, 8)
    nodes_v = torch.randint(*node_range, (batch_size, )).to(device)
    edges_per_graph = torch.cat([
        torch.randint(1, num_nodes, size=(1, )).to(device)
        for num_nodes in nodes_v
    ])
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack([
            torch.randint(0, num_nodes, size=(num_edges, )),
            torch.randint(0, num_nodes, size=(num_edges, )),
        ]).to(device) for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]
    edge_attr_list = [
        torch.rand(num_edges).to(device) for num_edges in edges_per_graph
    ]
    graph_attr_list = torch.rand(batch_size, 1, 5).to(device)

    batch_list = [
        Data(x=x, edge_index=edges,
             edge_attr=ea, ga=ga) for x, edges, ea, ga in zip(
                 x_list, edges_list, edge_attr_list, graph_attr_list)
    ]
    bslice = torch.FloatTensor(batch_size).uniform_() > 0.4
    batch_full = Batch.from_data_list(batch_list)
    batch_truth = Batch.from_data_list(
        [batch_list[e] for e in bslice.nonzero().squeeze()])
    batch_new = batch_full[bslice]
    compare(batch_truth, batch_new)


def compare(ba: Batch, bb: Batch):
    if set(ba.keys()) != set(bb.keys()):
        raise Exception()
    assert (ba.batch == bb.batch).all()
    assert (ba.ptr == bb.ptr).all()
    for k in ba.keys():
        try:
            rec_comp(ba[k], bb[k])
        except Exception as e:
            raise Exception(
                f"Batch comparison failed tensor for key {k}.") from e
        if k in ba._slice_dict or k in bb._slice_dict:
            try:
                rec_comp(ba._slice_dict[k], bb._slice_dict[k])
            except Exception as e:
                raise Exception(
                    f"Batch comparison failed _slice_dict for key {k}.") from e
        if k in ba._inc_dict or k in bb._inc_dict:
            try:
                rec_comp(ba._inc_dict[k], bb._inc_dict[k])
            except Exception as e:
                raise Exception(
                    f"Batch comparison failed _inc_dict for key {k}.") from e


def rec_comp(a, b):
    if not type(a) is type(b):
        raise Exception()
    if isinstance(a, dict):
        if not set(a.keys()) == set(b.keys()):
            raise Exception()
        for k in a:
            rec_comp(a[k], b[k])
    if isinstance(a, torch.Tensor):
        if not (a == b).all():
            raise Exception()
